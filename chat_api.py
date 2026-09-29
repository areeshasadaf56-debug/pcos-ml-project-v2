"""
chat_api.py

Powers the AI "personal check-in" chat. Each message from the user is sent
to Groq's chat completions API (OpenAI-compatible format) along with a
system prompt that defines the agent's role, tone, and -- critically --
its safety boundaries (no diagnosis, no medical directives, careful
handling of mental-health disclosures).

The model can reply with plain conversational text AND, when it learns
something concrete (a lifestyle fact, a stress signal, a reason to check
PCOS or Protection), call the `record_checkin_insights` tool to hand back
structured data. The frontend applies that structured data to the user's
HealthProfile via the existing /profile endpoints -- this file never
writes to the database directly, keeping a single write path.

Using Groq instead of a paid API: Groq's free tier is rate-limited
(usage-based) rather than requiring a purchased credit balance, and its
Chat Completions endpoint is OpenAI-compatible, including tool/function
calling -- which is why the request/response shapes below look like
OpenAI's format rather than Anthropic's Messages API, even though the
business logic (system prompt, insights tool, response contract returned
to Flutter) is unchanged from before.

AUTH: the /chat route now requires a valid `Authorization: Bearer
<token>` header (see auth_utils.require_auth). Previously this endpoint
had no auth at all, so anyone who found the URL could run up usage
against your Groq API key for free -- this closes that off, matching
the fact that the app already requires sign-in before this screen is
reachable.

NOTE ON DATA HANDLING: check Groq's current terms of service yourself
before sending real users' health disclosures through any third-party
API -- terms can change, and this app handles sensitive information
(mental health, reproductive health) that deserves that scrutiny.

Setup required on your PythonAnywhere account:
    pip install groq (inside your virtualenv, e.g. saheli-env --
    see "Start a console in this virtualenv" on the Web tab)
    Set an environment variable GROQ_API_KEY with your API key from
    console.groq.com (free signup, no billing required for the free tier).
    PythonAnywhere doesn't have a dedicated env-var box on the Web tab --
    set it directly in your WSGI file instead:
    os.environ["GROQ_API_KEY"] = "..."  (before the `from main_flask
    import app` line), then reload the web app.

Wire into main_flask.py with:
    from chat_api import chat_bp
    app.register_blueprint(chat_bp)
"""

import os
import io
import json
import base64
from flask import Blueprint, request, jsonify
from groq import Groq

from auth_utils import require_auth

# Optional extraction libraries for document attachments. Guarded so
# the whole /chat endpoint doesn't 500 if these haven't been installed
# yet -- see requirements.txt ("pypdf", "python-docx"). Run:
#   pip install pypdf python-docx
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx as docx_lib
except ImportError:
    docx_lib = None

chat_bp = Blueprint("chat", __name__)

# openai/gpt-oss-120b is confirmed (via Groq's /models endpoint) to
# support "tools" -- required for the record_checkin_insights function
# call this file relies on. Groq's free-model lineup changes over time,
# so if this one gets deprecated later, re-check
# https://api.groq.com/openai/v1/models (with your API key) for current
# models whose "supported_features" list includes "tools", and swap the
# name below -- llama-3.3-70b-versatile (used originally) is no longer
# available on this account as of writing.
MODEL_NAME = "openai/gpt-oss-120b"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at console.groq.com, then set it in your "
                "PythonAnywhere WSGI file (os.environ[\"GROQ_API_KEY\"] = ...), "
                "then reload the app."
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are the check-in companion inside Wellness Saheli, a women's health app. Your job is to have a warm, unhurried conversation with the user about how they've been -- their stress levels, sleep, exercise, diet, family or relationship pressures, and reproductive health -- and to gently guide them toward the app's own screening tools (a PCOS likelihood checker, and a WHO-guideline contraception eligibility checker) when something they say suggests it's relevant.

STRICT RULES YOU MUST FOLLOW:
1. You are NOT a doctor and must never diagnose anything. Never say "you have X" or "this means you have X." You can say things like "some of what you're describing is worth checking with the PCOS tool" or "a doctor would be able to tell you more about that."
2. Never give specific medical directives (dosages, what medication to take, whether to stop a medication). Encourage a real healthcare provider for anything like that.
3. Ask ONE question at a time. This is a conversation, not a form -- keep replies short (2-4 sentences), warm, and specific to what the user just said.
4. If the user discloses something suggesting they may be in emotional distress, at risk of self-harm, or in an unsafe situation (e.g. family violence): stay calm and supportive, do NOT try to solve it yourself, and call the record_checkin_insights tool with crisis_concern set to true so the app can surface real crisis resources. Keep talking to them supportively in your text reply too -- don't just go silent.
5. Use the record_checkin_insights tool whenever you learn something concrete and structured (not for every message -- only when there's real signal to record).
6. Only set suggested_tab when there's a genuine, specific reason tied to what the user said (e.g. they mentioned irregular periods + weight changes -> pcos; they mentioned being sexually active without protection -> protection). Don't suggest a tab just to be thorough.
7. Never make the user feel like they're being screened or judged. This should feel like talking to a caring, knowledgeable friend.
"""

# OpenAI/Groq tool-calling format wraps the schema in a "function" object,
# unlike Anthropic's flatter shape -- same fields, different envelope.
INSIGHTS_TOOL = {
    "type": "function",
    "function": {
        "name": "record_checkin_insights",
        "description": (
            "Record structured facts learned from the conversation so far, "
            "and optionally suggest a specific in-app tool to check. Only "
            "call this when there is real, concrete signal to record -- "
            "not on every turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lifestyle": {
                    "type": "object",
                    "properties": {
                        "regular_exercise": {"type": "boolean"},
                        "exercise_frequency": {"type": "string"},
                        "diet_quality": {"type": "string"},
                        "fast_food_frequent": {"type": "boolean"},
                        "average_sleep_hours": {"type": "number"},
                    },
                },
                "mental_health": {
                    "type": "object",
                    "properties": {
                        "self_reported_stress_level": {
                            "type": "integer",
                            "description": "1 (low) to 5 (high), only if the user gave enough info to estimate this",
                        },
                        "notes": {"type": "string"},
                    },
                },
                "reproductive_history": {
                    "type": "object",
                    "properties": {
                        "cycle_regularity": {
                            "type": "string",
                            "enum": ["Regular", "Irregular"],
                        },
                        "cycle_length_days": {"type": "integer"},
                    },
                },
                "suggested_tab": {
                    "type": "string",
                    "enum": ["pcos", "protection"],
                    "description": "Only include this field if there's a genuine, specific reason to suggest one of the app's tools.",
                },
                "suggested_tab_reason": {
                    "type": "string",
                    "description": "One short sentence explaining why, to show the user alongside the suggestion.",
                },
                "crisis_concern": {
                    "type": "boolean",
                    "description": "Set true if the user disclosed anything suggesting emotional crisis, self-harm risk, or an unsafe situation.",
                },
            },
        },
    },
}


def _language_instruction(language):
    """
    Turns the frontend's simple 'en'/'ur' toggle into an instruction
    appended to the system prompt. Since MODEL_NAME is a general LLM
    (not a translation API), it can just be told what language to
    reply in -- no separate translation step needed.
    """
    if language == "ur":
        return (
            "\n\nIMPORTANT: Reply in Urdu (اردو) for this entire "
            "conversation, using natural, warm, conversational Urdu -- "
            "not overly formal or literal. If the user writes to you in "
            "English, still reply in Urdu unless they explicitly ask you "
            "to switch back to English."
        )
    return ""


MAX_ATTACHMENT_CHARS = 6000  # keep extracted text from blowing up the prompt


def _extract_attachment_text(attachment):
    """
    Given {"file_name", "mime_type", "data_base64"} from the frontend,
    returns (description, extracted_text_or_None) to fold into the
    user's message as extra context.

    - Plain text files: decoded directly.
    - PDFs / Word docs: text extracted via pypdf / python-docx, if
      those packages are installed (see requirements.txt).
    - Images: MODEL_NAME is a text-only model, so the image's pixels
      can't actually be read here -- we only pass along the filename
      so the assistant can acknowledge it honestly instead of
      pretending to have looked at it. Swap in a vision-capable Groq
      model later (e.g. a llama-3.2-*-vision variant) to change that.
    """
    file_name = attachment.get("file_name", "attachment")
    mime_type = attachment.get("mime_type", "")
    data_b64 = attachment.get("data_base64")
    if not data_b64:
        return f"[User attached a file: {file_name}]", None

    try:
        raw = base64.b64decode(data_b64)
    except Exception:
        return f"[User attached a file: {file_name} (couldn't be read)]", None

    if mime_type.startswith("image/"):
        return (
            f"[User attached an image: {file_name}. Note: you cannot see "
            f"the contents of this image -- acknowledge it was received "
            f"and ask the user to describe what's in it if it's relevant.]",
            None,
        )

    if mime_type == "application/pdf":
        if PdfReader is None:
            return (
                f"[User attached a PDF: {file_name}, but the server can't "
                f"extract text from it yet -- ask them to paste the "
                f"relevant part as text instead.]",
                None,
            )
        try:
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return f"[User attached a PDF: {file_name}]", text[:MAX_ATTACHMENT_CHARS]
        except Exception:
            return f"[User attached a PDF: {file_name} (couldn't be read)]", None

    if mime_type in (
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        if docx_lib is None:
            return (
                f"[User attached a Word document: {file_name}, but the "
                f"server can't extract text from .doc/.docx yet -- ask "
                f"them to paste the relevant part as text instead.]",
                None,
            )
        try:
            doc = docx_lib.Document(io.BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs)
            return (
                f"[User attached a Word document: {file_name}]",
                text[:MAX_ATTACHMENT_CHARS],
            )
        except Exception:
            return (
                f"[User attached a Word document: {file_name} (couldn't be read)]",
                None,
            )

    # Plain text / anything else: try decoding as UTF-8 text.
    try:
        text = raw.decode("utf-8")
        return f"[User attached a text file: {file_name}]", text[:MAX_ATTACHMENT_CHARS]
    except UnicodeDecodeError:
        return f"[User attached a file: {file_name} (unsupported format)]", None


def _history_to_messages(history):
    """
    Converts the frontend's [{role, message}, ...] list into Groq/OpenAI
    Chat Completions format. Unlike Anthropic's Messages API (which takes
    the system prompt as a separate top-level field), OpenAI-style APIs
    expect the system prompt as the first message in the list -- the
    caller (chat()) prepends that before calling this function's output.
    """
    messages = []
    for entry in history:
        role = entry.get("role")
        text = entry.get("message", "")
        if role not in ("user", "assistant") or not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


@chat_bp.route("/chat", methods=["POST", "OPTIONS"])
@require_auth
def chat():
    """
    Body: {
        "message": "the user's new message",
        "history": [{"role": "user"|"assistant", "message": "..."}, ...],
        "language": "en" | "ur",             # optional, defaults to "en"
        "attachment": {                       # optional
            "file_name": "...",
            "mime_type": "...",
            "data_base64": "..."
        }
    }

    Returns: {
        "reply": "assistant's conversational reply",
        "profile_updates": {...} | null,   # matches record_checkin_insights shape, minus suggested_tab/reason/crisis
        "suggested_tab": "pcos" | "protection" | null,
        "suggested_tab_reason": "..." | null,
        "crisis_concern": bool
    }

    Requires a valid session (Authorization: Bearer <token>) -- see
    module docstring.
    """
    if request.method == "OPTIONS":
        return "", 204

    body = request.get_json(force=True) or {}
    user_message = body.get("message", "").strip()
    history = body.get("history", [])
    language = body.get("language", "en")
    attachment = body.get("attachment")

    attachment_note = None
    attachment_text = None
    if attachment:
        attachment_note, attachment_text = _extract_attachment_text(attachment)

    if not user_message and not attachment_note:
        return jsonify({"detail": "message is required"}), 422

    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"detail": str(e)}), 500

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + _language_instruction(language)}
    ]
    messages.extend(_history_to_messages(history))

    # Fold the attachment (if any) into the user's turn as extra context,
    # rather than a separate message -- keeps the model's understanding
    # of "what the user just sent" as one coherent turn.
    final_user_content = user_message or "(no text, see attachment)"
    if attachment_note:
        final_user_content += f"\n\n{attachment_note}"
    if attachment_text:
        final_user_content += f"\n\nAttached file content:\n{attachment_text}"

    messages.append({"role": "user", "content": final_user_content})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=800,
            tools=[INSIGHTS_TOOL],
            messages=messages,
        )
    except Exception as e:
        return jsonify({"detail": f"AI service error: {e}"}), 502

    choice = response.choices[0]
    reply_text = choice.message.content or ""
    profile_updates = None
    suggested_tab = None
    suggested_tab_reason = None
    crisis_concern = False

    tool_calls = choice.message.tool_calls or []
    for call in tool_calls:
        if call.function.name != "record_checkin_insights":
            continue
        try:
            data = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            data = {}
        suggested_tab = data.pop("suggested_tab", None)
        suggested_tab_reason = data.pop("suggested_tab_reason", None)
        crisis_concern = bool(data.pop("crisis_concern", False))
        # Whatever's left (lifestyle / mental_health / reproductive_history)
        # is the actual profile-shaped update.
        if data:
            profile_updates = data

    # When the model calls record_checkin_insights, Groq/OpenAI-style APIs
    # often return an EMPTY message.content for that turn -- the whole
    # response gets used for the tool call instead of conversational text.
    # Without this fix, the user sees a blank chat bubble even though the
    # structured data above was captured correctly. To guarantee a real
    # reply, do one follow-up completion: acknowledge the tool call in the
    # conversation, then let the model continue talking to the user.
    if not reply_text.strip() and tool_calls:
        followup_messages = list(messages)
        followup_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            followup_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "Recorded.",
                }
            )
        try:
            followup_response = client.chat.completions.create(
                model=MODEL_NAME,
                max_tokens=800,
                messages=followup_messages,
            )
            reply_text = followup_response.choices[0].message.content or ""
        except Exception:
            # If the follow-up call itself fails, fall back to a short
            # acknowledgement rather than showing the user a blank bubble.
            reply_text = "Thanks for sharing that with me."

    return jsonify(
        {
            "reply": reply_text.strip(),
            "profile_updates": profile_updates,
            "suggested_tab": suggested_tab,
            "suggested_tab_reason": suggested_tab_reason,
            "crisis_concern": crisis_concern,
        }
    )