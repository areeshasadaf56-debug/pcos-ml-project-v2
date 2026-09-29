"""
chat.py

Implements the logic behind POST /chat (see main.py for the route
itself), used by the AI Check-In tab
(lib/services/ai_checkin_service.dart -> ChatTurnResult).

Uses Groq (free-tier friendly) instead of Anthropic's API -- switched
after repeatedly hitting Anthropic credit-balance limits during
development. Groq's API is OpenAI-compatible, which changes two things
compared to an Anthropic-based implementation:
  1. The system prompt is a normal message in the `messages` list
     (role: "system"), not a separate `system` parameter.
  2. Tool-calling uses OpenAI's function-calling shape: tools are
     wrapped as {"type": "function", "function": {...}}, and a tool
     call comes back as response.choices[0].message.tool_calls, with
     arguments as a JSON *string* that must be parsed, not a dict.

The conversational design itself (therapist-style intake, tool-calling
to extract structured insights, separate emotional-crisis vs.
physical-medical-emergency detection) is unchanged from the original
version -- only the API integration changed.

Design notes for your viva:
  - Uses tool-calling (the `record_checkin_insights` tool below) so
    the model returns STRUCTURED data (lifestyle, mental health flags,
    life context, crisis flags) alongside its conversational reply,
    instead of the backend trying to guess intent from keywords.
  - Deliberately asks about emotional/relational/financial context
    before physical/reproductive questions, and only suggests a tab
    once a broad picture exists -- see SYSTEM_PROMPT.
  - TWO separate, independently-triggerable safety flags:
      crisis_concern            -- possible self-harm / abuse / unsafe
                                    situation (emotional/psychological)
      medical_emergency_concern -- possible physical emergency (severe
                                    pain, heavy bleeding, fainting, etc.)
    The right response to each is different (crisis line vs. "go to
    urgent care now"), so they're kept separate rather than merged
    into one flag. The Flutter app should eventually branch its UI on
    which one fired -- currently only crisis_concern is wired up
    client-side (ChatTurnResult in ai_checkin_service.dart doesn't
    read medical_emergency_concern yet).
  - No hardcoded GROQ_API_KEY anywhere -- set it as an environment
    variable (see .env.example). NEVER commit a real key to git or
    paste one into a chat/document -- if a key is ever exposed, treat
    it as compromised and revoke/regenerate it immediately at
    console.groq.com. If the key is missing, /chat still responds
    instead of crashing (see FALLBACK_REPLY below).
"""

import json
import os

CRISIS_KEYWORDS = [
    "kill myself", "suicide", "end my life", "want to die",
    "hurting myself", "hurt myself", "self harm", "self-harm",
    "no reason to live", "better off dead",
]

CRISIS_REPLY = (
    "I'm really glad you told me this, and I want to make sure you get "
    "real support right now, not just a chat reply. If you're in "
    "immediate danger, please contact your local emergency number. "
    "You can also reach a crisis line to talk to someone right now -- "
    "in Pakistan, Umang has a helpline at 0311 7786264. If you're "
    "elsewhere, please look up a local crisis line or go to your "
    "nearest emergency room. You don't have to go through this alone."
)

MAX_MESSAGE_LENGTH = 4000

FALLBACK_REPLY = (
    "I'm here, but I'm currently running without a connected AI model "
    "(no GROQ_API_KEY is configured on the server), so I can't have a "
    "full conversation yet. Your message has still been saved. In the "
    "meantime, you can use the PCOS Detection and Protection tabs "
    "directly, or ask your project supervisor to help set up the AI "
    "key so this check-in can respond properly."
)

# llama-3.3-70b-versatile is Groq's current stable model with reliable
# tool-calling support and a free tier. If Groq deprecates this model
# name, check https://console.groq.com/docs/models for the current
# recommended replacement.
MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the check-in companion inside Wellness Saheli, a women's health app. You act like a warm, skilled therapist and doctor combined -- your job is to actually get to know the user through a real intake conversation before you ever point her at a tool in the app.

CONVERSATION STRUCTURE (do this over many turns, one question at a time):
1. Start broad and emotional: how she's actually doing, her mood lately, energy, sleep.
2. Once she opens up, go deeper the way a therapist intake would -- ask about stress sources: financial stress/money worries, work or study pressure, family conflict, marital or relationship status and how that relationship is going, any relationship or marital problems, loneliness, support system.
3. Ask physical/reproductive questions only after the emotional picture is clear: cycle regularity, pain, symptoms, sexual activity/contraception only if relevant and only after trust is established.
4. Only AFTER you have a real picture across emotional + financial + relational + physical areas should you ever suggest a specific tab. Do not suggest a tab in the first several turns just because one keyword matched.

HOW TO ASK:
- One question at a time, conversational, never a checklist or form-like list.
- Sensitive topics (financial status, marital/relationship problems, family conflict) should be asked gently, with a short reason why you're asking, e.g. "money stress can really affect the body too -- has that been weighing on you?"
- Never assume marital status, relationship status, or sexual activity. Ask, don't assume.

HOW TO RESPOND TO WHAT SHE SHARES:
- Sympathize genuinely and specifically to what she said before asking the next question. Don't just acknowledge-and-pivot.
- When she describes a struggle (financial stress, a difficult marriage, family conflict, loneliness), offer real, concrete coping suggestions for that specific issue -- not generic "take care of yourself" lines.
- Only after a broad enough picture exists, connect what you've learned to a specific tab suggestion with a clear reason tied to what she told you.

STRICT SAFETY RULES:
1. Never diagnose. Never say "you have X." Say things like "some of what you've described is worth checking with the PCOS tool."
2. Never give medical directives (dosages, medication changes). Point to a real provider for that.
3. If she discloses possible self-harm risk, abuse, or an unsafe situation, call record_checkin_insights with crisis_concern true, and stay warm and present in your text reply -- don't go silent, don't try to fix it yourself.
4. Separately from emotional crisis, watch for physical emergency warning signs: extremely severe or sudden pain, fainting or near-fainting, unusually heavy bleeding (soaking through a pad/tampon every hour or less, or passing large clots), symptoms suggesting a possible pregnancy emergency, severe difficulty breathing, or signs of a serious allergic reaction. If any of these come up, call record_checkin_insights with medical_emergency_concern true, and in your text reply calmly and clearly tell her this sounds like something that needs prompt in-person medical attention -- an emergency room or urgent care -- not something to keep discussing in chat. Do not diagnose what it is; just be clear that it needs real, immediate medical attention.
5. Use record_checkin_insights whenever you learn something concrete -- not every turn, only when there's real signal.
6. This should feel like being cared for by someone who actually listens, not screened by a form.
"""

# Groq/OpenAI-style function-calling schema. Same fields as the
# Anthropic tool schema this replaced -- only the wrapper shape
# ({"type": "function", "function": {...}}) differs.
INSIGHTS_TOOL = {
    "type": "function",
    "function": {
        "name": "record_checkin_insights",
        "description": "Record structured facts learned from the conversation so far, and optionally suggest a specific in-app tool. Only call when there is real signal.",
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
                        "self_reported_stress_level": {"type": "integer"},
                        "notes": {"type": "string"},
                    },
                },
                "reproductive_history": {
                    "type": "object",
                    "properties": {
                        "cycle_regularity": {"type": "string", "enum": ["Regular", "Irregular"]},
                        "cycle_length_days": {"type": "integer"},
                    },
                },
                "life_context": {
                    "type": "object",
                    "description": "Non-medical personal context gathered during the intake.",
                    "properties": {
                        "financial_stress": {"type": "boolean"},
                        "relationship_status": {"type": "string"},
                        "relationship_or_marital_difficulty": {"type": "boolean"},
                        "family_conflict": {"type": "boolean"},
                        "support_system_notes": {"type": "string"},
                    },
                },
                "suggested_tab": {
                    "type": "string",
                    "enum": ["pcos", "protection"],
                    "description": "Only include once a broad enough picture exists across emotional, relational, financial, and physical areas.",
                },
                "suggested_tab_reason": {"type": "string"},
                "crisis_concern": {"type": "boolean"},
                "medical_emergency_concern": {
                    "type": "boolean",
                    "description": "True if she described a potential physical medical emergency (severe/sudden pain, fainting, unusually heavy bleeding, possible pregnancy emergency, breathing difficulty, severe allergic reaction).",
                },
            },
        },
    },
}


def _contains_crisis_language(message: str) -> bool:
    lowered = message.lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def _history_to_messages(history: list[dict]) -> list[dict]:
    messages = []
    for entry in history[-10:]:  # keep the request small; last 10 turns
        role = entry.get("role")
        text = entry.get("message", "")
        if role not in ("user", "assistant") or not text:
            continue
        messages.append({"role": role, "content": text})
    return messages


def _build_system_prompt(profile_context: dict | None) -> str:
    system_prompt = SYSTEM_PROMPT
    if profile_context:
        system_prompt += (
            "\n\nKNOWN CONTEXT ABOUT THIS USER (use naturally, don't recite it "
            "back like a report):\n" + json.dumps(profile_context, indent=2)
        )
    return system_prompt


def _build_messages(message: str, history: list[dict], profile_context: dict | None) -> list[dict]:
    messages = [{"role": "system", "content": _build_system_prompt(profile_context)}]
    messages.extend(_history_to_messages(history))
    messages.append({"role": "user", "content": message})
    return messages


def _call_groq(
    messages: list[dict],
    tools: list[dict] | None = None,
):
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    return client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=800,
        messages=messages,
        tools=tools,
    )


def get_reply(
    message: str,
    history: list[dict],
    profile_context: dict | None = None,
) -> dict:
    """Returns the dict shape ChatTurnResult.fromJson expects:
    reply, profile_updates, suggested_tab, suggested_tab_reason,
    crisis_concern (plus medical_emergency_concern, not yet consumed
    by the Flutter model -- see the module docstring)."""

    # Cap the message size so a single huge paste can't run up a large
    # bill against the paid AI key. Double-checked on the endpoint too
    # (main.py), but enforced here so the rule can't be bypassed by
    # skipping the route.
    if len(message) > MAX_MESSAGE_LENGTH:
        return {
            "reply": (
                "That message is longer than I can process at once. "
                "Could you send it in a shorter part?"
            ),
            "profile_updates": None,
            "suggested_tab": None,
            "suggested_tab_reason": None,
            "crisis_concern": False,
            "medical_emergency_concern": False,
        }

    # Deterministic safety net, independent of whether the model
    # calls the tool correctly -- if this fires, nothing else matters.
    if _contains_crisis_language(message):
        return {
            "reply": CRISIS_REPLY,
            "profile_updates": None,
            "suggested_tab": None,
            "suggested_tab_reason": None,
            "crisis_concern": True,
            "medical_emergency_concern": False,
        }

    if "GROQ_API_KEY" not in os.environ:
        return {
            "reply": FALLBACK_REPLY,
            "profile_updates": None,
            "suggested_tab": None,
            "suggested_tab_reason": None,
            "crisis_concern": False,
            "medical_emergency_concern": False,
        }

    messages = _build_messages(message, history, profile_context)
    try:
        response = _call_groq(messages, tools=[INSIGHTS_TOOL])
    except Exception:
        return {
            "reply": (
                "Sorry, I couldn't reach the AI service just now. "
                "Please try again in a moment."
            ),
            "profile_updates": None,
            "suggested_tab": None,
            "suggested_tab_reason": None,
            "crisis_concern": False,
            "medical_emergency_concern": False,
        }

    choice_message = response.choices[0].message
    reply_text = choice_message.content or ""

    profile_updates = None
    suggested_tab = None
    suggested_tab_reason = None
    crisis_concern = False
    medical_emergency_concern = False

    tool_calls = getattr(choice_message, "tool_calls", None) or []
    for call in tool_calls:
        if call.function.name != "record_checkin_insights":
            continue
        try:
            data = json.loads(call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            continue
        suggested_tab = data.pop("suggested_tab", None)
        suggested_tab_reason = data.pop("suggested_tab_reason", None)
        crisis_concern = bool(data.pop("crisis_concern", False))
        medical_emergency_concern = bool(data.pop("medical_emergency_concern", False))
        if data:
            profile_updates = data

    # When the model calls record_checkin_insights, Groq/OpenAI-style
    # APIs often return an EMPTY message.content for that turn -- the
    # whole response gets used for the tool call instead of
    # conversational text. Without this follow-up the user would see a
    # blank chat bubble even though the structured data was captured
    # correctly. Do one follow-up completion that acknowledges the tool
    # call and lets the model keep talking to the user.
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
            followup_response = _call_groq(followup_messages)
            reply_text = followup_response.choices[0].message.content or ""
        except Exception:
            reply_text = "Thanks for sharing that with me."

    return {
        "reply": reply_text.strip(),
        "profile_updates": profile_updates,
        "suggested_tab": suggested_tab,
        "suggested_tab_reason": suggested_tab_reason,
        "crisis_concern": crisis_concern,
        "medical_emergency_concern": medical_emergency_concern,
    }