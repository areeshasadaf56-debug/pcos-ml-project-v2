# Wellness Saheli — Product Vision & Implementation Roadmap

This document merges the long-term product vision with what is **already
built** (per REQUIREMENTS.md and the current codebase), so any future
session — human or Claude — knows exactly what exists vs. what's new work.

---

## 1. Vision Summary

Wellness Saheli = cycle tracker + PCOS/endo screener + AI health
companion + diary, unified into one product where the AI is the
central, most visible feature — not a hidden tab. Target: feel like a
private, intelligent, safe companion, not a generic chatbot or generic
period tracker.

---

## 2. Already Implemented (do not rebuild these)

| Area | Status |
|---|---|
| Sign up / sign in / reset password | Done (bcrypt, FastAPI) |
| Anonymous device-ID profile (no account required) | Done |
| Cycle data entry, current day/phase calculation | Done |
| Ovulation & fertility window estimate | Done |
| Symptom/mood logging | Done |
| PCOS: informational content | Done |
| PCOS: 22-field ML risk screening (RandomForest) | Done |
| Protection: method browser + effectiveness data | Done |
| Protection: eligibility checker (WHO MEC-based, 12/256 conditions rated) | Done, partial dataset |
| Protection: Additional Info + How-to-use tabs | Done |
| AI check-in: conversational, tool-calling insight extraction | Done |
| AI check-in: crisis-language detection (deterministic) | Done |
| AI check-in: medical-emergency detection (backend) | Done, not yet read by Flutter UI |
| AI check-in: graceful fallback with no API key | Done |
| Health profile: server-side sync (SQLite, JSON blob per user) | Done |
| Health Diary: unified timeline (PCOS/contraception/eligibility/AI log) | Done |
| Endometriosis: info + screening tab | Done |
| Settings: cycle data, name, privacy toggle | Done |
| Learn tab: educational content | Done |

**Do not redo these from scratch.** Extend/improve them per the gaps below.

---

## 3. Gap Analysis (vision vs. current state)

### 3.1 AI Prominence
- **Gap:** AI is one tab among several; vision wants it to be the app's
  visible center from first launch.
- **Action:** Redesign home/cycle screen to lead with an AI entry card
  ("How are you feeling today?"), not a small toggle.

### 3.2 AI Depth & Memory
- **Gap:** Current chat.py extracts structured data per turn but has
  no long-term pattern recognition ("your cramps have been worse the
  last 3 cycles").
- **Action:** Add a lightweight pattern-summary step: before each chat
  call, pull last N cycle/symptom entries from the profile and include
  a short factual summary in the system prompt context (already
  supported by `profile_context` param in chat.py — needs the Flutter
  side to actually populate and send it with real history, not just
  the live message).

### 3.3 Voice & Image Input
- **Gap:** Not built. Voice-to-text and image upload for the AI chat.
- **Action:** New scope. Flutter: `speech_to_text` package for voice
  input (transcribe locally, send as text — do not send raw audio to
  the backend, keeps `/chat`'s contract unchanged). Image upload is
  larger scope (needs multipart upload endpoint + vision-capable model
  call) — treat as a separate later phase, not bundled with voice.

### 3.4 Chat History Panel
- **Gap:** No visible "previous conversations" side panel.
- **Action:** Flutter-only — list `conversation_log` entries from the
  health profile (already stored) in a drawer/section on the AI screen.
  No backend change needed; the data already exists in
  `HealthProfile.conversation_log`.

### 3.5 Notifications from Mood/Symptom Patterns
- **Gap:** Not built.
- **Action:** New scope, needs care (see Safety Adjustments below).
  Requires: local notification scheduling (`flutter_local_notifications`)
  triggered by a simple rule engine reading recent mood/symptom logs —
  **not** a claim of continuous AI "training" on the user, just a
  scheduled check against stored data.

### 3.6 Google Sign-In + "Login Once" Persistence
- **Gap:** Not built (current auth is email/password only).
- **Action:** Add `google_sign_in` Flutter package + a
  `/signin_google` backend endpoint that verifies the Google ID token
  server-side and creates/looks up the account by verified email.
  "Login once" = store the signed-in state in secure local storage;
  only clear it on explicit logout, matching normal app behavior.

### 3.7 Splash Screen
- **Gap:** Need to confirm current state — not reviewed this session.
- **Action:** Standard Flutter splash (app icon/logo, brief), routes to
  sign-in if no stored session, or home if session exists.

### 3.8 Deep, Therapist-style Intake Questions
- **Status:** Partially done — `chat.py`'s SYSTEM_PROMPT already asks
  about financial stress, relationship difficulty, family conflict,
  loneliness, gently and with stated reasons (see REQUIREMENTS.md AI-01
  through AI-04).
- **Safety-required limit:** see Section 4 — will NOT be extended to
  probe sexual activity/abuse history as a standard/expected line of
  questioning.

### 3.9 Multilingual Architecture (Urdu, future)
- **Gap:** Not built (this was the paused Urdu/English toggle work).
- **Action:** Use Flutter's `flutter_localizations` + `.arb` files from
  the start for any new UI text, so retrofitting isn't needed later —
  matches the vision doc's own instruction not to hardcode text.

---

## 4. Safety-Required Adjustments to the Vision

A few items in the vision need a boundary before implementation —
stated plainly, not to block the idea, but because the unmodified
version would cross real safety lines:

1. **The AI must not treat probing sexual activity, marital/abuse
   status as a standard, expected part of every conversation**, even
   framed as "like a therapist." It may ask general emotional/life
   context questions (already implemented) and may respond supportively
   if a user *raises* something sensitive herself, but it must not
   pursue disclosure of sexual activity or abuse as a routine
   assessment step. This protects both the user (unsolicited probing
   of sexual history is not appropriate for a screening tool) and the
   product from a serious trust/safety failure.
2. **No claim that the AI is "trained on" a user's personal history.**
   It should *use* stored profile data as context per-request (already
   the architecture), not be described as continuously fine-tuned on
   individual users — that's both inaccurate and a privacy overreach.
3. **Notifications referencing mood ("you're feeling sad lately")
   must stay observational, never diagnostic, and must be easy to turn
   off** — matches the existing `ai_can_access_diary` / privacy-settings
   pattern already in the data model.
4. **PCOS/endometriosis assessments give a risk indication, never a
   diagnosis** — already the correct framing in the existing PCOS
   screen and REQUIREMENTS.md; keep this exact language pattern when
   building the endometriosis assessment out further.

---

## 5. Phased Roadmap

**Phase 1 — AI Prominence Redesign (Flutter only, no backend change)**
Home/cycle screen redesign: AI entry card, "how are you feeling today"
prompt, contextual suggestions pulling from existing profile data.

**Phase 2 — AI Memory Wiring**
Populate and send `profile_context` from Flutter to `/chat` with real
recent cycle/symptom/mood data (backend already accepts this field).
Add simple pattern-summary logic before sending to the model.

**Phase 3 — Chat UX Upgrade**
Voice input (local transcription), chat history panel from existing
`conversation_log` data. Image upload deferred to a later phase.

**Phase 4 — Auth Upgrade**
Google Sign-In, persistent "login once" session, splash screen.

**Phase 5 — Notifications**
Local notification rule engine off existing mood/symptom logs, with
clear opt-out.

**Phase 6 — Endometriosis Assessment Completion**
Build out the structured questionnaire (mirrors PCOS assessment
pattern) with the same non-diagnostic risk-level output.

**Phase 7 — Localization**
Urdu/English toggle, `.arb`-based architecture (resumes the paused work
from earlier this session).

Each phase should be committed and tested independently — do not start
Phase 2 before Phase 1 is verified working, per the "don't overengineer
everything at once" principle already in this project's own history
(see: the venv/git/deployment issues earlier this session, all caused
by parallel unfinished work).

---

## 6. Additional Recommendations (beyond the original vision)

- **Rate-limit `/chat`** once voice/image input is added — larger
  inputs make abuse/cost overrun more likely on a free-tier AI plan.
- **Version the health profile schema** (a `schema_version` field)
  once Phase 2–3 start adding new profile fields, so old app installs
  don't send/receive mismatched shapes.
- **Add the medical-emergency UI branch** (Flutter reading
  `medical_emergency_concern`, already returned by the backend) before
  adding voice/image input — it's a small, already-half-done safety
  feature that's currently sitting incomplete (see REQUIREMENTS.md
  known gap).