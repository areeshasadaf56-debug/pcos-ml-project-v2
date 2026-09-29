"""
main.py

FastAPI backend for the Wellness Saheli Flutter app.

Exposes:
  /predict                      -- PCOS ML prediction (stateless, public)
  /signup, /signin, /logout,
  /reset_password/request, /reset_password/confirm -- verified password reset
  /profile/{user_id}             -- health profile, OWNER-ONLY (auth required)
  /chat                          -- AI check-in, auth required (protects the
                                     paid Groq API key from anonymous use)
  /conditions, /methods_reference,
  /effectiveness, /eligibility   -- contraceptive eligibility tool (stateless, public)

Run from the project root:
    uvicorn app_backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import base64
import binascii
import json
import math
import os

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app_backend import auth, chat, database, eligibility_data

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"
FORCE_HTTPS = os.getenv(
    "FORCE_HTTPS", "true" if IS_PRODUCTION else "false"
).strip().lower() in {"1", "true", "yes"}


def _configured_cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if origins:
        if "*" in origins:
            raise RuntimeError("CORS_ORIGINS must contain explicit origins, not '*'.")
        return origins
    if IS_PRODUCTION:
        raise RuntimeError("CORS_ORIGINS must be configured in production.")
    return [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
    ]


CORS_ORIGINS = _configured_cors_origins()
# `flutter run -d chrome` / `flutter run -d web-server` pick a RANDOM port
# (e.g. localhost:52153), so a fixed dev allowlist blocks the browser with a
# CORS error. Outside production we therefore also accept any loopback origin
# on any port. In production this stays off unless ALLOW_LOCALHOST_CORS is
# set to true (set it to false or remove it once the app is published).
ALLOW_LOCALHOST_CORS = os.getenv("ALLOW_LOCALHOST_CORS", "false").strip().lower() in {"1", "true", "yes"}
CORS_DEV_ORIGIN_REGEX = (
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    if (not IS_PRODUCTION or ALLOW_LOCALHOST_CORS)
    else None
)
APP_DEPLOYMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_deployment")
MAX_PROFILE_BYTES = 300_000  # ~300 KB -- generous for the diary/profile JSON blob
MAX_CHAT_ATTACHMENT_BYTES = 8_000_000

PCOS_NUMERIC_RANGES = {
    "age_yrs": (10.0, 100.0),
    "weight_kg": (20.0, 350.0),
    "height_cm": (100.0, 250.0),
    "cycle_length_days": (10.0, 90.0),
    "prl": (0.0, 1000.0),
    "vit_d3": (0.0, 200.0),
    "prg": (0.0, 1000.0),
    "rbs": (0.0, 1000.0),
    "bp_systolic": (50.0, 250.0),
    "bp_diastolic": (30.0, 150.0),
    "follicle_no_l": (0.0, 100.0),
    "follicle_no_r": (0.0, 100.0),
    "avg_f_size_l": (0.0, 50.0),
    "avg_f_size_r": (0.0, 50.0),
    "endometrium": (0.0, 50.0),
}

# ---- Load model, scaler, and metadata once at startup ----
model = joblib.load(os.path.join(APP_DEPLOYMENT_DIR, "pcos_app_model.joblib"))
scaler = joblib.load(os.path.join(APP_DEPLOYMENT_DIR, "pcos_app_scaler.joblib"))

with open(os.path.join(APP_DEPLOYMENT_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_order"]
CYCLE_ENCODING = metadata["categorical_encodings"]["Cycle(R/I)"]
BINARY_FIELDS = metadata["categorical_encodings"]["binary_yes_no_fields"]
BINARY_ENCODING = metadata["categorical_encodings"]["binary_encoding"]

# ---- Screening bands -------------------------------------------------------
# A single 0.5 cutoff is the wrong shape for screening. Out-of-fold on this
# model it catches only ~75% of true cases, so roughly 1 in 4 women with PCOS
# are told "No PCOS Detected" and never follow up. Lowering the cutoff raises
# recall but adds false alarms. So the API reports three bands and leaves the
# judgement call visible:
#
#   p < 0.35            -> low       (OOF recall at this cut ~0.831 of cases
#                                      correctly sent above the 0.35 line)
#   0.35 <= p < 0.5     -> elevated  "worth screening"
#   p >= 0.5            -> high
#
# `prediction` still uses the 0.5 boundary so existing clients are unchanged.
RISK_ELEVATED = 0.35
RISK_HIGH = 0.5
RISK_BANDS = {
    "low": (
        "Low likelihood",
        "Your answers do not strongly suggest PCOS. Keep tracking your cycle "
        "and symptoms, and see a doctor if they change.",
    ),
    "elevated": (
        "Possible - worth screening",
        "Some of your answers are associated with PCOS. This is not a "
        "diagnosis, but it is worth discussing with a doctor.",
    ),
    "high": (
        "High likelihood",
        "Your answers are strongly associated with PCOS. Please arrange an "
        "appointment with a doctor for proper testing and confirmation.",
    ),
}

app = FastAPI(title="PCOS Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_DEV_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


def _apply_security_headers(request: Request, response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-DNS-Prefetch-Control"] = "off"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/profile") or request.url.path in {
        "/chat",
        "/signup",
        "/signin",
        "/logout",
        "/reset_password",
        "/reset_password/request",
        "/reset_password/confirm",
    }:
        response.headers["Cache-Control"] = "no-store"
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


@app.middleware("http")
async def security_headers(request: Request, call_next):
    forwarded_proto = request.headers.get(
        "x-forwarded-proto", request.url.scheme
    ).split(",", 1)[0].strip().lower()
    if IS_PRODUCTION and FORCE_HTTPS and forwarded_proto == "http":
        redirect = RedirectResponse(
            url=str(request.url.replace(scheme="https")),
            status_code=307,
        )
        return _apply_security_headers(request, redirect)

    response = await call_next(request)
    return _apply_security_headers(request, response)


# Creates all tables on first run; a no-op after that.
database.init_db()


def _client_key(request: Request) -> str:
    """Best-effort client identifier for rate limiting. Falls back to
    a constant if the client host isn't available (e.g. some test
    clients) -- rate limiting still applies, just shared across those
    callers rather than being a hole."""
    return request.client.host if request.client else "unknown"


def get_current_user_id(
    authorization: str | None = Header(default=None),
) -> int:
    """FastAPI dependency: verifies the `Authorization: Bearer <token>`
    header and returns the signed-in user's id. Raises 401 if missing,
    malformed, or the token is invalid/expired."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Please sign in again.",
        )
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return auth.verify_token(token)
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


class PCOSInput(BaseModel):
    """
    Request body the Flutter app must send. All fields use plain,
    human-readable values -- the API handles converting Yes/No and
    Regular/Irregular into the numbers the model expects.
    """
    age_yrs: float = Field(..., description="Age in years")
    weight_kg: float = Field(..., description="Weight in kilograms")
    height_cm: float = Field(..., description="Height in centimeters")
    cycle_regularity: str = Field(..., description="'Regular' or 'Irregular'")
    cycle_length_days: float = Field(..., description="Average cycle length in days")
    prl: float = Field(..., description="PRL (ng/mL)")
    vit_d3: float = Field(..., description="Vitamin D3 (ng/mL)")
    prg: float = Field(..., description="PRG (ng/mL)")
    rbs: float = Field(..., description="Random Blood Sugar (mg/dl)")
    bp_systolic: float = Field(..., description="BP Systolic (mmHg)")
    bp_diastolic: float = Field(..., description="BP Diastolic (mmHg)")
    follicle_no_l: float = Field(..., description="Follicle No. (Left)")
    follicle_no_r: float = Field(..., description="Follicle No. (Right)")
    avg_f_size_l: float = Field(..., description="Avg. Follicle size (Left) mm")
    avg_f_size_r: float = Field(..., description="Avg. Follicle size (Right) mm")
    endometrium: float = Field(..., description="Endometrium thickness (mm)")
    weight_gain: str = Field(..., description="'Yes' or 'No'")
    hair_growth: str = Field(..., description="'Yes' or 'No'")
    skin_darkening: str = Field(..., description="'Yes' or 'No'")
    hair_loss: str = Field(..., description="'Yes' or 'No'")
    pimples: str = Field(..., description="'Yes' or 'No'")
    fast_food: str = Field(..., description="'Yes' or 'No'")
    regular_exercise: str = Field(..., description="'Yes' or 'No'")


class PCOSOutput(BaseModel):
    """Response returned to the Flutter app."""
    prediction: str
    pcos_probability: float
    model_used: str
    # Screening band, so the UI can act on a "possible" result instead of
    # collapsing everything above 0.5 into a hard yes/no.
    risk_level: str
    risk_label: str
    screening_advice: str


def encode_binary(value: str, field_name: str) -> int:
    if not isinstance(value, str) or len(value) > 32:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be 'Yes' or 'No'.",
        )
    normalized = value.strip().capitalize()
    if normalized not in BINARY_ENCODING:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{field_name}' must be 'Yes' or 'No', got '{value}'"
        )
    return BINARY_ENCODING[normalized]


def encode_cycle(value: str) -> int:
    if not isinstance(value, str) or len(value) > 32:
        raise HTTPException(
            status_code=422,
            detail="cycle_regularity must be 'Regular' or 'Irregular'.",
        )
    normalized = value.strip().capitalize()
    if normalized not in CYCLE_ENCODING:
        raise HTTPException(
            status_code=422,
            detail=f"cycle_regularity must be 'Regular' or 'Irregular', got '{value}'"
        )
    return CYCLE_ENCODING[normalized]


def _validate_pcos_input(data: PCOSInput) -> None:
    if data.height_cm <= 0:
        raise HTTPException(
            status_code=422,
            detail="Height must be greater than 0 to calculate BMI.",
        )

    for field_name, (minimum, maximum) in PCOS_NUMERIC_RANGES.items():
        value = getattr(data, field_name)
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Field '{field_name}' must be a finite number "
                    f"between {minimum:g} and {maximum:g}."
                ),
            )


def build_feature_vector(data: PCOSInput) -> np.ndarray:
    _validate_pcos_input(data)
    bmi = data.weight_kg / ((data.height_cm / 100) ** 2)

    # NOTE: `cycle_length_days` is accepted and range-validated, but it is
    # deliberately NOT a model input. The training column of that name holds
    # 12 integer codes in the range 0-12, not days, so a real user value
    # (10-90) always fell past the largest value the model had seen and
    # collapsed into a single leaf. Dropping it changed cross-validated AUC by
    # +0.0007. The field stays in the request so the Flutter form and the saved
    # user profile keep working unchanged.
    values_by_name = {
        "Age (yrs)": data.age_yrs,
        "BMI": bmi,
        "Cycle(R/I)": encode_cycle(data.cycle_regularity),
        "PRL(ng/mL)": data.prl,
        "Vit D3 (ng/mL)": data.vit_d3,
        "PRG(ng/mL)": data.prg,
        "RBS(mg/dl)": data.rbs,
        "BP _Systolic (mmHg)": data.bp_systolic,
        "BP _Diastolic (mmHg)": data.bp_diastolic,
        "Follicle No. (L)": data.follicle_no_l,
        "Follicle No. (R)": data.follicle_no_r,
        "Avg. F size (L) (mm)": data.avg_f_size_l,
        "Avg. F size (R) (mm)": data.avg_f_size_r,
        "Endometrium (mm)": data.endometrium,
        "Weight gain(Y/N)": encode_binary(data.weight_gain, "weight_gain"),
        "hair growth(Y/N)": encode_binary(data.hair_growth, "hair_growth"),
        "Skin darkening (Y/N)": encode_binary(data.skin_darkening, "skin_darkening"),
        "Hair loss(Y/N)": encode_binary(data.hair_loss, "hair_loss"),
        "Pimples(Y/N)": encode_binary(data.pimples, "pimples"),
        "Fast food (Y/N)": encode_binary(data.fast_food, "fast_food"),
        "Reg.Exercise(Y/N)": encode_binary(data.regular_exercise, "regular_exercise"),
    }

    ordered_values = [values_by_name[feature_name] for feature_name in FEATURE_ORDER]
    return np.array(ordered_values, dtype=float).reshape(1, -1)


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok", "model": metadata["model_name"]}


@app.post("/predict", response_model=PCOSOutput)
def predict(data: PCOSInput):
    try:
        feature_vector = build_feature_vector(data)
        scaled_vector = scaler.transform(feature_vector)
        pred_class = model.predict(scaled_vector)[0]
        pred_proba = model.predict_proba(scaled_vector)[0][1]  # probability of class 1 (PCOS)
    except HTTPException:
        raise
    except Exception:
        # Never leak model/library internals in the response.
        raise HTTPException(
            status_code=500,
            detail="Could not process the prediction. Please check your inputs and try again.",
        )

    if pred_proba >= RISK_HIGH:
        risk_level = "high"
    elif pred_proba >= RISK_ELEVATED:
        risk_level = "elevated"
    else:
        risk_level = "low"
    risk_label, screening_advice = RISK_BANDS[risk_level]

    return PCOSOutput(
        prediction="PCOS Detected" if pred_class == 1 else "No PCOS Detected",
        pcos_probability=round(float(pred_proba), 4),
        model_used=metadata["model_name"],
        risk_level=risk_level,
        risk_label=risk_label,
        screening_advice=screening_advice,
    )


# =================================================================
# ACCOUNTS -- /signup, /signin, /logout, /reset_password/*
# =================================================================

class SignUpRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=6, max_length=72)


class SignInRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=72)


class PasswordResetRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


class PasswordResetConfirmRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    code: str = Field(..., min_length=8, max_length=8, pattern=r"^\d{8}$")
    new_password: str = Field(..., min_length=6, max_length=72)


@app.post("/signup")
def signup(data: SignUpRequest, request: Request):
    try:
        result = auth.sign_up(data.name, data.email, data.password, _client_key(request))
        return result
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/signin")
def signin(data: SignInRequest, request: Request):
    try:
        result = auth.sign_in(data.email, data.password, _client_key(request))
        return result
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.post("/logout")
def logout(authorization: str | None = Header(default=None)):
    """Best-effort: invalidates the session if a token was sent. Always
    returns ok so the app can clear its local state regardless."""
    if authorization and authorization.startswith("Bearer "):
        auth.invalidate_session(authorization.removeprefix("Bearer ").strip())
    return {"status": "ok"}


@app.post("/reset_password/request")
def request_password_reset_endpoint(
    data: PasswordResetRequest,
    request: Request,
):
    try:
        debug_code = auth.request_password_reset(data.email, _client_key(request))
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    response = {"status": "ok"}
    if debug_code:
        response["debug_code"] = debug_code
    return response


@app.post("/reset_password/confirm")
def confirm_password_reset_endpoint(
    data: PasswordResetConfirmRequest,
    request: Request,
):
    try:
        auth.confirm_password_reset(
            data.email,
            data.code,
            data.new_password,
            _client_key(request),
        )
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    return {"status": "ok"}


@app.post("/reset_password", status_code=410)
def reset_password_endpoint():
    raise HTTPException(
        status_code=410,
        detail="Password reset now requires a verification code.",
    )


# =================================================================
# HEALTH PROFILE -- /profile/{user_id}  (OWNER-ONLY)
# =================================================================

@app.get("/profile/{user_id}")
def get_profile(user_id: str, current_user_id: int = Depends(get_current_user_id)):
    if str(current_user_id) != user_id:
        raise HTTPException(
            status_code=403, detail="You don't have permission to access this profile."
        )
    profile = database.get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="No profile found for this user.")
    return profile


@app.put("/profile/{user_id}")
def put_profile(
    user_id: str,
    profile: dict,
    current_user_id: int = Depends(get_current_user_id),
):
    if str(current_user_id) != user_id:
        raise HTTPException(
            status_code=403, detail="You don't have permission to modify this profile."
        )

    if "user_id" in profile and str(profile["user_id"]) != user_id:
        raise HTTPException(
            status_code=422,
            detail="Profile user_id must match the URL.",
        )

    profile["user_id"] = user_id
    profile_size = len(json.dumps(profile, ensure_ascii=False).encode("utf-8"))
    if profile_size > MAX_PROFILE_BYTES:
        raise HTTPException(status_code=413, detail="Profile payload is too large.")

    database.upsert_profile(user_id, profile)
    return {"status": "ok"}


@app.delete("/profile/{user_id}")
def delete_profile(user_id: str, current_user_id: int = Depends(get_current_user_id)):
    if str(current_user_id) != user_id:
        raise HTTPException(
            status_code=403, detail="You don't have permission to delete this profile."
        )
    database.delete_profile(user_id)
    return {"status": "deleted", "user_id": user_id}


# =================================================================
# AI CHECK-IN -- /chat  (auth required)
# =================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    history: list[dict] = Field(default_factory=list, max_length=10)
    profile_context: dict | None = None
    language: str | None = Field(default=None, max_length=16)
    attachment: dict | None = None


ALLOWED_CHAT_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


def _validate_chat_request(data: ChatRequest) -> None:
    if not data.message.strip() and data.attachment is None:
        raise HTTPException(status_code=422, detail="Message or attachment is required.")

    for entry in data.history:
        if entry.get("role") not in {"user", "assistant"}:
            raise HTTPException(status_code=422, detail="Invalid chat history role.")
        message = entry.get("message")
        if not isinstance(message, str) or not message or len(message) > 4000:
            raise HTTPException(
                status_code=422,
                detail="Each chat history message must contain 1 to 4000 characters.",
            )

    if data.profile_context is not None:
        try:
            context_size = len(
                json.dumps(data.profile_context, ensure_ascii=False).encode("utf-8")
            )
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid profile context.")
        if context_size > 100_000:
            raise HTTPException(status_code=413, detail="Profile context is too large.")

    if data.language not in (None, "en", "ur"):
        raise HTTPException(status_code=422, detail="Unsupported chat language.")

    if data.attachment is None:
        return

    file_name = data.attachment.get("file_name")
    mime_type = data.attachment.get("mime_type")
    encoded_data = data.attachment.get("data_base64")
    if not isinstance(file_name, str) or not file_name or len(file_name) > 255:
        raise HTTPException(status_code=422, detail="Invalid attachment file name.")
    if mime_type not in ALLOWED_CHAT_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Attachment type is not supported.")
    if not isinstance(encoded_data, str) or not encoded_data:
        raise HTTPException(status_code=422, detail="Attachment data is required.")
    if len(encoded_data) > ((MAX_CHAT_ATTACHMENT_BYTES + 2) // 3) * 4:
        raise HTTPException(status_code=413, detail="Attachment is too large.")
    try:
        decoded_data = base64.b64decode(encoded_data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="Attachment data is not valid base64.")
    if len(decoded_data) > MAX_CHAT_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="Attachment is too large.")


# /chat is rate-limited per client (like the auth endpoints) because
# every call spends money on the Groq API key -- an anonymous flood of
# messages is the one cost/abuse vector a free-tier AI setup can't
# absorb. The limits below are generous for a real user but cap an
# attacker.
CHAT_RATE_LIMIT_MAX = 30
CHAT_RATE_LIMIT_WINDOW_SECONDS = 300


@app.post("/chat")
def chat_endpoint(
    data: ChatRequest,
    request: Request,
    current_user_id: int = Depends(get_current_user_id),
):
    try:
        auth._check_rate_limit(
            f"chat:{_client_key(request)}",
            max_attempts=CHAT_RATE_LIMIT_MAX,
            window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS,
        )
    except auth.AuthError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    _validate_chat_request(data)
    try:
        return chat.get_reply(
            data.message,
            data.history,
            data.profile_context,
            attachment=data.attachment,
            language=data.language,
        )
    except chat.AttachmentError as e:
        detail = str(e)
        status_code = 413 if "too large" in detail.lower() else 422
        raise HTTPException(status_code=status_code, detail=detail)


# =================================================================
# CONTRACEPTIVE ELIGIBILITY -- /conditions, /methods_reference,
# /effectiveness, /eligibility (stateless, no personal data -- public)
# =================================================================

class EligibilityRequest(BaseModel):
    condition_ids: list[str] = Field(..., max_length=100)


@app.get("/conditions")
def get_conditions():
    return eligibility_data.list_conditions()


@app.get("/methods_reference")
def get_methods_reference():
    return eligibility_data.list_methods()


@app.get("/effectiveness")
def get_effectiveness():
    return eligibility_data.list_effectiveness()


@app.post("/eligibility")
def post_eligibility(data: EligibilityRequest):
    if any(not condition_id or len(condition_id) > 100 for condition_id in data.condition_ids):
        raise HTTPException(
            status_code=422,
            detail="Each condition id must contain 1 to 100 characters.",
        )
    try:
        return eligibility_data.check_eligibility(data.condition_ids)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))