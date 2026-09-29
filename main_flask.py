"""
main_flask.py

Flask backend that serves the app-matched PCOS model
(app_deployment/pcos_app_model.joblib) to the Flutter app.

Same behavior as the original FastAPI main.py, but written directly in
Flask (native WSGI) so it runs reliably on PythonAnywhere's free tier
without needing an ASGI-to-WSGI bridge.

Exposes a single POST endpoint /predict that accepts the 22 form
fields (raw, human-readable values), applies the same scaling used
during training, runs the model, and returns a prediction + probability.

Also registers two additional blueprints (added for the Health Profile
diary + AI check-in feature):
    - health_profile_bp: /profile/<user_id> diary storage endpoints
    - chat_bp: /chat AI check-in endpoint

Both live as flat files in this same folder (health_profile_api.py,
chat_api.py) -- there's no app_backend/ package in this project, so
they're imported directly by filename, not as a package.

ACCOUNTS + SESSIONS: signup/signin now issue a real session token
(see auth_utils.py) tied to a per-account `user_id` -- previously
signup/signin returned only {"name": ...} with no token at all, and
/profile was keyed by an anonymous, unauthenticated device id that
anyone could read or overwrite if they knew or guessed it. The
Flutter app now uses the account's user_id (not the device id) once
signed in, and sends the token back as `Authorization: Bearer
<token>` on every request that touches personal data.
"""

import os
import json
import secrets
import sqlite3
import joblib
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from health_profile_api import health_profile_bp
from chat_api import chat_bp
from auth_utils import (
    RateLimitError,
    check_rate_limit,
    clear_rate_limit,
    invalidate_all_sessions_for_account,
    invalidate_session,
    issue_session,
)

APP_DEPLOYMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_deployment")

# ---- Load model, scaler, and metadata once at startup ----
model = joblib.load(os.path.join(APP_DEPLOYMENT_DIR, "pcos_app_model.joblib"))
scaler = joblib.load(os.path.join(APP_DEPLOYMENT_DIR, "pcos_app_scaler.joblib"))

with open(os.path.join(APP_DEPLOYMENT_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_order"]
CYCLE_ENCODING = metadata["categorical_encodings"]["Cycle(R/I)"]
BINARY_ENCODING = metadata["categorical_encodings"]["binary_encoding"]

# All 22 fields the Flutter app must send in the JSON body of /predict.
# Note: "prg" holds the Progesterone value (ng/mL) -- the abbreviation
# stays as the field/key name so nothing sending this JSON shape breaks,
# only the human-facing label in the app itself was unclear before.
REQUIRED_FIELDS = [
    "age_yrs", "weight_kg", "height_cm", "cycle_regularity",
    "cycle_length_days", "prl", "vit_d3", "prg", "rbs",
    "bp_systolic", "bp_diastolic", "follicle_no_l", "follicle_no_r",
    "avg_f_size_l", "avg_f_size_r", "endometrium", "weight_gain",
    "hair_growth", "skin_darkening", "hair_loss", "pimples",
    "fast_food", "regular_exercise",
]

app = Flask(__name__)

# These two were missing from the live server entirely -- both files
# (health_profile_api.py, chat_api.py) need to be uploaded to this same
# folder for these imports to succeed.
app.register_blueprint(health_profile_bp)
app.register_blueprint(chat_bp)

# ---- Simple on-server accounts store (SQLite) ----
# Storing accounts here (instead of only on the phone) means a user's
# account survives reinstalling the app, switching phones, etc. -- the
# app itself only keeps a "remember me" flag locally for convenience.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.db")


def _init_accounts_table() -> None:
    """Creates the accounts table if it doesn't exist, and migrates in
    the `user_id` column for deployments created before session tokens
    existed (accounts made before this column existed get one
    backfilled here so they can still sign in and get a token)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL
        )
        """
    )
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(accounts)")]
    if "user_id" not in existing_cols:
        conn.execute("ALTER TABLE accounts ADD COLUMN user_id TEXT")

    # Backfill any pre-existing accounts that don't have a user_id yet.
    rows = conn.execute(
        "SELECT email FROM accounts WHERE user_id IS NULL OR user_id = ''"
    ).fetchall()
    for (email,) in rows:
        conn.execute(
            "UPDATE accounts SET user_id = ? WHERE email = ?",
            (secrets.token_hex(16), email),
        )
    conn.commit()
    conn.close()


# Run once at import time, same pattern as health_profile_api.py's init_db().
_init_accounts_table()


def get_db():
    """Opens a connection to the accounts database."""
    return sqlite3.connect(DB_PATH)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _client_key() -> str:
    """Best-effort client identifier for rate limiting."""
    return request.remote_addr or "unknown"


@app.route("/signup", methods=["POST", "OPTIONS"])
def signup():
    """Creates a new account and an initial session.
    Body: {name, email, password}.
    Returns: {name, user_id, token, expires_at}."""
    if request.method == "OPTIONS":
        return "", 204

    try:
        check_rate_limit(f"signup:{_client_key()}", max_attempts=5, window_seconds=600)
    except RateLimitError as e:
        return jsonify({"detail": e.detail}), 429

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = normalize_email(data.get("email") or "")
    password = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"detail": "Name, email, and password are all required."}), 422

    if len(password) < 6:
        return jsonify({"detail": "Password must be at least 6 characters."}), 422

    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT 1 FROM accounts WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            return jsonify({
                "detail": "An account with this email already exists. Try signing in instead."
            }), 409

        user_id = secrets.token_hex(16)
        conn.execute(
            "INSERT INTO accounts (email, password_hash, name, user_id) VALUES (?, ?, ?, ?)",
            (email, generate_password_hash(password), name, user_id),
        )
        conn.commit()
    finally:
        conn.close()

    token, expires_at = issue_session(user_id)
    return jsonify({"name": name, "user_id": user_id, "token": token, "expires_at": expires_at})


@app.route("/signin", methods=["POST", "OPTIONS"])
def signin():
    """Validates credentials and issues a session.
    Body: {email, password}.
    Returns: {name, user_id, token, expires_at}."""
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email") or "")
    password = data.get("password") or ""

    rate_key = f"signin:{_client_key()}:{email}"
    try:
        check_rate_limit(rate_key, max_attempts=8, window_seconds=300)
    except RateLimitError as e:
        return jsonify({"detail": e.detail}), 429

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT password_hash, name, user_id FROM accounts WHERE email = ?", (email,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        # Same message as a wrong password below -- distinguishing them
        # lets an attacker enumerate which emails have accounts.
        return jsonify({"detail": "Incorrect email or password."}), 401

    password_hash, name, user_id = row
    if not check_password_hash(password_hash, password):
        return jsonify({"detail": "Incorrect email or password."}), 401

    clear_rate_limit(rate_key)
    token, expires_at = issue_session(user_id)
    return jsonify({"name": name, "user_id": user_id, "token": token, "expires_at": expires_at})


@app.route("/logout", methods=["POST", "OPTIONS"])
def logout():
    """Best-effort: invalidates the session if a valid token was sent.
    Always returns ok so the app can clear its local state regardless."""
    if request.method == "OPTIONS":
        return "", 204

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        invalidate_session(auth_header[len("Bearer "):].strip())
    return jsonify({"status": "ok"})


@app.route("/reset_password", methods=["POST", "OPTIONS"])
def reset_password():
    """Resets a password for an existing account. Body: {email, new_password}.
    Always returns the same generic {"status": "ok"} whether or not the
    email exists, so this can't be used to check which emails are
    registered. Known limitation: with no email-verification step,
    this can't confirm the caller owns the account -- it's rate
    limited instead. A real fix needs an emailed one-time code/link."""
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email") or "")
    new_password = data.get("new_password") or ""

    try:
        check_rate_limit(f"reset:{_client_key()}:{email}", max_attempts=5, window_seconds=900)
    except RateLimitError as e:
        return jsonify({"detail": e.detail}), 429

    if not new_password or len(new_password) < 6:
        return jsonify({"detail": "Password must be at least 6 characters."}), 422

    conn = get_db()
    try:
        row = conn.execute(
            "SELECT user_id FROM accounts WHERE email = ?", (email,)
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE accounts SET password_hash = ? WHERE email = ?",
                (generate_password_hash(new_password), email),
            )
            conn.commit()
    finally:
        conn.close()

    if row is not None:
        # Force every existing session for this account to re-authenticate.
        invalidate_all_sessions_for_account(row[0])

    return jsonify({"status": "ok"})


@app.after_request
def add_cors_headers(response):
    """Allow requests from the Flutter app (any origin, mirrors the
    original CORSMiddleware(allow_origins=["*"]) behavior). No
    Access-Control-Allow-Credentials header is set -- auth here is a
    Bearer token in the Authorization header, not a cookie, so there's
    no session-cookie/CSRF risk that flag would normally guard
    against."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def encode_binary(value: str, field_name: str) -> int:
    """
    Converts a 'Yes'/'No' string into the 1/0 the model expects.

    Args:
        value (str): 'Yes' or 'No' (case-insensitive).
        field_name (str): name of the field, used only for error messages.

    Returns:
        int: 1 for Yes, 0 for No.
    """
    normalized = str(value).strip().capitalize()
    if normalized not in BINARY_ENCODING:
        raise ValueError(f"Field '{field_name}' must be 'Yes' or 'No', got '{value}'")
    return BINARY_ENCODING[normalized]


def encode_cycle(value: str) -> int:
    """
    Converts 'Regular'/'Irregular' into the 2/4 the model expects.

    Args:
        value (str): 'Regular' or 'Irregular' (case-insensitive).

    Returns:
        int: 2 for Regular, 4 for Irregular.
    """
    normalized = str(value).strip().capitalize()
    if normalized not in CYCLE_ENCODING:
        raise ValueError(f"cycle_regularity must be 'Regular' or 'Irregular', got '{value}'")
    return CYCLE_ENCODING[normalized]


def build_feature_vector(data: dict) -> np.ndarray:
    """
    Converts the incoming request dict into a feature vector in the
    EXACT order the model was trained on (FEATURE_ORDER from metadata).

    Args:
        data (dict): validated request body (22 fields).

    Returns:
        numpy.ndarray: shape (1, 22), ready to be scaled and predicted on.
    """
    weight_kg = float(data["weight_kg"])
    height_cm = float(data["height_cm"])
    bmi = weight_kg / ((height_cm / 100) ** 2)

    values_by_name = {
        "Age (yrs)": float(data["age_yrs"]),
        "BMI": bmi,
        "Cycle(R/I)": encode_cycle(data["cycle_regularity"]),
        "Cycle length(days)": float(data["cycle_length_days"]),
        "PRL(ng/mL)": float(data["prl"]),
        "Vit D3 (ng/mL)": float(data["vit_d3"]),
        "PRG(ng/mL)": float(data["prg"]),
        "RBS(mg/dl)": float(data["rbs"]),
        "BP _Systolic (mmHg)": float(data["bp_systolic"]),
        "BP _Diastolic (mmHg)": float(data["bp_diastolic"]),
        "Follicle No. (L)": float(data["follicle_no_l"]),
        "Follicle No. (R)": float(data["follicle_no_r"]),
        "Avg. F size (L) (mm)": float(data["avg_f_size_l"]),
        "Avg. F size (R) (mm)": float(data["avg_f_size_r"]),
        "Endometrium (mm)": float(data["endometrium"]),
        "Weight gain(Y/N)": encode_binary(data["weight_gain"], "weight_gain"),
        "hair growth(Y/N)": encode_binary(data["hair_growth"], "hair_growth"),
        "Skin darkening (Y/N)": encode_binary(data["skin_darkening"], "skin_darkening"),
        "Hair loss(Y/N)": encode_binary(data["hair_loss"], "hair_loss"),
        "Pimples(Y/N)": encode_binary(data["pimples"], "pimples"),
        "Fast food (Y/N)": encode_binary(data["fast_food"], "fast_food"),
        "Reg.Exercise(Y/N)": encode_binary(data["regular_exercise"], "regular_exercise"),
    }

    ordered_values = [values_by_name[feature_name] for feature_name in FEATURE_ORDER]
    return np.array(ordered_values, dtype=float).reshape(1, -1)


@app.route("/", methods=["GET"])
def health_check():
    """Simple endpoint to confirm the API is running."""
    return jsonify({"status": "ok", "model": metadata["model_name"]})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Runs the PCOS model on the submitted form data.

    Expects a JSON body with the 22 fields from the Flutter app.
    Returns JSON: {prediction, pcos_probability, model_used}.
    """
    # Browsers send a CORS "preflight" OPTIONS request before the real
    # POST -- just acknowledge it, no body needed.
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"detail": "Request body must be valid JSON."}), 400

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        return jsonify({"detail": f"Missing required field(s): {', '.join(missing)}"}), 422

    try:
        feature_vector = build_feature_vector(data)
    except (ValueError, TypeError) as exc:
        return jsonify({"detail": str(exc)}), 422

    try:
        scaled_vector = scaler.transform(feature_vector)
        pred_class = model.predict(scaled_vector)[0]
        pred_proba = model.predict_proba(scaled_vector)[0][1]  # probability of class 1 (PCOS)
    except Exception:
        # Never leak model/library internals in the response.
        return jsonify({
            "detail": "Could not process the prediction. Please check your inputs and try again."
        }), 500

    return jsonify({
        "prediction": "PCOS Detected" if pred_class == 1 else "No PCOS Detected",
        "pcos_probability": round(float(pred_proba), 4),
        "model_used": metadata["model_name"],
    })


# ============================================================
# Contraceptive eligibility reference data
# ============================================================
# Modeled on the WHO Medical Eligibility Criteria for Contraceptive
# Use (5th edition, 2015) -- a well-established, publicly documented
# global health standard. The underlying medical facts (which
# conditions affect which methods, and how severely) are not
# copyrightable; only WHO's specific wording/artwork is, and none of
# that is reproduced here.
#
# NOTE: This is an expanded, curated reference covering 60 commonly
# referenced conditions against all 9 real WHO method categories --
# built for general educational use in this app, not as a clinical
# decision tool, and not pulled from an official machine-readable
# dataset file. Categories:
#   1 = Use in any circumstance
#   2 = Generally use the method
#   3 = Use with caution / not usually recommended unless no better option
#   4 = Method should not be used

METHODS = [
    {"id": "chc", "label": "Combined hormonal contraceptives"},
    {"id": "pop", "label": "Progestogen-only pills"},
    {"id": "inj", "label": "Progestogen-only injectables"},
    {"id": "imp", "label": "Implants"},
    {"id": "lng_iud", "label": "Levonorgestrel IUD"},
    {"id": "cu_iud", "label": "Copper intrauterine device"},
    {"id": "barrier", "label": "Barrier methods"},
    {"id": "lam", "label": "Lactational amenorrhoea method"},
    {"id": "sterilization", "label": "Female sterilization"},
]

CONDITIONS = [
    # ---- Original 18 ----
    {"id": "smoking_35_plus", "label": "Smoking (age 35 or older)"},
    {"id": "obesity_bmi30", "label": "Obesity (BMI 30 or above)"},
    {"id": "multiple_cvd_risk", "label": "Multiple risk factors for cardiovascular disease"},
    {"id": "hypertension_controlled", "label": "Hypertension, adequately controlled"},
    {"id": "hypertension_uncontrolled", "label": "Hypertension, uncontrolled (\u2265160/100)"},
    {"id": "vte_history", "label": "History of DVT/PE (blood clots)"},
    {"id": "migraine_no_aura", "label": "Migraine without aura"},
    {"id": "migraine_with_aura", "label": "Migraine with aura"},
    {"id": "diabetes_no_vascular", "label": "Diabetes, no vascular disease"},
    {"id": "diabetes_with_vascular", "label": "Diabetes with vascular disease"},
    {"id": "breastfeeding_lt6weeks", "label": "Breastfeeding, less than 6 weeks postpartum"},
    {"id": "postpartum_non_breastfeeding", "label": "Postpartum (not breastfeeding), under 21 days"},
    {"id": "current_breast_cancer", "label": "Current breast cancer"},
    {"id": "past_breast_cancer_5yrs", "label": "Past breast cancer, none for 5+ years"},
    {"id": "cervical_cancer_awaiting_treatment", "label": "Cervical cancer, awaiting treatment"},
    {"id": "viral_hepatitis_active", "label": "Viral hepatitis (active/flare)"},
    {"id": "high_risk_hiv_sti", "label": "High risk of HIV/STI exposure"},
    {"id": "known_hiv_on_art", "label": "Known HIV infection, on treatment"},

    # ---- New: expanded to match the reference tool ----
    {"id": "anticonvulsant_certain", "label": "Anticonvulsant therapy (phenytoin, carbamazepine, barbiturates, primidone, topiramate, oxcarbazepine)"},
    {"id": "anticonvulsant_lamotrigine", "label": "Anticonvulsant therapy (lamotrigine)"},
    {"id": "antimicrobial_rifampicin", "label": "Antimicrobial therapy (rifampicin or rifabutin)"},
    {"id": "antimicrobial_broad_spectrum", "label": "Antimicrobial therapy (broad-spectrum antibiotics)"},
    {"id": "antiretroviral_therapy", "label": "Antiretroviral therapy (ART)"},
    {"id": "benign_ovarian_tumours", "label": "Benign ovarian tumours (including cysts)"},
    {"id": "bp_unavailable", "label": "Blood pressure measurement unavailable"},
    {"id": "breast_undiagnosed_mass", "label": "Breast disease: undiagnosed mass"},
    {"id": "breast_benign_disease", "label": "Breast disease: benign breast disease"},
    {"id": "breast_family_history_cancer", "label": "Family history of breast cancer"},
    {"id": "breastfeeding_6wk_6mo", "label": "Breastfeeding, 6 weeks to under 6 months (primarily breastfeeding)"},
    {"id": "breastfeeding_6mo_plus", "label": "Breastfeeding, 6 months or more postpartum"},
    {"id": "cardiovascular_disease", "label": "Cardiovascular disease (current or history)"},
    {"id": "cervical_ectropion", "label": "Cervical ectropion"},
    {"id": "cin", "label": "Cervical intraepithelial neoplasia (CIN)"},
    {"id": "cirrhosis_mild", "label": "Cirrhosis, mild (compensated)"},
    {"id": "cirrhosis_severe", "label": "Cirrhosis, severe (decompensated)"},
    {"id": "history_cholestasis", "label": "History of cholestasis (pregnancy-related)"},
    {"id": "history_pregnancy_hypertension", "label": "History of high blood pressure during pregnancy (current BP normal)"},
    {"id": "history_pelvic_surgery", "label": "History of pelvic surgery"},
    {"id": "iron_deficient_anaemia", "label": "Iron-deficient anaemia"},
    {"id": "known_dyslipidaemias", "label": "Known dyslipidaemias (without other cardiovascular risk factors)"},
    {"id": "known_thrombogenic_mutations", "label": "Known thrombogenic mutations (e.g. Factor V Leiden)"},
    {"id": "liver_tumours_benign", "label": "Liver tumours, benign (focal nodular hyperplasia)"},
    {"id": "liver_tumours_malignant", "label": "Liver tumours, malignant (hepatocellular)"},
    {"id": "severe_dysmenorrhoea", "label": "Severe dysmenorrhoea"},
    {"id": "sti_current_purulent", "label": "Current purulent cervicitis, chlamydia, or gonorrhoea"},
    {"id": "sti_other", "label": "Other STIs (excluding HIV and hepatitis)"},
    {"id": "sickle_cell_disease", "label": "Sickle cell disease"},
    {"id": "smoking_under_35", "label": "Smoking, under age 35"},
    {"id": "stroke_history", "label": "History of stroke"},
    {"id": "varicose_veins", "label": "Varicose veins"},
    {"id": "superficial_venous_thrombosis", "label": "Superficial venous thrombosis (SVT)"},
    {"id": "sle_antiphospholipid_positive", "label": "Systemic lupus erythematosus (SLE), positive/unknown antiphospholipid antibodies"},
    {"id": "unexplained_vaginal_bleeding", "label": "Unexplained vaginal bleeding (suspicious for serious condition)"},
    {"id": "uterine_fibroids_no_distortion", "label": "Uterine fibroids, without distortion of the uterine cavity"},
    {"id": "uterine_fibroids_with_distortion", "label": "Uterine fibroids, with distortion of the uterine cavity"},
    {"id": "vaginal_bleeding_irregular", "label": "Irregular vaginal bleeding pattern, without heavy bleeding"},
    {"id": "vaginal_bleeding_heavy", "label": "Heavy or prolonged vaginal bleeding"},
    {"id": "valvular_heart_disease_uncomplicated", "label": "Valvular heart disease, uncomplicated"},
    {"id": "valvular_heart_disease_complicated", "label": "Valvular heart disease, complicated (pulmonary hypertension, atrial fibrillation risk, history of endocarditis)"},
    {"id": "viral_hepatitis_carrier", "label": "Viral hepatitis, carrier or chronic (mild)"},

    # ---- New: round 2, matching additional reference screens ----
    {"id": "diabetes_gestational_history", "label": "Diabetes: history of gestational disease"},
    {"id": "diabetes_nephropathy", "label": "Diabetes with nephropathy, retinopathy, or neuropathy"},
    {"id": "diabetes_other_vascular_20yrs", "label": "Diabetes: other vascular disease, or diabetes of over 20 years' duration"},
    {"id": "endometrial_cancer", "label": "Endometrial cancer"},
    {"id": "endometriosis", "label": "Endometriosis"},
    {"id": "epilepsy", "label": "Epilepsy (not on anticonvulsant therapy)"},
    {"id": "gall_bladder_disease", "label": "Gall bladder disease"},
    {"id": "gestational_trophoblastic_disease", "label": "Gestational trophoblastic disease"},
    {"id": "hiv_high_risk", "label": "High risk of HIV"},
    {"id": "hiv_mild_stage", "label": "HIV, asymptomatic or mild clinical disease (WHO Stage 1 or 2)"},
    {"id": "hiv_severe_stage", "label": "HIV, severe or advanced clinical disease (WHO Stage 3 or 4)"},
]

# condition_id -> { method_id: category }
ELIGIBILITY_MATRIX = {
    # ---- Original 18 ----
    "smoking_35_plus": {"chc": 3, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "obesity_bmi30": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "multiple_cvd_risk": {"chc": 4, "pop": 2, "inj": 3, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "hypertension_controlled": {"chc": 3, "pop": 1, "inj": 2, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "hypertension_uncontrolled": {"chc": 4, "pop": 2, "inj": 3, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "vte_history": {"chc": 4, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "migraine_no_aura": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "migraine_with_aura": {"chc": 4, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "diabetes_no_vascular": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "diabetes_with_vascular": {"chc": 4, "pop": 2, "inj": 3, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "breastfeeding_lt6weeks": {"chc": 4, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "postpartum_non_breastfeeding": {"chc": 3, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 4, "sterilization": 1},
    "current_breast_cancer": {"chc": 4, "pop": 4, "inj": 4, "imp": 4, "lng_iud": 4, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "past_breast_cancer_5yrs": {"chc": 3, "pop": 3, "inj": 3, "imp": 3, "lng_iud": 3, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "cervical_cancer_awaiting_treatment": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 4, "cu_iud": 4, "barrier": 1, "lam": 1, "sterilization": 3},
    "viral_hepatitis_active": {"chc": 3, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "high_risk_hiv_sti": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 1},
    "known_hiv_on_art": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 2},

    # ---- New: expanded to match the reference tool ----
    "anticonvulsant_certain": {"chc": 3, "pop": 3, "inj": 1, "imp": 2, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "anticonvulsant_lamotrigine": {"chc": 3, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "antimicrobial_rifampicin": {"chc": 3, "pop": 3, "inj": 1, "imp": 2, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "antimicrobial_broad_spectrum": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "antiretroviral_therapy": {"chc": 2, "pop": 2, "inj": 1, "imp": 2, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 2},
    "benign_ovarian_tumours": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "bp_unavailable": {"chc": 3, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "breast_undiagnosed_mass": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "breast_benign_disease": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "breast_family_history_cancer": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "breastfeeding_6wk_6mo": {"chc": 3, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "breastfeeding_6mo_plus": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "cardiovascular_disease": {"chc": 4, "pop": 2, "inj": 3, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "cervical_ectropion": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "cin": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "cirrhosis_mild": {"chc": 3, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "cirrhosis_severe": {"chc": 4, "pop": 3, "inj": 3, "imp": 3, "lng_iud": 3, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "history_cholestasis": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "history_pregnancy_hypertension": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "history_pelvic_surgery": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "iron_deficient_anaemia": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 2},
    "known_dyslipidaemias": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "known_thrombogenic_mutations": {"chc": 4, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "liver_tumours_benign": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "liver_tumours_malignant": {"chc": 4, "pop": 3, "inj": 3, "imp": 3, "lng_iud": 3, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "severe_dysmenorrhoea": {"chc": 1, "pop": 1, "inj": 1, "imp": 2, "lng_iud": 1, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 1},
    "sti_current_purulent": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 4, "cu_iud": 4, "barrier": 1, "lam": 1, "sterilization": 3},
    "sti_other": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 1},
    "sickle_cell_disease": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 2},
    "smoking_under_35": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "stroke_history": {"chc": 4, "pop": 2, "inj": 3, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "varicose_veins": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "superficial_venous_thrombosis": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "sle_antiphospholipid_positive": {"chc": 4, "pop": 3, "inj": 3, "imp": 3, "lng_iud": 3, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "unexplained_vaginal_bleeding": {"chc": 3, "pop": 3, "inj": 3, "imp": 3, "lng_iud": 4, "cu_iud": 4, "barrier": 1, "lam": 1, "sterilization": 3},
    "uterine_fibroids_no_distortion": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 2},
    "uterine_fibroids_with_distortion": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 4, "cu_iud": 4, "barrier": 1, "lam": 1, "sterilization": 3},
    "vaginal_bleeding_irregular": {"chc": 1, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "vaginal_bleeding_heavy": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 1},
    "valvular_heart_disease_uncomplicated": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "valvular_heart_disease_complicated": {"chc": 4, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 4},
    "viral_hepatitis_carrier": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},

    # ---- New: round 2 ----
    "diabetes_gestational_history": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "diabetes_nephropathy": {"chc": 4, "pop": 2, "inj": 3, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "diabetes_other_vascular_20yrs": {"chc": 4, "pop": 2, "inj": 3, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 3},
    "endometrial_cancer": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 4, "cu_iud": 4, "barrier": 1, "lam": 1, "sterilization": 3},
    "endometriosis": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 2},
    "epilepsy": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 1},
    "gall_bladder_disease": {"chc": 2, "pop": 2, "inj": 2, "imp": 2, "lng_iud": 2, "cu_iud": 1, "barrier": 1, "lam": 1, "sterilization": 2},
    "gestational_trophoblastic_disease": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 1, "cu_iud": 4, "barrier": 1, "lam": 1, "sterilization": 3},
    "hiv_high_risk": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 1},
    "hiv_mild_stage": {"chc": 1, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 2, "cu_iud": 2, "barrier": 1, "lam": 1, "sterilization": 2},
    "hiv_severe_stage": {"chc": 2, "pop": 1, "inj": 1, "imp": 1, "lng_iud": 3, "cu_iud": 3, "barrier": 1, "lam": 1, "sterilization": 3},
}

# Typical-use failure rates (% of users who become pregnant in a year
# of typical use). Public health statistics, not copyrightable.
EFFECTIVENESS = [
    {"method": "Implants", "typical_use_failure_percent": 0.05,
     "note": "After procedure, little or nothing to do or remember."},
    {"method": "Female sterilization", "typical_use_failure_percent": 0.5,
     "note": "After procedure, little or nothing to do or remember."},
    {"method": "Vasectomy", "typical_use_failure_percent": 0.15,
     "note": "Use another method for the first 3 months."},
    {"method": "IUD (hormonal or copper)", "typical_use_failure_percent": 0.8,
     "note": "After procedure, little or nothing to do or remember."},
    {"method": "Injectables", "typical_use_failure_percent": 6,
     "note": "Repeat injection on schedule, every 2\u20133 months."},
    {"method": "Lactational amenorrhoea method (LAM)", "typical_use_failure_percent": 2,
     "note": "Fully effective only for the first 6 months, with exclusive breastfeeding."},
    {"method": "Patch & vaginal ring", "typical_use_failure_percent": 9,
     "note": "Change/replace on schedule (weekly or monthly)."},
    {"method": "Pills (combined or progestogen-only)", "typical_use_failure_percent": 9,
     "note": "Take at the same time every day."},
    {"method": "Male condom", "typical_use_failure_percent": 18,
     "note": "Use correctly every time you have sex."},
    {"method": "Female condom", "typical_use_failure_percent": 21,
     "note": "Use correctly every time you have sex."},
    {"method": "Withdrawal", "typical_use_failure_percent": 22,
     "note": "Use correctly every time you have sex."},
    {"method": "Fertility awareness methods", "typical_use_failure_percent": 24,
     "note": "Abstain from sex or use condoms on fertile days."},
    {"method": "Spermicides", "typical_use_failure_percent": 28,
     "note": "Use correctly every time you have sex."},
    {"method": "Diaphragm", "typical_use_failure_percent": 12,
     "note": "Use correctly every time you have sex."},
]


@app.route("/conditions", methods=["GET"])
def get_conditions():
    """Returns the list of selectable conditions for the eligibility tool."""
    return jsonify(CONDITIONS)


@app.route("/methods_reference", methods=["GET"])
def get_methods_reference():
    """Returns the list of the 9 WHO-modeled contraceptive methods."""
    return jsonify(METHODS)


@app.route("/effectiveness", methods=["GET"])
def get_effectiveness():
    """Returns typical-use failure rates for common methods."""
    return jsonify(EFFECTIVENESS)


@app.route("/eligibility", methods=["POST", "OPTIONS"])
def check_eligibility():
    """
    Given a list of selected condition IDs, returns the eligibility
    category (1-4) for each of the 9 methods -- taking the most
    restrictive (highest-numbered) category across all selected
    conditions for each method, which is the standard approach when
    someone has more than one relevant condition.

    Body: {"condition_ids": ["smoking_35_plus", "vte_history", ...]}
    """
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}
    selected_ids = data.get("condition_ids") or []

    valid_ids = {c["id"] for c in CONDITIONS}
    unknown = [cid for cid in selected_ids if cid not in valid_ids]
    if unknown:
        return jsonify({"detail": f"Unknown condition id(s): {', '.join(unknown)}"}), 422

    results = []
    for method in METHODS:
        method_id = method["id"]
        worst_category = 1
        for cid in selected_ids:
            cat = ELIGIBILITY_MATRIX.get(cid, {}).get(method_id, 1)
            if cat > worst_category:
                worst_category = cat
        results.append({
            "method_label": method["label"],
            "category": worst_category,
        })

    return jsonify(results)


if __name__ == "__main__":
    # Only used for local testing; PythonAnywhere runs this via WSGI directly.
    app.run(host="0.0.0.0", port=8080)
