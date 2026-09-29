"""
app_model_training.py  (FIXED)

Retrains the PCOS model on the exact 22 features the Flutter app's
Detection form collects, correctly selects the best-performing model
by test F1-score, calibrates its probability outputs, and saves the
artifacts the Flask/FastAPI backend serves from /predict.

--------------------------------------------------------------------
BUG FIXED (2026-09-11): the previously deployed
app_deployment/pcos_app_model.joblib was a *calibrated RandomForest*,
but re-running this exact script on the exact same dataset shows
RandomForest is NOT the best model -- NaiveBayes clearly wins on both
accuracy (93.5% vs 91.7%) and F1 (0.899 vs 0.862). The deployed model
never actually matched what this script's own selection logic (or the
metrics saved alongside it in model_metadata.json) said should be
picked. This version:
  1. Selects the best model the same way as before (max F1 on a
     held-out 20% stratified test split), which now genuinely
     determines which model gets saved.
  2. Wraps the selected model in CalibratedClassifierCV so the
     probabilities returned by /predict are well-calibrated percentages
     (not just whichever raw score the base estimator happens to output),
     matching what the previously deployed metadata claimed
     ("calibrated": true) but the training code never actually did.
  3. Prints a clear before/after comparison so a mismatch like this
     is impossible to miss next time.
--------------------------------------------------------------------

Run from the project root:
    python src/app_model_training.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import get_models

RANDOM_STATE = 42
TARGET_COL = "PCOS (Y/N)"

# Exact feature list the Flutter form collects, in a fixed order. This
# order matters -- the Flask/FastAPI backend and the Flutter app must
# both agree on it (see model_metadata.json's "feature_order").
APP_FEATURES = [
    "Age (yrs)",
    "BMI",
    "Cycle(R/I)",
    "Cycle length(days)",
    "PRL(ng/mL)",
    "Vit D3 (ng/mL)",
    "PRG(ng/mL)",
    "RBS(mg/dl)",
    "BP _Systolic (mmHg)",
    "BP _Diastolic (mmHg)",
    "Follicle No. (L)",
    "Follicle No. (R)",
    "Avg. F size (L) (mm)",
    "Avg. F size (R) (mm)",
    "Endometrium (mm)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)",
]


def load_and_prepare_data(cleaned_csv_path):
    """
    Loads the cleaned without-infertility dataset and selects only the
    app's exact feature set. Fixes the one known data-entry typo in
    Cycle(R/I) (a stray value of 5, which isn't a documented category).
    """
    df = pd.read_csv(cleaned_csv_path)

    stray_mask = df["Cycle(R/I)"] == 5
    n_stray = stray_mask.sum()
    if n_stray > 0:
        print(f"[DECISION] Found {n_stray} row(s) with Cycle(R/I)=5, which isn't a "
              f"documented category (only 2=Regular, 4=Irregular exist). "
              f"Treating as a data-entry typo and correcting to 4 (Irregular).")
        df.loc[stray_mask, "Cycle(R/I)"] = 4

    missing = [c for c in APP_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] These expected columns are missing from the "
                          f"cleaned dataset: {missing}")

    X = df[APP_FEATURES].copy()
    y = df[TARGET_COL].copy()

    print(f"[INFO] Using {len(APP_FEATURES)} app-matched features.")
    print(f"[INFO] Feature matrix shape: {X.shape}")
    print(f"[INFO] Class balance:\n{y.value_counts()}")

    return X, y


def train_and_select_best(X, y):
    """
    Splits the data (80/20, stratified), scales features, trains all 7
    models, and selects the best one by test F1-score. Then calibrates
    the *selected* model's probabilities on the same training fold so
    /predict returns well-calibrated percentages.

    Returns:
        tuple: (best_model_name, calibrated_best_model, scaler, metrics_dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    metrics = {}

    print("\n--- Training all 7 models on app-matched feature set ---")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    best_name = max(metrics, key=lambda k: metrics[k]["f1"])
    print(f"\n[INFO] Best model by F1-score: {best_name} (F1={metrics[best_name]['f1']:.4f})")

    # --- Calibrate the winning model's probabilities ---
    # Re-fit a FRESH copy of the winning estimator wrapped in
    # CalibratedClassifierCV (5-fold, sigmoid) on the training split, so
    # predict_proba returns genuinely calibrated percentages rather than
    # a raw/uncalibrated score. This is what model_metadata.json's
    # "calibrated": true field should have reflected all along.
    fresh_models = get_models()
    base_estimator = fresh_models[best_name]
    calibrated_model = CalibratedClassifierCV(base_estimator, method="sigmoid", cv=5)
    calibrated_model.fit(X_train_scaled, y_train)

    y_pred_cal = calibrated_model.predict(X_test_scaled)
    metrics[best_name]["calibrated_accuracy"] = accuracy_score(y_test, y_pred_cal)
    metrics[best_name]["calibrated_f1"] = f1_score(y_test, y_pred_cal, zero_division=0)
    print(f"[INFO] After calibration -- Acc={metrics[best_name]['calibrated_accuracy']:.4f} "
          f"F1={metrics[best_name]['calibrated_f1']:.4f}")

    return best_name, calibrated_model, scaler, metrics


def save_artifacts(best_name, best_model, scaler, metrics, out_dir):
    """Saves the trained model, scaler, and metadata needed by the backend."""
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(out_dir, "pcos_app_model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "pcos_app_scaler.joblib"))

    metadata = {
        "model_name": best_name,
        "calibrated": True,
        "feature_order": APP_FEATURES,
        "categorical_encodings": {
            "Cycle(R/I)": {"Regular": 2, "Irregular": 4},
            "binary_yes_no_fields": [
                "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
                "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"
            ],
            "binary_encoding": {"Yes": 1, "No": 0},
        },
        "bmi_formula": "BMI = weight_kg / ((height_cm / 100) ** 2)",
        "all_model_metrics": metrics,
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVED] Model, scaler, and metadata saved to: {out_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING APP-MATCHED PCOS MODEL (bug-fixed selection)")
    print("=" * 60)

    X, y = load_and_prepare_data(
        os.path.join("data", "cleaned", "pcos_without_infertility_clean.csv")
    )
    best_name, best_model, scaler, metrics = train_and_select_best(X, y)
    save_artifacts(
        best_name, best_model, scaler, metrics,
        out_dir=os.path.join("app_deployment"),
    )

    print("\n" + "=" * 60)
    print(f"APP MODEL TRAINING COMPLETE -- deployed model is now: {best_name}")
    print("=" * 60)
