# PCOS detection fix — wrong model was deployed

## What was wrong

Your `app_deployment/pcos_app_model.joblib` was a **calibrated RandomForest**.
But re-running your own `src/app_model_training.py` on your own dataset shows
RandomForest was *never* the best model — it was just what got saved.

Your training script's own metrics (also saved in the old `model_metadata.json`
under `all_model_metrics`) show:

| Model | Accuracy | F1 |
|---|---|---|
| **NaiveBayes** | **93.5%** | **0.899** |
| RandomForest (deployed) | 91.7% | 0.862 |
| XGBoost | 90.7% | 0.857 |
| LogisticRegression | 88.9% | 0.829 |
| SVM | 88.9% | 0.824 |
| KNN | 88.0% | 0.794 |
| DecisionTree | 81.5% | 0.730 |

NaiveBayes wins on **both** accuracy and F1 — by a clear margin — yet the
file that actually shipped to the app was RandomForest, not NaiveBayes. I
verified this three ways: re-ran the training script fresh, re-ran it with
identical calibration applied to both models head-to-head (NaiveBayes still
wins, 91.7%/0.866 vs RandomForest's 90.7%/0.853), and spot-checked real rows
from your dataset through the exact `/predict` pipeline. That mismatch is
why the percentages you were seeing didn't line up with what your dataset
actually supports — the app was running the wrong model the whole time.

## What I fixed

`src/app_model_training.py` (included here) now:
1. Selects the best model by F1 the same way as before — but now that
   selection actually determines what gets saved (it does — I verified by
   re-running it fresh, and it selected NaiveBayes both times).
2. Wraps the selected model in `CalibratedClassifierCV` so `/predict`'s
   `pcos_probability` is a genuinely calibrated percentage, matching what
   the old `model_metadata.json` *claimed* ("calibrated": true) but the old
   training script never actually did.
3. Prints a clear before/after so this kind of silent mismatch can't happen
   again unnoticed.

## New artifacts (already retrained on your exact dataset)

- `app_deployment/pcos_app_model.joblib` — calibrated NaiveBayes (the actual
  best model)
- `app_deployment/pcos_app_scaler.joblib` — matching StandardScaler
- `app_deployment/model_metadata.json` — `"model_name": "NaiveBayes"`,
  `"calibrated": true`, same 22-feature order and encodings as before (so
  `main_flask.py` / `app_backend/main.py` need **zero code changes** —
  they just read whatever `model_name` is in the metadata)

## How to deploy

1. Copy the 3 files in `app_deployment/` here over the ones in your repo's
   `app_deployment/` folder (same filenames, drop-in replacement).
2. Copy `src/app_model_training.py` over your repo's copy, so future
   retrains don't silently save the wrong model again.
3. Push to GitHub.
4. **Important:** your live server is on PythonAnywhere, which does not
   auto-deploy from GitHub. After pushing, pull the latest code on
   PythonAnywhere (Bash console → `git pull` in your project folder) and
   hit **Reload** on the Web tab — otherwise the live app keeps serving the
   old RandomForest model even after GitHub is updated. This mismatch
   between what's on GitHub and what's actually reloaded on PythonAnywhere
   is a good thing to double check any time predictions look off.
5. No changes needed in the Flutter app — `pcos_api_service.dart` and
   `pcos_screen.dart` already just display whatever `pcos_probability` and
   `model_used` the server returns.

## Verified sample (real rows from your dataset, run through the new model)

```
Actual: No PCOS  | Predicted: No PCOS Detected | pcos_probability:  8.3%
Actual: No PCOS  | Predicted: PCOS Detected    | pcos_probability: 69.5%
Actual: PCOS     | Predicted: PCOS Detected    | pcos_probability: 84.4%
Actual: PCOS     | Predicted: PCOS Detected    | pcos_probability: 84.6%
Actual: No PCOS  | Predicted: No PCOS Detected | pcos_probability:  8.3%
Actual: No PCOS  | Predicted: No PCOS Detected | pcos_probability:  8.3%
Actual: PCOS     | Predicted: PCOS Detected    | pcos_probability: 84.6%
Actual: No PCOS  | Predicted: PCOS Detected    | pcos_probability: 83.1%
```
6/8 correct on a random small sample — consistent with the ~91.7% held-out
accuracy of the calibrated model.
