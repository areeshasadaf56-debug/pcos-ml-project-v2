"""
models.py

Trains 7 classifiers on each dataset's preprocessed train/test split,
computes Accuracy/Precision/Recall/F1, saves a confusion matrix heatmap
per model, and computes 5-fold cross-validation accuracy (mean +/- std).

Run from the project root:
    python src/models.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it saves plots without needing a display window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score

RANDOM_STATE = 42


def get_models():
    """
    Builds the dictionary of 7 classifiers with sensible default
    hyperparameters. GridSearchCV blocks are included as comments
    below each model for optional later tuning.

    Returns:
        dict: model name -> unfitted sklearn/xgboost estimator.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        # from sklearn.model_selection import GridSearchCV
        # param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
        # grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5)

        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        # param_grid = {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]}
        # grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)

        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        # param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}
        # grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        # param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
        # grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5)

        "KNN": KNeighborsClassifier(n_neighbors=5),
        # param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
        # grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

        "NaiveBayes": GaussianNB(),
        # GaussianNB has essentially no hyperparameters worth grid-searching.

        "XGBoost": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
        # param_grid = {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]}
        # grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric="logloss"), param_grid, cv=5)
    }
    return models


def load_split(split_dir):
    """
    Loads a previously saved train/test split (from preprocessing.py).

    Args:
        split_dir (str): directory containing the saved split files.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    try:
        X_train = np.load(os.path.join(split_dir, "X_train_scaled.npy"))
        X_test = np.load(os.path.join(split_dir, "X_test_scaled.npy"))
        y_train = pd.read_csv(os.path.join(split_dir, "y_train.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(split_dir, "y_test.csv")).iloc[:, 0]
        with open(os.path.join(split_dir, "feature_names.txt")) as f:
            feature_names = f.read().splitlines()
        return X_train, X_test, y_train, y_test, feature_names
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find split files in {split_dir}: {e}")
        print("Make sure you've run src/preprocessing.py successfully first.")
        raise SystemExit(1)


def save_confusion_matrix(y_test, y_pred, model_name, out_dir):
    """
    Saves a confusion matrix heatmap PNG for one model.

    Args:
        y_test (array-like): true labels.
        y_pred (array-like): predicted labels.
        model_name (str): name of the model (used in filename/title).
        out_dir (str): directory to save the PNG into.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No PCOS", "PCOS"], yticklabels=["No PCOS", "PCOS"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] {out_path}")


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, out_dir):
    """
    Trains one model, computes classification metrics, saves a confusion
    matrix, and computes 5-fold cross-validation accuracy.

    Args:
        model: unfitted sklearn/xgboost estimator.
        model_name (str): name of the model.
        X_train, X_test, y_train, y_test: train/test split arrays.
        out_dir (str): directory to save results (plots) into.

    Returns:
        dict: metrics for this model (accuracy, precision, recall, f1,
              cv_mean, cv_std, full classification_report dict).
    """
    print(f"\n--- Training {model_name} ---")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"5-fold CV Accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

    save_confusion_matrix(y_test, y_pred, model_name, out_dir)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "classification_report": report_dict,
    }


def run_all_models(split_dir, results_dir, dataset_label):
    """
    Runs all 7 models on one dataset's split, saving confusion matrices
    and a combined metrics JSON.

    Args:
        split_dir (str): directory containing the saved train/test split.
        results_dir (str): directory to save results into.
        dataset_label (str): label used in log messages.

    Returns:
        dict: model_name -> metrics dict (as returned by evaluate_model).
    """
    print("\n" + "=" * 60)
    print(f"MODELING: {dataset_label}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_split(split_dir)
    models = get_models()

    all_results = {}
    for name, model in models.items():
        all_results[name] = evaluate_model(
            model, name, X_train, X_test, y_train, y_test, results_dir
        )

    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "full_features_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {metrics_path}")

    return all_results


if __name__ == "__main__":
    run_all_models(
        split_dir=os.path.join("data", "cleaned", "split_infertility"),
        results_dir=os.path.join("results", "infertility_dataset"),
        dataset_label="Infertility dataset",
    )

    run_all_models(
        split_dir=os.path.join("data", "cleaned", "split_without_infertility"),
        results_dir=os.path.join("results", "without_infertility_dataset"),
        dataset_label="Without-infertility dataset",
    )

    print("\n" + "=" * 60)
    print("MODELING COMPLETE FOR BOTH DATASETS")
    print("=" * 60)