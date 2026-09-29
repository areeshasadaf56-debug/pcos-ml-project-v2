"""
feature_reduction.py

For each dataset, applies two feature reduction techniques:
  1. PCA - reduced to the number of components needed for 95% explained variance.
  2. SelectKBest (f_classif) - keeps the top-K statistically significant features.

Re-runs all 7 models (same ones from models.py) on both reduced versions,
using the SAME train/test split saved by preprocessing.py, and stores
these results separately from the full-feature results for later comparison.

Run from the project root:
    python src/feature_reduction.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from models import get_models, load_split, evaluate_model

RANDOM_STATE = 42


def apply_pca(X_train, X_test, out_dir, variance_threshold=0.95):
    """
    Fits PCA on the training data only, keeping enough components to
    explain at least `variance_threshold` of the variance. Transforms
    both train and test sets, and saves a scree/explained-variance plot.

    Args:
        X_train (np.ndarray): scaled training features.
        X_test (np.ndarray): scaled test features.
        out_dir (str): directory to save the plot into.
        variance_threshold (float): minimum cumulative explained variance.

    Returns:
        tuple: (X_train_pca, X_test_pca, n_components_used)
    """
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_variance >= variance_threshold) + 1)
    n_components = max(n_components, 1)

    print(f"[INFO] PCA: {n_components} component(s) needed for "
          f"{variance_threshold*100:.0f}% explained variance "
          f"(out of {X_train.shape[1]} original features).")

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.axhline(y=variance_threshold, color="red", linestyle="--",
                label=f"{variance_threshold*100:.0f}% threshold")
    plt.axvline(x=n_components, color="green", linestyle="--",
                label=f"{n_components} components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA - Explained Variance")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "pca_explained_variance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] {out_path}")

    return X_train_pca, X_test_pca, n_components


def apply_selectkbest(X_train, X_test, y_train, feature_names, out_dir, k=None):
    """
    Fits SelectKBest (f_classif) on the training data only, keeping the
    top-k features by ANOVA F-score. If k is None, keeps the top half of
    features (rounded up). Saves a bar chart of feature scores.

    Args:
        X_train (np.ndarray): scaled training features.
        X_test (np.ndarray): scaled test features.
        y_train (array-like): training labels.
        feature_names (list): names of the original features, in order.
        out_dir (str): directory to save the plot into.
        k (int or None): number of top features to keep.

    Returns:
        tuple: (X_train_kbest, X_test_kbest, selected_feature_names)
    """
    if k is None:
        k = max(1, int(np.ceil(len(feature_names) / 2)))

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_kbest = selector.fit_transform(X_train, y_train)
    X_test_kbest = selector.transform(X_test)

    scores = selector.scores_
    selected_mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_names, selected_mask) if m]

    print(f"[INFO] SelectKBest: keeping top {k} of {len(feature_names)} features.")
    print(f"[INFO] Selected features: {selected_features}")

    os.makedirs(out_dir, exist_ok=True)
    order = np.argsort(scores)[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(
        [feature_names[i] for i in order],
        [scores[i] for i in order],
        color=["steelblue" if selected_mask[i] else "lightgray" for i in order],
    )
    plt.xlabel("ANOVA F-score")
    plt.title("Feature Scores (SelectKBest) - blue = selected")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "feature_scores.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] {out_path}")

    return X_train_kbest, X_test_kbest, selected_features


def run_models_on_reduced(X_train, X_test, y_train, y_test, out_dir, version_label):
    """
    Runs all 7 models on a reduced feature set (PCA or SelectKBest output)
    and returns their metrics, without saving per-model confusion matrices
    (those already exist for the full-feature run in models.py).

    Args:
        X_train, X_test, y_train, y_test: reduced train/test split arrays.
        out_dir (str): directory for this dataset's results.
        version_label (str): "PCA" or "SelectKBest".

    Returns:
        dict: model_name -> metrics dict.
    """
    print(f"\n--- Running all 7 models on {version_label}-reduced features ---")
    models = get_models()
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.model_selection import cross_val_score

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        print(f"{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} "
              f"CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}")

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{version_label.lower()}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {out_path}")

    return results


def process_dataset_reduction(split_dir, results_dir, dataset_label):
    """
    Runs the full feature reduction pipeline (PCA + SelectKBest, then
    re-running all 7 models on each) for one dataset.

    Args:
        split_dir (str): directory containing the saved train/test split.
        results_dir (str): directory to save results into.
        dataset_label (str): label used in log messages.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print(f"FEATURE REDUCTION: {dataset_label}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_split(split_dir)

    X_train_pca, X_test_pca, n_components = apply_pca(X_train, X_test, results_dir)
    run_models_on_reduced(X_train_pca, X_test_pca, y_train, y_test, results_dir, "PCA")

    X_train_kbest, X_test_kbest, selected_features = apply_selectkbest(
        X_train, X_test, y_train, feature_names, results_dir
    )
    run_models_on_reduced(X_train_kbest, X_test_kbest, y_train, y_test, results_dir, "SelectKBest")


if __name__ == "__main__":
    process_dataset_reduction(
        split_dir=os.path.join("data", "cleaned", "split_infertility"),
        results_dir=os.path.join("results", "infertility_dataset"),
        dataset_label="Infertility dataset",
    )

    process_dataset_reduction(
        split_dir=os.path.join("data", "cleaned", "split_without_infertility"),
        results_dir=os.path.join("results", "without_infertility_dataset"),
        dataset_label="Without-infertility dataset",
    )

    print("\n" + "=" * 60)
    print("FEATURE REDUCTION COMPLETE FOR BOTH DATASETS")
    print("=" * 60)