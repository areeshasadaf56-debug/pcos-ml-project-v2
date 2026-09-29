"""
feature_reduction_improved.py

An improved feature reduction pass that ACTUALLY searches for better
results, rather than testing a single fixed configuration:

  - SelectKBest: tries multiple values of k (not just "top half").
  - RFE (Recursive Feature Elimination) with a Random Forest estimator:
    tries multiple numbers of features to keep. This was in the original
    spec as an alternative to SelectKBest but wasn't used in Step 5 --
    added here per your instructor's request.
  - PCA: tries multiple explained-variance thresholds (90%, 95%, 99%).

For each dataset, every configuration is evaluated with ALL 7 models
using 5-fold cross-validation (not a single train/test split -- CV
gives a more reliable signal for picking the best config, since a
single split can be lucky/unlucky). The single best (technique, config,
model) combination is reported and compared directly against the
original full-feature and Step 5 results.

Run from the project root (after preprocessing.py has run):
    python src/feature_reduction_improved.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from models import get_models
from feature_reduction import load_split  # reuses the split-loading logic

RANDOM_STATE = 42


def search_selectkbest(X_train, y_train, feature_names, k_values):
    """
    Tries SelectKBest at multiple values of k, evaluating each with all
    7 models via 5-fold CV accuracy on the training set.

    Args:
        X_train (np.ndarray): scaled training features.
        y_train (array-like): training labels.
        feature_names (list): names of the original features.
        k_values (list of int): k values to try.

    Returns:
        list of dict: one entry per (k, model) combination tried, each
            with keys: technique, k, model, cv_mean, cv_std, selected_features.
    """
    results = []
    models = get_models()

    for k in k_values:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_k = selector.fit_transform(X_train, y_train)
        selected = [f for f, m in zip(feature_names, selector.get_support()) if m]

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train_k, y_train, cv=5, scoring="f1")
            results.append({
                "technique": "SelectKBest",
                "k": k,
                "model": name,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "selected_features": selected,
            })
    return results


def search_rfe(X_train, y_train, feature_names, k_values):
    """
    Tries RFE (with a Random Forest estimator) at multiple numbers of
    features to keep, evaluating each with all 7 models via 5-fold CV.

    Args:
        X_train (np.ndarray): scaled training features.
        y_train (array-like): training labels.
        feature_names (list): names of the original features.
        k_values (list of int): numbers of features to keep, to try.

    Returns:
        list of dict: same structure as search_selectkbest, technique="RFE".
    """
    results = []
    models = get_models()

    for k in k_values:
        rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        selector = RFE(estimator=rfe_estimator, n_features_to_select=k)
        X_train_k = selector.fit_transform(X_train, y_train)
        selected = [f for f, m in zip(feature_names, selector.get_support()) if m]

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train_k, y_train, cv=5, scoring="f1")
            results.append({
                "technique": "RFE",
                "k": k,
                "model": name,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "selected_features": selected,
            })
    return results


def search_pca(X_train, y_train, variance_thresholds):
    """
    Tries PCA at multiple explained-variance thresholds, evaluating each
    with all 7 models via 5-fold CV.

    Args:
        X_train (np.ndarray): scaled training features.
        y_train (array-like): training labels.
        variance_thresholds (list of float): thresholds to try, e.g. [0.90, 0.95, 0.99].

    Returns:
        list of dict: technique="PCA", k=number of components used.
    """
    results = []
    models = get_models()

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_train)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    for threshold in variance_thresholds:
        n_components = int(np.argmax(cumulative_variance >= threshold) + 1)
        n_components = max(n_components, 1)

        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train)

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring="f1")
            results.append({
                "technique": f"PCA ({threshold*100:.0f}% var)",
                "k": n_components,
                "model": name,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
                "selected_features": None,
            })
    return results


def get_full_feature_baseline(X_train, y_train):
    """
    Computes the 5-fold CV F1 baseline using ALL original features (no
    reduction), so the search results can be compared against it fairly.

    Args:
        X_train (np.ndarray): scaled training features.
        y_train (array-like): training labels.

    Returns:
        list of dict: one entry per model, technique="Full features (baseline)".
    """
    results = []
    models = get_models()
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        results.append({
            "technique": "Full features (baseline)",
            "k": X_train.shape[1],
            "model": name,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "selected_features": None,
        })
    return results


def run_search(split_dir, results_dir, dataset_label):
    """
    Runs the full config search (baseline + SelectKBest + RFE + PCA) for
    one dataset, saves all results to CSV, and prints the single best
    configuration found.

    Args:
        split_dir (str): directory with the saved train/test split.
        results_dir (str): directory to save results into.
        dataset_label (str): label used in log messages.

    Returns:
        pandas.DataFrame: all results, sorted by CV F1 descending.
    """
    print("\n" + "=" * 60)
    print(f"IMPROVED FEATURE REDUCTION SEARCH: {dataset_label}")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_split(split_dir)
    n_features = X_train.shape[1]

    # Choose sensible k ranges based on how many features this dataset has
    max_k = n_features - 1 if n_features > 1 else 1
    k_values = sorted(set(
        k for k in [2, 3, 5, 7, 10, 15, 20, max_k] if 1 <= k <= max_k
    ))

    print(f"[INFO] Dataset has {n_features} features. Testing k values: {k_values}")

    all_results = []
    all_results += get_full_feature_baseline(X_train, y_train)
    all_results += search_selectkbest(X_train, y_train, feature_names, k_values)
    all_results += search_rfe(X_train, y_train, feature_names, k_values)
    all_results += search_pca(X_train, y_train, [0.90, 0.95, 0.99])

    df = pd.DataFrame(all_results).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "feature_reduction_search_results.csv")
    df.drop(columns=["selected_features"]).to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}")

    print("\n--- Top 10 configurations by 5-fold CV F1-score ---")
    print(df.drop(columns=["selected_features"]).head(10).to_string(index=False))

    baseline_best = df[df["technique"] == "Full features (baseline)"]["cv_f1_mean"].max()
    overall_best = df.iloc[0]

    print(f"\n[INFO] Best full-feature baseline F1: {baseline_best:.4f}")
    print(f"[INFO] Best overall config: {overall_best['technique']} "
          f"(k={overall_best['k']}) + {overall_best['model']} "
          f"-> F1={overall_best['cv_f1_mean']:.4f} (+/-{overall_best['cv_f1_std']:.4f})")

    improvement = overall_best["cv_f1_mean"] - baseline_best
    if improvement > 0.005:
        print(f"[RESULT] Feature reduction IMPROVED results by {improvement:.4f} F1 over full features.")
    elif improvement < -0.005:
        print(f"[RESULT] Best reduction config is still {abs(improvement):.4f} F1 below full-feature baseline.")
    else:
        print(f"[RESULT] Feature reduction made negligible difference ({improvement:+.4f} F1).")

    if overall_best["selected_features"] is not None:
        print(f"[INFO] Best config's selected features: {overall_best['selected_features']}")

    # Plot: best CV F1 achieved per technique family, for a quick visual comparison
    plt.figure(figsize=(9, 5))
    technique_family = df["technique"].apply(lambda t: t.split(" (")[0])
    best_per_family = df.assign(family=technique_family).groupby("family")["cv_f1_mean"].max().sort_values()
    plt.barh(best_per_family.index, best_per_family.values, color="steelblue")
    plt.xlabel("Best 5-fold CV F1-score")
    plt.title(f"Best F1 by Technique - {dataset_label}")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "feature_reduction_search_summary.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[SAVED] {plot_path}")

    return df


if __name__ == "__main__":
    run_search(
        split_dir=os.path.join("data", "cleaned", "split_infertility"),
        results_dir=os.path.join("results", "infertility_dataset"),
        dataset_label="Infertility Dataset",
    )

    run_search(
        split_dir=os.path.join("data", "cleaned", "split_without_infertility"),
        results_dir=os.path.join("results", "without_infertility_dataset"),
        dataset_label="Without-Infertility Dataset",
    )

    print("\n" + "=" * 60)
    print("IMPROVED FEATURE REDUCTION SEARCH COMPLETE FOR BOTH DATASETS")
    print("=" * 60)