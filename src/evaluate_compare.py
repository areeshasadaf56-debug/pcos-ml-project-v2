"""
evaluate_compare.py

For each dataset, builds a single comparison table (7 models x
Accuracy/Precision/Recall/F1 for Full features | PCA | SelectKBest),
a grouped bar chart comparing accuracy across all three feature-set
versions, and a written report.md summarizing findings.

Run from the project root (after models.py and feature_reduction.py):
    python src/evaluate_compare.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAMES = [
    "LogisticRegression", "DecisionTree", "RandomForest",
    "SVM", "KNN", "NaiveBayes", "XGBoost",
]
METRICS = ["accuracy", "precision", "recall", "f1"]
VERSIONS = ["full", "pca", "selectkbest"]
VERSION_LABELS = {"full": "Full features", "pca": "PCA", "selectkbest": "SelectKBest-RFE"}


def load_metrics_json(path):
    """
    Loads a metrics JSON file saved by models.py or feature_reduction.py.

    Args:
        path (str): path to the JSON file.

    Returns:
        dict: model_name -> metrics dict.

    Raises:
        SystemExit: if the file is missing, with a clear message telling
        the user which earlier script needs to be run first.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {path}.")
        print("Make sure models.py and feature_reduction.py have both run successfully.")
        raise SystemExit(1)


def build_comparison_table(results_dir):
    """
    Builds a single dataframe with rows = 7 models, columns = each
    metric for each feature-set version (Full | PCA | SelectKBest-RFE).

    Args:
        results_dir (str): the dataset's results directory containing
            full_features_metrics.json, pca_metrics.json, selectkbest_metrics.json.

    Returns:
        pandas.DataFrame: the comparison table, indexed by model name.
    """
    full_metrics = load_metrics_json(os.path.join(results_dir, "full_features_metrics.json"))
    pca_metrics = load_metrics_json(os.path.join(results_dir, "pca_metrics.json"))
    kbest_metrics = load_metrics_json(os.path.join(results_dir, "selectkbest_metrics.json"))

    version_data = {"full": full_metrics, "pca": pca_metrics, "selectkbest": kbest_metrics}

    rows = []
    for model_name in MODEL_NAMES:
        row = {"Model": model_name}
        for version in VERSIONS:
            m = version_data[version][model_name]
            for metric in METRICS:
                col_name = f"{VERSION_LABELS[version]} - {metric.capitalize()}"
                row[col_name] = round(m[metric], 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


def save_comparison_table(df, results_dir):
    """
    Saves the comparison table to CSV.

    Args:
        df (pandas.DataFrame): the comparison table.
        results_dir (str): directory to save into.

    Returns:
        None
    """
    out_path = os.path.join(results_dir, "comparison_table.csv")
    df.to_csv(out_path)
    print(f"[SAVED] {out_path}")


def plot_accuracy_comparison(results_dir):
    """
    Builds a grouped bar chart comparing Accuracy across Full/PCA/SelectKBest
    for all 7 models, and saves it as a PNG.

    Args:
        results_dir (str): the dataset's results directory.

    Returns:
        None
    """
    full_metrics = load_metrics_json(os.path.join(results_dir, "full_features_metrics.json"))
    pca_metrics = load_metrics_json(os.path.join(results_dir, "pca_metrics.json"))
    kbest_metrics = load_metrics_json(os.path.join(results_dir, "selectkbest_metrics.json"))

    full_acc = [full_metrics[m]["accuracy"] for m in MODEL_NAMES]
    pca_acc = [pca_metrics[m]["accuracy"] for m in MODEL_NAMES]
    kbest_acc = [kbest_metrics[m]["accuracy"] for m in MODEL_NAMES]

    x = np.arange(len(MODEL_NAMES))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, full_acc, width, label="Full features")
    plt.bar(x, pca_acc, width, label="PCA")
    plt.bar(x + width, kbest_acc, width, label="SelectKBest-RFE")

    plt.xticks(x, MODEL_NAMES, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Accuracy Comparison Across Feature Sets")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(results_dir, "accuracy_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] {out_path}")


def find_best_model(df):
    """
    Finds the best-performing model based on Full-features Accuracy.

    Args:
        df (pandas.DataFrame): the comparison table.

    Returns:
        tuple: (model_name, accuracy_value)
    """
    col = "Full features - Accuracy"
    best_model = df[col].idxmax()
    best_acc = df.loc[best_model, col]
    return best_model, best_acc


def summarize_feature_reduction_effect(df):
    """
    Compares average accuracy across the three feature-set versions to
    determine whether feature reduction helped or hurt, on average.

    Args:
        df (pandas.DataFrame): the comparison table.

    Returns:
        dict: average accuracy for each version, and a text conclusion.
    """
    avg_full = df["Full features - Accuracy"].mean()
    avg_pca = df["PCA - Accuracy"].mean()
    avg_kbest = df["SelectKBest-RFE - Accuracy"].mean()

    best_reduced = max(avg_pca, avg_kbest)
    diff = best_reduced - avg_full

    if diff > 0.005:
        conclusion = (
            f"Feature reduction helped on average (best reduced-feature average accuracy "
            f"{best_reduced:.4f} vs full-feature average {avg_full:.4f})."
        )
    elif diff < -0.005:
        conclusion = (
            f"Feature reduction hurt performance slightly on average (best reduced-feature "
            f"average accuracy {best_reduced:.4f} vs full-feature average {avg_full:.4f})."
        )
    else:
        conclusion = (
            f"Feature reduction had negligible effect on average accuracy "
            f"(full: {avg_full:.4f}, PCA: {avg_pca:.4f}, SelectKBest: {avg_kbest:.4f})."
        )

    return {"avg_full": avg_full, "avg_pca": avg_pca, "avg_kbest": avg_kbest, "conclusion": conclusion}


def get_class_balance(cleaned_csv_path, target_col="PCOS (Y/N)"):
    """
    Reads the cleaned dataset and reports class balance for the target column.

    Args:
        cleaned_csv_path (str): path to the cleaned CSV.
        target_col (str): name of the target column.

    Returns:
        pandas.Series: value counts of the target column.
    """
    df = pd.read_csv(cleaned_csv_path)
    return df[target_col].value_counts()


def write_report(results_dir, cleaned_csv_path, dataset_label, df, best_model, best_acc, reduction_summary):
    """
    Writes report.md summarizing the dataset, class balance, best model,
    feature reduction effect, and limitations.

    Args:
        results_dir (str): directory to save the report into.
        cleaned_csv_path (str): path to the cleaned dataset (for class balance).
        dataset_label (str): human-readable dataset name.
        df (pandas.DataFrame): the comparison table.
        best_model (str): name of the best-performing model.
        best_acc (float): accuracy of the best-performing model.
        reduction_summary (dict): output of summarize_feature_reduction_effect().

    Returns:
        None
    """
    class_balance = get_class_balance(cleaned_csv_path)
    n_total = class_balance.sum()
    n_features = len(df.index)  # not used, just for clarity if needed later

    report_lines = [
        f"# Report: {dataset_label}",
        "",
        "## Dataset Overview",
        f"- Total samples after cleaning: {n_total}",
        f"- Class balance (PCOS Y/N): {class_balance.to_dict()}",
        "",
        "## Best-Performing Model",
        f"- **{best_model}** achieved the highest accuracy on full features: **{best_acc:.4f}**.",
        f"- Full comparison table available in `comparison_table.csv`.",
        "",
        "## Effect of Feature Reduction",
        f"- Average accuracy (Full features): {reduction_summary['avg_full']:.4f}",
        f"- Average accuracy (PCA): {reduction_summary['avg_pca']:.4f}",
        f"- Average accuracy (SelectKBest-RFE): {reduction_summary['avg_kbest']:.4f}",
        f"- {reduction_summary['conclusion']}",
        "",
        "## Limitations and Caveats",
        "- Dataset size is modest (541 rows before cleaning), so results may vary across different train/test splits.",
        "- Class balance should be checked above; if imbalanced, accuracy alone can be misleading — precision/recall/F1 in `comparison_table.csv` give a fuller picture.",
        "- PCA components are linear combinations of original features and lose direct clinical interpretability compared to SelectKBest's original-feature selection.",
        "- Default hyperparameters were used for all models; GridSearchCV blocks are available (commented out) in `models.py` for further tuning.",
        "",
    ]

    out_path = os.path.join(results_dir, "report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[SAVED] {out_path}")


def process_dataset_comparison(results_dir, cleaned_csv_path, dataset_label):
    """
    Runs the full comparison and documentation pipeline for one dataset.

    Args:
        results_dir (str): the dataset's results directory.
        cleaned_csv_path (str): path to the dataset's cleaned CSV.
        dataset_label (str): human-readable dataset name.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print(f"COMPARISON & DOCUMENTATION: {dataset_label}")
    print("=" * 60)

    df = build_comparison_table(results_dir)
    print(df)
    save_comparison_table(df, results_dir)

    plot_accuracy_comparison(results_dir)

    best_model, best_acc = find_best_model(df)
    print(f"\n[INFO] Best model (full features): {best_model} (accuracy={best_acc:.4f})")

    reduction_summary = summarize_feature_reduction_effect(df)
    print(f"[INFO] {reduction_summary['conclusion']}")

    write_report(results_dir, cleaned_csv_path, dataset_label, df, best_model, best_acc, reduction_summary)


if __name__ == "__main__":
    process_dataset_comparison(
        results_dir=os.path.join("results", "infertility_dataset"),
        cleaned_csv_path=os.path.join("data", "cleaned", "pcos_infertility_clean.csv"),
        dataset_label="Infertility Dataset",
    )

    process_dataset_comparison(
        results_dir=os.path.join("results", "without_infertility_dataset"),
        cleaned_csv_path=os.path.join("data", "cleaned", "pcos_without_infertility_clean.csv"),
        dataset_label="Without-Infertility Dataset",
    )

    print("\n" + "=" * 60)
    print("COMPARISON & DOCUMENTATION COMPLETE FOR BOTH DATASETS")
    print("=" * 60)