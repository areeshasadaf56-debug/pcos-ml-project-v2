"""
preprocessing.py

Splits each cleaned dataset into train/test sets (80/20, stratified),
scales numeric features with StandardScaler (fit on train only), and
saves the resulting arrays to disk so every later script (models,
feature reduction, evaluation) reuses the exact same split for a fair
comparison.

Run from the project root:
    python src/preprocessing.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET_COL = "PCOS (Y/N)"


def load_clean_dataset(path):
    """
    Loads a cleaned CSV dataset from disk.

    Args:
        path (str): path to the cleaned CSV file.

    Returns:
        pandas.DataFrame: the loaded dataset.

    Raises:
        SystemExit: if the file is missing, prints a clear message.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find cleaned file at: {path}")
        print("Make sure you've run src/data_cleaning.py successfully first.")
        raise SystemExit(1)


def split_features_target(df, target_col, drop_cols=None):
    """
    Separates a dataframe into feature matrix X and target vector y.
    Drops ID-style columns that shouldn't be used as predictive features.

    Args:
        df (pandas.DataFrame): the full cleaned dataframe.
        target_col (str): name of the target column.
        drop_cols (list or None): extra non-feature columns to drop
            (e.g. ID columns). Defaults to common ID column names if None.

    Returns:
        tuple: (X pandas.DataFrame, y pandas.Series)
    """
    if drop_cols is None:
        drop_cols = ["Sl. No", "Patient File No."]

    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[target_col] + existing_drop_cols)
    y = df[target_col]

    print(f"[INFO] Dropped non-feature columns: {existing_drop_cols}")
    print(f"[INFO] Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    print(f"[INFO] Class balance in target:\n{y.value_counts()}")

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """
    Performs an 80/20 stratified train/test split, then scales all
    numeric features with StandardScaler fit only on the training set.

    Args:
        X (pandas.DataFrame): feature matrix.
        y (pandas.Series): target vector.
        test_size (float): proportion of data to hold out for testing.
        random_state (int): seed for reproducibility.

    Returns:
        dict: contains X_train_scaled, X_test_scaled, y_train, y_test,
              the fitted scaler, and the original (unscaled) train/test
              splits for reference.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[INFO] Train class balance:\n{y_train.value_counts()}")
    print(f"[INFO] Test class balance:\n{y_test.value_counts()}")

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "feature_names": X.columns.tolist(),
        "scaler": scaler,
    }


def save_split(split_dict, out_dir):
    """
    Saves the train/test split arrays and feature names to disk as .npy/.csv
    files so later scripts can load the exact same split without re-splitting.

    Args:
        split_dict (dict): output of split_and_scale().
        out_dir (str): directory to save the split into.

    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train_scaled.npy"), split_dict["X_train_scaled"])
    np.save(os.path.join(out_dir, "X_test_scaled.npy"), split_dict["X_test_scaled"])
    split_dict["y_train"].to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    split_dict["y_test"].to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    with open(os.path.join(out_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(split_dict["feature_names"]))

    print(f"[SAVED] Split arrays and feature names saved to: {out_dir}")


def process_dataset(cleaned_path, out_dir, label):
    """
    Runs the full preprocessing pipeline (load -> split features/target ->
    split/scale -> save) for one dataset.

    Args:
        cleaned_path (str): path to the cleaned CSV file.
        out_dir (str): directory to save the split into.
        label (str): dataset label used in log messages.

    Returns:
        dict: the split dictionary returned by split_and_scale().
    """
    print("\n" + "=" * 60)
    print(f"PREPROCESSING: {label}")
    print("=" * 60)

    df = load_clean_dataset(cleaned_path)
    X, y = split_features_target(df, TARGET_COL)
    split_dict = split_and_scale(X, y)
    save_split(split_dict, out_dir)

    return split_dict


if __name__ == "__main__":
    process_dataset(
        cleaned_path=os.path.join("data", "cleaned", "pcos_infertility_clean.csv"),
        out_dir=os.path.join("data", "cleaned", "split_infertility"),
        label="Infertility dataset",
    )

    process_dataset(
        cleaned_path=os.path.join("data", "cleaned", "pcos_without_infertility_clean.csv"),
        out_dir=os.path.join("data", "cleaned", "split_without_infertility"),
        label="Without-infertility dataset",
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE FOR BOTH DATASETS")
    print("=" * 60)
    