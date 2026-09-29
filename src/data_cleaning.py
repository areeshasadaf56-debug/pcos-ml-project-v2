"""
data_cleaning.py

Cleans the two PCOS datasets (infertility CSV and without-infertility XLSX)
completely separately. Never merges or shares data between them.

Run from the project root:
    python src/data_cleaning.py
"""

import os
import re
import pandas as pd

RANDOM_STATE = 42

# Plausible physiological ranges used for unit sanity checks
HEIGHT_CM_MIN, HEIGHT_CM_MAX = 120, 200
VALID_BLOOD_GROUPS = {11, 12, 13, 14, 15, 16, 17, 18}
BLOOD_GROUP_TEXT_MAP = {
    "A+": 11, "A-": 12, "B+": 13, "B-": 14,
    "O+": 15, "O-": 16, "AB+": 17, "AB-": 18,
}


def load_dataset(path, sheet_name=None):
    """
    Loads a dataset from a CSV or Excel file.

    Args:
        path (str): path to the file.
        sheet_name (str or None): sheet name if an Excel file, else None.

    Returns:
        pandas.DataFrame: the loaded dataset.

    Raises:
        SystemExit: if the file cannot be found or read, prints a clear
        message instead of crashing with a raw traceback.
    """
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"[ERROR] Could not find file at: {path}")
        print("Check that the file exists in data/raw/ with the exact name expected.")
        raise SystemExit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        raise SystemExit(1)


def print_data_summary(df, label):
    """
    Prints shape, dtypes, and null counts for a dataframe.

    Args:
        df (pandas.DataFrame): dataframe to summarize.
        label (str): a description printed above the summary (e.g. "BEFORE cleaning").

    Returns:
        None
    """
    print(f"\n--- {label} ---")
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nNull counts:")
    print(df.isnull().sum())


def drop_empty_unnamed_columns(df):
    """
    Drops columns whose name starts with 'Unnamed' AND are almost entirely empty
    (more than 90% null). Prints which columns were dropped and why.

    Args:
        df (pandas.DataFrame): input dataframe.

    Returns:
        pandas.DataFrame: dataframe with junk unnamed columns removed.
    """
    to_drop = []
    for col in df.columns:
        if str(col).startswith("Unnamed") and df[col].isnull().mean() > 0.9:
            to_drop.append(col)
    if to_drop:
        print(f"[INFO] Dropping stray empty column(s): {to_drop}")
        df = df.drop(columns=to_drop)
    return df


def strip_column_names(df):
    """
    Strips leading/trailing whitespace from all column names.
    Prints a before/after mapping for any column that actually changed.

    Args:
        df (pandas.DataFrame): input dataframe.

    Returns:
        pandas.DataFrame: dataframe with cleaned column names.
    """
    rename_map = {col: col.strip() for col in df.columns if col != col.strip()}
    if rename_map:
        print(f"[INFO] Stripped whitespace from {len(rename_map)} column name(s):")
        for old, new in rename_map.items():
            print(f"    '{old}' -> '{new}'")
        df = df.rename(columns=rename_map)
    return df


def fix_height_units(df, height_col):
    """
    Asserts Height(Cm) values are within a plausible centimeter range.
    If any values look like they were entered in a different unit
    (e.g. under 10, which would suggest meters), converts them to cm.

    Args:
        df (pandas.DataFrame): input dataframe.
        height_col (str): the exact column name for height.

    Returns:
        pandas.DataFrame: dataframe with height values verified/corrected.
    """
    if height_col not in df.columns:
        return df

    out_of_range = df[
        (df[height_col] < HEIGHT_CM_MIN) | (df[height_col] > HEIGHT_CM_MAX)
    ]
    if len(out_of_range) > 0:
        print(f"[WARNING] {len(out_of_range)} row(s) have {height_col} outside "
              f"{HEIGHT_CM_MIN}-{HEIGHT_CM_MAX} cm. Attempting to fix (assuming meters entered by mistake).")
        mask = df[height_col] < 10  # meters, e.g. 1.56
        df.loc[mask, height_col] = df.loc[mask, height_col] * 100
    else:
        print(f"[OK] All {height_col} values are within plausible range "
              f"({HEIGHT_CM_MIN}-{HEIGHT_CM_MAX} cm). No conversion needed.")
    return df


def fix_yes_no_columns(df):
    """
    Finds all columns with '(Y/N)' in their name, asserts values are
    strictly 0/1, and coerces 'Yes'/'No' text to 1/0 if found.

    Args:
        df (pandas.DataFrame): input dataframe.

    Returns:
        pandas.DataFrame: dataframe with Yes/No columns normalized to 0/1.
    """
    yn_cols = [c for c in df.columns if "(Y/N)" in c]
    for col in yn_cols:
        if df[col].dtype == object:
            print(f"[INFO] Coercing text values in '{col}' to 0/1.")
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
            )
        bad_values = df[~df[col].isin([0, 1]) & df[col].notnull()]
        if len(bad_values) > 0:
            print(f"[WARNING] Column '{col}' has values outside {{0,1}}: "
                  f"{bad_values[col].unique().tolist()}")
    return df


def fix_blood_group(df, bg_col="Blood Group"):
    """
    Asserts Blood Group values are strictly in {11..18}. Maps common
    text blood-group codes (A+, A-, etc.) to the numeric code if found.

    Args:
        df (pandas.DataFrame): input dataframe.
        bg_col (str): the blood group column name.

    Returns:
        pandas.DataFrame: dataframe with Blood Group normalized.
    """
    if bg_col not in df.columns:
        return df

    if df[bg_col].dtype == object:
        print(f"[INFO] Mapping text blood group codes in '{bg_col}' to numeric codes.")
        df[bg_col] = df[bg_col].astype(str).str.strip().map(BLOOD_GROUP_TEXT_MAP).fillna(df[bg_col])

    invalid = df[~df[bg_col].isin(VALID_BLOOD_GROUPS)]
    if len(invalid) > 0:
        print(f"[WARNING] {len(invalid)} row(s) have invalid '{bg_col}' values: "
              f"{invalid[bg_col].unique().tolist()}")
    else:
        print(f"[OK] All '{bg_col}' values are valid ({sorted(VALID_BLOOD_GROUPS)}).")
    return df


def split_combined_bp_column(df):
    """
    Looks for a combined blood-pressure column (e.g. containing '120/80'
    style strings) and splits it into separate systolic/diastolic columns
    if BP _Systolic / BP _Diastolic don't already exist as numeric columns.

    Args:
        df (pandas.DataFrame): input dataframe.

    Returns:
        pandas.DataFrame: dataframe with systolic/diastolic guaranteed separate.
    """
    has_systolic = any("Systolic" in c for c in df.columns)
    has_diastolic = any("Diastolic" in c for c in df.columns)
    if has_systolic and has_diastolic:
        print("[OK] BP Systolic/Diastolic already exist as separate columns.")
        return df

    combined_col = None
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).str.contains(r"^\d{2,3}/\d{2,3}$").any():
            combined_col = c
            break

    if combined_col:
        print(f"[INFO] Splitting combined BP column '{combined_col}' into systolic/diastolic.")
        split_vals = df[combined_col].astype(str).str.split("/", expand=True)
        df["BP _Systolic (mmHg)"] = pd.to_numeric(split_vals[0], errors="coerce")
        df["BP _Diastolic (mmHg)"] = pd.to_numeric(split_vals[1], errors="coerce")
        df = df.drop(columns=[combined_col])
    return df


def clean_numeric_text_artifacts(series):
    """
    Cleans stray text artifacts from a numeric-looking column, such as
    a trailing extra period (e.g. '1.99.' -> '1.99'), before numeric coercion.

    Args:
        series (pandas.Series): the column to clean.

    Returns:
        pandas.Series: series with obvious text artifacts fixed (still as strings).
    """
    def fix_value(v):
        if isinstance(v, str):
            v = v.strip()
            # Fix a trailing extra period, e.g. "1.99." -> "1.99"
            if re.match(r"^\d+\.\d+\.$", v):
                v = v[:-1]
        return v
    return series.apply(fix_value)


def fix_beta_hcg_duplication(df, col1, col2):
    """
    Coerces both beta-HCG columns to numeric (fixing text artifacts first).
    If one column is null but the other isn't for the same row, fills the
    null one with the other's value. Prints how many rows were affected.

    Args:
        df (pandas.DataFrame): input dataframe.
        col1 (str): name of the 'I beta-HCG' column.
        col2 (str): name of the 'II beta-HCG' column.

    Returns:
        pandas.DataFrame: dataframe with beta-HCG columns cleaned and cross-filled.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return df

    df[col1] = pd.to_numeric(clean_numeric_text_artifacts(df[col1]), errors="coerce")
    df[col2] = pd.to_numeric(clean_numeric_text_artifacts(df[col2]), errors="coerce")

    mask_fill_1 = df[col1].isnull() & df[col2].notnull()
    mask_fill_2 = df[col2].isnull() & df[col1].notnull()

    n_filled_1 = mask_fill_1.sum()
    n_filled_2 = mask_fill_2.sum()

    if n_filled_1 > 0:
        df.loc[mask_fill_1, col1] = df.loc[mask_fill_1, col2]
    if n_filled_2 > 0:
        df.loc[mask_fill_2, col2] = df.loc[mask_fill_2, col1]

    print(f"[INFO] Beta-HCG cross-fill: {n_filled_1} row(s) filled in '{col1}' "
          f"from '{col2}', {n_filled_2} row(s) filled in '{col2}' from '{col1}'.")
    return df


def resolve_remaining_nulls(df, dataset_label):
    """
    Handles any remaining NaNs after all rule-based fixes, using explicit,
    documented decisions rather than silently guessing:

      - AMH(ng/mL): if a value can't be parsed as a number (e.g. stray text),
        it's coerced to NaN. Since there's no reciprocal column to recover
        it from, the row is dropped (documented below).
      - Marraige Status (Yrs): missing values are filled with 0, on the
        assumption the field was left blank because it doesn't apply
        (e.g. not yet married).
      - Fast food (Y/N): missing values are filled with the column's mode
        (most common answer), as a neutral default for a single missing entry.
      - Any other NaN found after these rules is printed (row + column)
        and NOT invented — you'll see it in the output if it happens.

    Args:
        df (pandas.DataFrame): input dataframe.
        dataset_label (str): name used in log messages.

    Returns:
        pandas.DataFrame: dataframe with all nulls resolved.
    """
    # --- AMH: coerce to numeric, drop unrecoverable rows ---
    amh_col = next((c for c in df.columns if c.strip().startswith("AMH")), None)
    if amh_col:
        before_nulls = df[amh_col].isnull().sum()
        df[amh_col] = pd.to_numeric(df[amh_col], errors="coerce")
        after_nulls = df[amh_col].isnull().sum()
        newly_bad = after_nulls - before_nulls
        if newly_bad > 0:
            bad_rows = df[df[amh_col].isnull()]
            print(f"[DECISION] {dataset_label}: {newly_bad} row(s) had a non-numeric "
                  f"value in '{amh_col}' with no way to recover it. Dropping these row(s) "
                  f"(Patient File No.: "
                  f"{bad_rows['Patient File No.'].tolist() if 'Patient File No.' in df.columns else bad_rows.index.tolist()}).")
            df = df.dropna(subset=[amh_col])

    # --- Marraige Status (Yrs): fill with 0 ---
    marriage_col = next((c for c in df.columns if "Marraige Status" in c), None)
    if marriage_col and df[marriage_col].isnull().sum() > 0:
        n = df[marriage_col].isnull().sum()
        print(f"[DECISION] {dataset_label}: filling {n} missing value(s) in "
              f"'{marriage_col}' with 0 (assumed not applicable / not yet married).")
        df[marriage_col] = df[marriage_col].fillna(0)

    # --- Fast food (Y/N): fill with mode ---
    fastfood_col = next((c for c in df.columns if "Fast food" in c), None)
    if fastfood_col and df[fastfood_col].isnull().sum() > 0:
        n = df[fastfood_col].isnull().sum()
        mode_val = df[fastfood_col].mode()[0]
        print(f"[DECISION] {dataset_label}: filling {n} missing value(s) in "
              f"'{fastfood_col}' with the column mode ({mode_val}).")
        df[fastfood_col] = df[fastfood_col].fillna(mode_val)

    # --- Final catch-all: anything still null gets flagged, not invented ---
    remaining = df.isnull().sum()
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        print(f"[STOP] {dataset_label}: unresolved NaNs remain after all rules:")
        print(remaining)
        raise SystemExit(
            f"Unresolved NaNs in {dataset_label} — please tell Claude which rows/columns "
            f"these are so a rule can be added, rather than guessing a value."
        )

    return df


def clean_infertility_csv():
    """
    Cleans the PCOS_infertility.csv dataset end-to-end and saves the result.

    Returns:
        pandas.DataFrame: the cleaned infertility dataset.
    """
    print("\n" + "=" * 60)
    print("CLEANING: PCOS_infertility.csv")
    print("=" * 60)

    path = os.path.join("data", "raw", "PCOS_infertility.csv")
    df = load_dataset(path)

    print_data_summary(df, "BEFORE cleaning (infertility)")

    df = strip_column_names(df)
    df = drop_empty_unnamed_columns(df)

    col1 = next(c for c in df.columns if c.strip().startswith("I") and "beta-HCG" in c and not c.strip().startswith("II"))
    col2 = next(c for c in df.columns if c.strip().startswith("II") and "beta-HCG" in c)
    df = fix_beta_hcg_duplication(df, col1, col2)

    df = resolve_remaining_nulls(df, "infertility dataset")

    print_data_summary(df, "AFTER cleaning (infertility)")

    assert df.isnull().sum().sum() == 0, "There are still NaNs after cleaning!"

    os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)
    out_path = os.path.join("data", "cleaned", "pcos_infertility_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}  (shape: {df.shape})")

    return df


def clean_without_infertility_xlsx():
    """
    Cleans the PCOS_data_without_infertility.xlsx dataset end-to-end and saves the result.

    Returns:
        pandas.DataFrame: the cleaned without-infertility dataset.
    """
    print("\n" + "=" * 60)
    print("CLEANING: PCOS_data_without_infertility.xlsx")
    print("=" * 60)

    path = os.path.join("data", "raw", "PCOS_data_without_infertility.xlsx")
    df = load_dataset(path, sheet_name="Full_new")

    print_data_summary(df, "BEFORE cleaning (without infertility)")

    df = strip_column_names(df)
    df = drop_empty_unnamed_columns(df)

    height_col = next((c for c in df.columns if c.startswith("Height")), None)
    df = fix_height_units(df, height_col)

    df = fix_yes_no_columns(df)
    df = fix_blood_group(df)
    df = split_combined_bp_column(df)

    col1 = next(c for c in df.columns if c.strip().startswith("I") and "beta-HCG" in c and not c.strip().startswith("II"))
    col2 = next(c for c in df.columns if c.strip().startswith("II") and "beta-HCG" in c)
    df = fix_beta_hcg_duplication(df, col1, col2)

    df = resolve_remaining_nulls(df, "without-infertility dataset")

    print_data_summary(df, "AFTER cleaning (without infertility)")

    assert df.isnull().sum().sum() == 0, "There are still NaNs after cleaning!"

    os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)
    out_path = os.path.join("data", "cleaned", "pcos_without_infertility_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path}  (shape: {df.shape})")

    return df


if __name__ == "__main__":
    clean_infertility_csv()
    clean_without_infertility_xlsx()
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE FOR BOTH DATASETS")
    print("=" * 60)