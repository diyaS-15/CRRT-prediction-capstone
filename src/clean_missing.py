import os
import pandas as pd

# columns that should use forward-fill (vitals / time-based signals)
VITAL_COLS = [
    "initial_temp_c",
    "carboxyhemoglobin",
    "admission_weight_kg",
    "estimated_dry_weight_kg",
    "total_urine_output_ml_first_24h",
    "total_urine_output_ml_25_48h",
]

def clean_missing_values(
    input_file="synthetic_data.csv",
    output_file="cleaned_data.csv"
):
    path_in = os.path.join("data", input_file)
    df = pd.read_csv(path_in)

    # make sure vitals are numeric
    for c in VITAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # forward-fill vitals (if patient has multiple rows later, this helps)
    # If you have patient id column, add it here:
    # if "record_id" in df.columns:
    #     df = df.sort_values(["record_id", "admission_datetime"])
    #     df[VITAL_COLS] = df.groupby("record_id")[VITAL_COLS].ffill()

    for c in VITAL_COLS:
        if c in df.columns:
            df[c] = df[c].ffill()

    # mean impute remaining numeric columns (excluding vitals already handled)
    num_cols = df.select_dtypes(include="number").columns
    for c in num_cols:
        if c in VITAL_COLS:
            continue
        df[c] = df[c].fillna(df[c].mean())

    # save cleaned
    path_out = os.path.join("data", output_file)
    df.to_csv(path_out, index=False)
    print(f"Saved cleaned file -> {path_out}")

if __name__ == "__main__":
    clean_missing_values(input_file="synthetic_data.csv", output_file="cleaned_data.csv")