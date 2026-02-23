import pandas as pd

def clean_missing_vitals(
    df: pd.DataFrame,
    patient_id_col: str = "record_id",
    time_col: str = "admission_datetime",
    vital_cols=None,
    ffill_limit: int = 2,
    mean_threshold: float = 0.20,
    drop_threshold: float = 0.70,
):
    """
    Strategy:
    - If missing > drop_threshold => drop column
    - Else if missing <= mean_threshold => mean/median fill (median safer)
    - Else => forward-fill within patient sorted by time (limited)
    """

    if vital_cols is None:
        # Example guess list (edit to match your dataset)
        vital_cols = ["initial_temp_c", "initial_gcs"]

    # Make sure time col is datetime if it exists
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    methods_used = {}

    for col in vital_cols:
        if col not in df.columns:
            continue

        miss_rate = df[col].isna().mean()

        # Drop if too missing
        if miss_rate > drop_threshold:
            df = df.drop(columns=[col])
            methods_used[col] = f"DROPPED (missing={miss_rate:.2%})"
            continue

        # Mean/median if low missing
        if miss_rate <= mean_threshold:
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            methods_used[col] = f"MEDIAN (missing={miss_rate:.2%})"
            continue

        # Otherwise forward-fill within patient if we can sort by time
        if patient_id_col in df.columns and time_col in df.columns:
            df = df.sort_values([patient_id_col, time_col])
            df[col] = df.groupby(patient_id_col)[col].ffill(limit=ffill_limit)
            # If still missing after ffill, use median as fallback
            df[col] = df[col].fillna(df[col].median())
            methods_used[col] = f"FFILL(limit={ffill_limit}) + MEDIAN fallback (missing={miss_rate:.2%})"
        else:
            # No time ordering available -> fallback median
            df[col] = df[col].fillna(df[col].median())
            methods_used[col] = f"MEDIAN fallback (no time/patient order) (missing={miss_rate:.2%})"

    return df, methods_used


if __name__ == "__main__":
    # Example usage (edit filename/columns)
    df = pd.read_csv("data/synthetic_data.csv")
    cleaned, methods = clean_missing_vitals(
        df,
        patient_id_col="record_id",
        time_col="admission_datetime",
        vital_cols=["initial_temp_c", "initial_gcs"],  # add more vitals later
    )

    print("Imputation methods used:")
    for k, v in methods.items():
        print(f"- {k}: {v}")

    cleaned.to_csv("data/cleaned_data.csv", index=False)
    print("Saved: data/cleaned_data.csv")