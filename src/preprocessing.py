import pandas as pd
import os
import numpy as np 
import sys 
from typing import Tuple

# Public API, names other files can access 
__all__ = [
    "load_and_preprocess",
    "engineer_features",
    "FEATURE_COLS",
    "TARGET_COL",
]


# CONSTANTS 
TARGET_COL = "crrt_within_48h"
# import in train_model so files are same with what cols go into 
FEATURE_COLS = [
    "age",
    "tbsa_2nd_3rd",
    "inhalation_flag",
    "revised_baux_score",
    "hours_injury_to_admission",
    "late_admission_flag",
    "fluid_balance_24h",
    "fluid_overload_flag",
    "burn_severity_tier",
    "urine_output_per_kg",
    "carboxyhemoglobin_risk_flag",
    "hypothermia_flag"
]

# HELPER FUNCTIONS 
def load_burn_data(file_name="synthetic_data.csv"):
    # dynamic path construction 
    path = os.path.join("data", file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"BCQP data not found at {path}")
    return pd.read_csv(path)

def normalize_yes_no(val):
    """Convert Yes/No/1/0 string variants to integer 1 or 0."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return 1
    if s in {"no", "n", "false", "0"}:
        return 0
    return None

def assign_severity_tier(score):
    if pd.isna(score):
        return np.nan
    if score < 60:
        return 0   
    elif score < 100:
        return 1   
    elif score < 140:
        return 2   
    else:
        return 3  

# function so other files can call without rerunning 
def engineer_features(dfog=pd.DataFrame) -> pd.DataFrame: 
    df = dfog.copy() # copy df so not work on original 
    
    # FEATURES 

    # Revised Baux Score Feature (Age + TBSA + 17), 17=clinically validated value for AKI 
    df["inhalation_injury"] = df["inhalation_injury"].str.strip().str.lower()
    df["inhalation_flag"] = df["inhalation_injury"].map(lambda x: 1 if x in {"yes", "y", "true", "1"} else 0)
    df["revised_baux_score"] = df["age"] + df["tbsa_2nd_3rd"] + (17 * df["inhalation_flag"])
    print("\nRevised Baux Score feature:")
    print(df[["age", "tbsa_2nd_3rd", "inhalation_flag", "revised_baux_score"]].head())

    # Hours from Injury to Admission, clinically >6=late presentation
    df["injury_datetime"]    = pd.to_datetime(df["injury_datetime"],    errors="coerce")
    df["admission_datetime"] = pd.to_datetime(df["admission_datetime"], errors="coerce")
    df["hours_injury_to_admission"] = (df["admission_datetime"] - df["injury_datetime"]).dt.total_seconds() / 3600.0
    # flags patients arriving >6 hours after injury
    df["late_admission_flag"] = (df["hours_injury_to_admission"] > 6).astype(int)
    print("\nHours Injury to Admission feature:")
    print(df[["injury_datetime", "admission_datetime","hours_injury_to_admission", "late_admission_flag"]].head())

    # CRRT within 48hrs labels, 1 if CRRT within 48hrs, 0 if no CRRT, drop rows where src col missing=NaN + dropped at test/train
    a = df["crrt_first_24h"].map(normalize_yes_no)
    b = df["crrt_25_48h"].map(normalize_yes_no)
    df["crrt_within_48h"] = [np.nan if (ai is None and bi is None) else int((ai == 1) or (bi == 1))for ai, bi in zip(a, b)]
    print("\nCRRT Within 48h Target feature:")
    print(df["crrt_within_48h"].value_counts(dropna=False))
    print(f"  Positive rate: {df['crrt_within_48h'].mean():.1%} "f"| Missing: {df['crrt_within_48h'].isna().sum()}")

    # Total Fluid Balance, large pos fluid balance= CRRT predictor
    df["total_crystalloid_ml_first_24h"]  = pd.to_numeric(df["total_crystalloid_ml_first_24h"],  errors="coerce")
    df["total_colloid_ml_first_24h"]      = pd.to_numeric(df["total_colloid_ml_first_24h"],      errors="coerce")
    df["total_urine_output_ml_first_24h"] = pd.to_numeric(df["total_urine_output_ml_first_24h"], errors="coerce")
    df["fluid_balance_24h"] = (df["total_crystalloid_ml_first_24h"].fillna(0) + df["total_colloid_ml_first_24h"].fillna(0) - df["total_urine_output_ml_first_24h"].fillna(0))
    # positive = fluid overloaded, negative = net output
    df["fluid_overload_flag"] = (df["fluid_balance_24h"] > 0).astype(int)
    print("\nFluid Balance (First 24h) feature:")
    print(df[["total_crystalloid_ml_first_24h", "total_colloid_ml_first_24h","total_urine_output_ml_first_24h", "fluid_balance_24h","fluid_overload_flag"]].head())

    # Burn severity tier, from revised baux score low<60, moderate 60-99, high:100-139, critical>=140 (from published burn mortality risk)
    df["burn_severity_tier"] = df["revised_baux_score"].map(assign_severity_tier)
    tier_labels = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}
    print("\nBurn Severity Tier feature:")
    print(df["burn_severity_tier"].map(tier_labels).value_counts(dropna=False))

    # Urine output per kilogram of bodyweight, AKI standard <0.5 mL/kg/hr for 6+ hours signals kidney failure 
    # normalize weight 
    df["weight_kg"] = pd.to_numeric(df.get("weight_kg"), errors="coerce")
    df["urine_output_per_kg"] = (df["total_urine_output_ml_first_24h"] / df["weight_kg"].replace(0, np.nan))
    # flag when below aki standard 
    df["low_urine_output_flag"] = (df["urine_output_per_kg"] < 12).astype(int)  
    print("\nUrine Output per kg feature:")
    print(df[["total_urine_output_ml_first_24h", "weight_kg", "urine_output_per_kg", "low_urine_output_flag"]].head())

    # Carboxyhemoglobin risk flag, clinical threshold >=25% for CO poisioning 
    df["carboxyhemoglobin"] = pd.to_numeric(df["carboxyhemoglobin"], errors="coerce")
    df["carboxyhemoglobin_risk_flag"] = (df["carboxyhemoglobin"] >= 25).astype(int)
    print("\nCarboxyhemoglobin Risk Flag feature:")
    print(df[["carboxyhemoglobin", "carboxyhemoglobin_risk_flag"]].head())

    # Hypothermia risk fglag, <36 clinical indicator of high AKI risk 
    df["initial_temp_c"] = pd.to_numeric(df["initial_temp_c"], errors="coerce")
    df["hypothermia_flag"] = (df["initial_temp_c"] < 36.0).astype(int)
    print("\nHypothermia Flag feature:")
    print(df[["initial_temp_c", "hypothermia_flag"]].head())

    # Engineered Features 
    engineered_cols = [
        "baux_score",                
        "inhalation_flag",           
        "revised_baux_score",        
        "hours_injury_to_admission", 
        "late_admission_flag",       
        "crrt_within_48h",           
        "fluid_balance_24h",         
        "fluid_overload_flag",       
        "burn_severity_tier",  
        "urine_output_per_kg",
        "carboxyhemoglobin_risk_flag",
        "hypothermia_flag"

    ]
    # Printing 
    print("Engineered features: ")
    for col in engineered_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"  {col:<35} mean={s.mean():.2f}  missing={s.isna().sum()}")
    return df 

# Load + preprocess data 
def load_and_preprocess(csv_path: str = "data/synthetic_data.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return engineer_features(df)

# Main - execution in this file 
if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_data.csv"
    df = load_and_preprocess(csv_path)





