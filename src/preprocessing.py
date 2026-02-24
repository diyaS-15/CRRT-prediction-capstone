import pandas as pd
import os
import numpy as np 

# CONSTANTS 
TARGET_COL = "crrt_within_48h"
# import in train_model so files are same with what cols go into 
FEATURE_COLS = [
    "age",
    "tbsa_2nd_3rd",
    "baux_score",
    "inhalation_flag",
    "revised_baux_score",
    "hours_injury_to_admission",
    "late_admission_flag",
    "fluid_balance_24h",
    "fluid_overload_flag",
    "burn_severity_tier",
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

df = load_burn_data()

# FEATURES 

print("COLUMNS:", df.columns.tolist())
# --- Feature engineering: Baux Score = Age + %TBSA ---
# Update these column names if your CSV uses different names
# --- Feature engineering: Baux Score = age + %TBSA ---
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["tbsa_2nd_3rd"] = pd.to_numeric(df["tbsa_2nd_3rd"], errors="coerce")  # %TBSA
df["baux_score"] = df["age"] + df["tbsa_2nd_3rd"]

print(df[["age", "tbsa_2nd_3rd", "baux_score"]].head())

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
]
# Printing 
print("\n" + "=" * 55)
print("  ENGINEERED FEATURES — summary stats")
print("=" * 55)
for col in engineered_cols:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        print(f"  {col:<35} mean={s.mean():.2f}  missing={s.isna().sum()}")
print("=" * 55)









