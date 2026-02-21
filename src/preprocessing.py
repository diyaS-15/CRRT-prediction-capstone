import pandas as pd
import os

def load_burn_data(file_name="synthetic_data.csv"):
    # dynamic path construction 
    path = os.path.join("data", file_name)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"BCQP data not found at {path}. Check your folder structure!")
    
    return pd.read_csv(path)

df = load_burn_data()

print("COLUMNS:", df.columns.tolist())
# --- Feature engineering: Baux Score = Age + %TBSA ---
# Update these column names if your CSV uses different names
# --- Feature engineering: Baux Score = age + %TBSA ---
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["tbsa_2nd_3rd"] = pd.to_numeric(df["tbsa_2nd_3rd"], errors="coerce")  # %TBSA
df["baux_score"] = df["age"] + df["tbsa_2nd_3rd"]

print(df[["age", "tbsa_2nd_3rd", "baux_score"]].head())