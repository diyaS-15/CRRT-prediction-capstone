import pandas as pd
import os

def load_burn_data(file_name="synthetic_data.csv"):
    # dynamic path construction 
    path = os.path.join("data", file_name)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"BCQP data not found at {path}. Check your folder structure!")
    
    return pd.read_csv(path)

df = load_burn_data()
print(df.head())