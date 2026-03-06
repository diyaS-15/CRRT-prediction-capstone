import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from clean_missing import clean_missing_values
clean_missing_values("synthetic_data.csv", "cleaned_data.csv")

df = pd.read_csv(os.path.join("data", "cleaned_data.csv"))
df["label"] = (df["age"] > df["age"].median()).astype(int)

X = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")
y = df["label"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = HistGradientBoostingClassifier()
model.fit(X_scaled, y)

# --- Top 10 highest-risk predictions (simple: on same data used for training) ---
import numpy as np

# Make sure record_id exists
if "record_id" in df.columns:
    record_ids = df["record_id"].values
else:
    record_ids = np.arange(len(df))  # fallback if no record_id column

# Probability of class 1 (risk)
risk_scores = model.predict_proba(X_scaled)[:, 1]

top_risk = pd.DataFrame({
    "record_id": record_ids,
    "risk_score": risk_scores,
    "label": y.values
}).sort_values("risk_score", ascending=False).head(10)

print("\nTop 10 highest-risk predictions:")
print(top_risk.to_string(index=False))

os.makedirs("results", exist_ok=True)
top_risk.to_csv("results/top10_high_risk_predictions.csv", index=False)
print("\nSaved: results/top10_high_risk_predictions.csv")

print("trained")
