import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler



df = pd.read_csv(os.path.join("data", "synthetic_data.csv"))
df["label"] = (df["age"] > df["age"].median()).astype(int)

X = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")
y = df["label"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = HistGradientBoostingClassifier()
model.fit(X_scaled, y)

print("trained")
