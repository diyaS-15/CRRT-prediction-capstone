import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv(os.path.join("data", "synthetic_data.csv"))
df["label"] = (df["age"] > df["age"].median()).astype(int)

X = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")
y = df["label"]

model = HistGradientBoostingClassifier()
model.fit(X, y)

print("trained")
