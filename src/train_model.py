import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

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

print("trained")

y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

print(accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred),f1_score(y, y_pred), roc_auc_score(y, y_proba))

