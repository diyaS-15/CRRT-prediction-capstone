"""Shared helpers for the training scripts (train_xgb.py, train_catboost.py,
train_lightgbm.py). These were previously copy-pasted identically across all
three files; extracted here so a bug fix or behavior change only needs to
happen in one place.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
)

from ..features.preprocessing import FEATURE_COLS


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing steps for numeric and categorical features."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    numeric_cols = df[available].select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in available if c not in numeric_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def verify_no_patient_leakage(train_df, val_df, test_df, group_col: str) -> dict:
    """Check that no patient appears in more than one split."""
    train_ids = set(train_df[group_col])
    val_ids = set(val_df[group_col])
    test_ids = set(test_df[group_col])

    leakage_report = {
        "group_col": group_col,
        "train_unique_ids": len(train_ids),
        "val_unique_ids": len(val_ids),
        "test_unique_ids": len(test_ids),
        "train_val_overlap": len(train_ids & val_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "val_test_overlap": len(val_ids & test_ids),
    }

    leakage_report["leakage_found"] = (
        leakage_report["train_val_overlap"] > 0 or
        leakage_report["train_test_overlap"] > 0 or
        leakage_report["val_test_overlap"] > 0
    )

    return leakage_report


def safe_auc(y, p):
    """Safely calculate ROC-AUC only if both classes are present."""
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def safe_prauc(y, p):
    """Safely calculate PR-AUC only if both classes are present."""
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None


def get_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_proba),
        "pr_auc": safe_prauc(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred": y_pred,
    }
