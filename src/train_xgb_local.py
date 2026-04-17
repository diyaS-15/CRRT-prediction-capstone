# Train and evaluate an XGBoost model for CRRT prediction

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, recall_score, precision_score, f1_score

from xgboost import XGBClassifier
from .split import make_patient_level_split
from .preprocessing import load_and_preprocess, FEATURE_COLS, TARGET_COL, RANDOM__SEED

# decision threshold (lower=more sensitive to catch more cases but potential more false positives)
# [REEVALUATE AFTER ROC CURVE]
DECISION_THRESHOLD = 0.75

# Choose which label column to predict
def pick_label(df: pd.DataFrame) -> str:
    # Read label name from environment variable
    label = os.getenv("LABEL_COL", "crrt_25_48h").strip()
    
    if label in df.columns:
        return label

    if "crrt_25_48h" in df.columns:
        return "crrt_25_48h"

    raise ValueError("No valid CRRT label column found in dataframe.")
    

# Build preprocessing steps for numeric and categorical features
def build_preprocessor(df: pd.DataFrame):
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

# Check that no patient appears in more than one split
def verify_no_patient_leakage(train_df, val_df, test_df, group_col):
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

def main():
   # Load dataset from local file path
    data_path = "/Users/anagantikrutika/Desktop/CSE485/synthetic_data.csv"
    print("Using data path =", data_path)
    df = load_and_preprocess(data_path)

    # Make sure the label and group columns exist
    label_col = TARGET_COL if TARGET_COL in df.columns else pick_label(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    group_col = "patient_id" if "patient_id" in df.columns else "record_id"
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")

    # Drop rows with missing label before splitting
    df = df[df[label_col].notna()].copy()

    # Final approved 24-hour predictors only
    approved_features = [c for c in FEATURE_COLS if c in df.columns and c != label_col]
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]

    if len(approved_features) == 0:
        raise ValueError("No approved 24-hour predictors found in dataframe.")

    print("Approved 24-hour predictors:", approved_features)
    if missing_features:
        print("Missing approved predictors:", missing_features)

    # Split data by patient/group to avoid leakage
    splits = make_patient_level_split(df, group_col=group_col, val_size=0.10, test_size=0.20, seed=RANDOM__SEED)

    train_df = df.iloc[splits.train_idx]
    val_df   = df.iloc[splits.val_idx]
    test_df  = df.iloc[splits.test_idx]

    # Verify there is no overlap of patients across splits
    leakage_report = verify_no_patient_leakage(train_df, val_df, test_df, group_col)
    print("Leakage check:", leakage_report)
    # error to stop training if there's a data leak 
    if leakage_report["leakage_found"]:
        raise RuntimeError("patient data leaked") 
    
    # Use only approved 24-hour predictors
    X_train = train_df[approved_features].copy()
    X_val   = val_df[approved_features].copy()
    X_test  = test_df[approved_features].copy()

    # Convert labels to 0/1 format
    # 0 = no and 1 = yes
    label_map = {
    "No": 0, "Yes": 1,
    "no": 0, "yes": 1,
    False: 0, True: 1,
    0: 0, 1: 1
    }

    y_train = train_df[label_col].map(label_map).astype(int)
    y_val   = val_df[label_col].map(label_map).astype(int)
    y_test  = test_df[label_col].map(label_map).astype(int)

    # scaling positive bc minority class so it's not ignored 
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.2f}  (train positives: {pos_count}, negatives: {neg_count})")

    # Create preprocessing pipeline using training data columns
    preprocessor = build_preprocessor(X_train)

    # Set up XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM__SEED,
        n_jobs=-1,
    )

    clf = Pipeline([("prep", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    # Get feature names after preprocessing
    feature_names = clf.named_steps["prep"].get_feature_names_out()

    # Get feature importance scores from XGBoost
    importances = clf.named_steps["model"].feature_importances_

    # Create a table of features and their importance scores
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # predict on X validation + test 
    val_proba  = clf.predict_proba(X_val)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]
    # prediction threshold adjuatable w/ deicsion threshold now 
    val_pred  = (val_proba  >= DECISION_THRESHOLD).astype(int)
    test_pred = (test_proba >= DECISION_THRESHOLD).astype(int)

    # Confusion matrix values for validation and test sets
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    print("VAL confusion matrix:")
    print("TP:", val_tp, "FP:", val_fp, "TN:", val_tn, "FN:", val_fn)

    print("TEST confusion matrix:")
    print("TP:", test_tp, "FP:", test_fp, "TN:", test_tn, "FN:", test_fn)

    val_results = pd.DataFrame({
        group_col: val_df[group_col].values,
        "actual": y_val.values,
        "pred_proba": val_proba,
        "pred_label": val_pred
    })

    test_results = pd.DataFrame({
        group_col: test_df[group_col].values,
        "actual": y_test.values,
        "pred_proba": test_proba,
        "pred_label": test_pred
    })

    # Export false positives and false negatives for VALIDATION set
    val_false_positives = val_results[
        (val_results["actual"] == 0) & (val_results["pred_label"] == 1)
    ]

    val_false_negatives = val_results[
        (val_results["actual"] == 1) & (val_results["pred_label"] == 0)
    ]

    # Export false positives and false negatives for TEST set
    test_false_positives = test_results[
        (test_results["actual"] == 0) & (test_results["pred_label"] == 1)
    ]

    test_false_negatives = test_results[
        (test_results["actual"] == 1) & (test_results["pred_label"] == 0)
    ]

    # Safely calculate ROC-AUC only if both classes are present
    def safe_auc(y, p):
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None

    # Safely calculate PR-AUC only if both classes are present
    def safe_prauc(y, p):
        return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None

    # Store model evaluation results
    metrics = {
        "label_col": label_col,
        "group_col": group_col,
        "decision_threshold": DECISION_THRESHOLD,
        "scale_pos_weight": float(scale_pos_weight),
        "approved_24h_predictors": approved_features,
        "missing_approved_predictors": missing_features,
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "val_sensitivity": float(recall_score(y_val,  val_pred,  zero_division=0)),
        "test_sensitivity":float(recall_score(y_test, test_pred, zero_division=0)),
        "val_precision": float(precision_score(y_val,  val_pred,  zero_division=0)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "val_f1": float(f1_score(y_val,  val_pred,  zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "val_roc_auc": safe_auc(y_val, val_proba),
        "test_roc_auc": safe_auc(y_test, test_proba),
        "val_pr_auc": safe_prauc(y_val, val_proba),
        "test_pr_auc": safe_prauc(y_test, test_proba),
        "val_tp": int(val_tp),
        "val_fp": int(val_fp),
        "val_tn": int(val_tn),
        "val_fn": int(val_fn),
        "test_tp": int(test_tp),
        "test_fp": int(test_fp),
        "test_tn": int(test_tn),
        "test_fn": int(test_fn),
    }

    # Print summary of model results
    print("Label:", label_col, "| Group:", group_col)
    print("Train/Val/Test rows:", metrics["train_rows"], metrics["val_rows"], metrics["test_rows"])
    print("VAL  acc/ROC-AUC/PR-AUC:", metrics["val_accuracy"], metrics["val_roc_auc"], metrics["val_pr_auc"])
    print("TEST acc/ROC-AUC/PR-AUC:", metrics["test_accuracy"], metrics["test_roc_auc"], metrics["test_pr_auc"])

    # Create reports folder if it does not exist
    os.makedirs("reports", exist_ok=True)
    # Save evaluation metrics
    with open("reports/xgb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: reports/xgb_metrics.json")

    # Save approved 24-hour predictors
    with open("reports/approved_24h_predictors.json", "w") as f:
        json.dump({
            "approved_24h_predictors": approved_features,
            "missing_approved_predictors": missing_features
        }, f, indent=2)
    print("Saved: reports/approved_24h_predictors.json")

    # Save trained model pipeline
    joblib.dump(clf, "reports/xgb_pipeline.joblib")
    print("Saved: reports/xgb_pipeline.joblib")

    # Save patient-level predictions
    val_results.to_csv("reports/val_predictions.csv", index=False)
    test_results.to_csv("reports/test_predictions.csv", index=False)
    print("Saved: reports/val_predictions.csv")
    print("Saved: reports/test_predictions.csv")

    # Save split leakage check results
    with open("reports/split_check.json", "w") as f:
        json.dump(leakage_report, f, indent=2)
    print("Saved: reports/split_check.json")

    # Save confusion matrix results
    confusion_report = {
        "validation": {
            "tp": int(val_tp),
            "fp": int(val_fp),
            "tn": int(val_tn),
            "fn": int(val_fn),
        },
        "test": {
            "tp": int(test_tp),
            "fp": int(test_fp),
            "tn": int(test_tn),
            "fn": int(test_fn),
        }
    }

    with open("reports/confusion_matrix.json", "w") as f:
        json.dump(confusion_report, f, indent=2)
    print("Saved: reports/confusion_matrix.json")

    # Save false positive and false negative cases
    val_false_positives.to_csv("reports/val_false_positives.csv", index=False)
    val_false_negatives.to_csv("reports/val_false_negatives.csv", index=False)
    test_false_positives.to_csv("reports/test_false_positives.csv", index=False)
    test_false_negatives.to_csv("reports/test_false_negatives.csv", index=False)
    
    print("Saved: reports/val_false_positives.csv")
    print("Saved: reports/val_false_negatives.csv")
    print("Saved: reports/test_false_positives.csv")
    print("Saved: reports/test_false_negatives.csv")

    # Save feature importance results
    feature_importance_df.to_csv("reports/xgb_feature_importance.csv", index=False)
    print("Saved: reports/xgb_feature_importance.csv")

    # Print top 10 most important features
    print("Top 10 features:")
    print(feature_importance_df.head(10))

if __name__ == "__main__":
    main()