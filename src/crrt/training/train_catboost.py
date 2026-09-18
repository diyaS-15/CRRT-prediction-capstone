# Train and evaluate an CatBoost algorithm model for CRRT prediction
import os
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

from catboost import CatBoostClassifier
from ..data.split import make_patient_level_split, get_Xy
from ..features.preprocessing import load_and_preprocess, TARGET_COL, RANDOM_SEED
from .common import build_preprocessor, verify_no_patient_leakage, safe_auc, safe_prauc, get_metrics

# decision threshold (lower=more sensitive to catch more cases but potential more false positives)
# tuned per-run on the validation set (see THRESHOLD_CANDIDATES) rather than fixed,
# since missing a true CRRT case (false negative) is clinically costlier than a false alarm.
THRESHOLD_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

def main():
   # Load dataset from local file path
    data_path = os.getenv("BCQP_DATA_PATH", "data/synthetic_data.csv")
    df = load_and_preprocess(data_path)

    # Make sure the label and group columns exist
    label_col = TARGET_COL
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    group_col = "patient_id" if "patient_id" in df.columns else "record_id"
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")

    # Drop rows with missing target before splitting so SplitResult indices
    # align with the df we slice into train/val/test below.
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Split data by patient/group to avoid leakage
    splits = make_patient_level_split(df, group_col=group_col, val_size=0.10, test_size=0.20, seed=RANDOM_SEED)

    train_df = df.iloc[splits.train_idx]
    val_df   = df.iloc[splits.val_idx]
    test_df  = df.iloc[splits.test_idx]

    # Verify there is no overlap of patients across splits
    leakage_report = verify_no_patient_leakage(train_df, val_df, test_df, group_col)
    print("Leakage check:", leakage_report)
    # error to stop training if there's a data leak 
    if leakage_report["leakage_found"]:
        raise RuntimeError("patient data leaked") 
    
    X_train, X_val, X_test, y_train, y_val, y_test = get_Xy(df, splits)
    # scaling positive bc minority class so it's not ignored 
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"scale_pos_weight: {scale_pos_weight:.2f}  (train positives: {(y_train==1).sum()}, negatives: {(y_train==0).sum()})")

    # Create preprocessing pipeline using training data columns
    preprocessor = build_preprocessor(X_train)

    # Set up CatBoost model; tell it to care more about the positive (CRRT=1)
    # class since CRRT=1 is rare, so give it higher weight
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        class_weights=[1.0, float(scale_pos_weight)],  # [weight for class 0, weight for class 1]
    )

    clf = Pipeline([("prep", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    # threshold tuning on validation set, ranked to minimize false negatives first
    # (a missed CRRT case is clinically costlier than a false alarm)
    val_proba = clf.predict_proba(X_val)[:, 1]
    threshold_rows = []
    for threshold in THRESHOLD_CANDIDATES:
        m = get_metrics(y_val, val_proba, threshold)
        threshold_rows.append({
            "threshold": threshold,
            "fn": m["fn"],
            "recall": m["recall"],
            "precision": m["precision"],
            "f1": m["f1"],
            "pr_auc": m["pr_auc"],
            "fp": m["fp"],
            "accuracy": m["accuracy"],
        })
    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        by=["fn", "recall", "precision", "f1", "pr_auc", "fp", "accuracy"],
        ascending=[True, False, False, False, False, True, False],
    )
    best_threshold = float(threshold_df.iloc[0]["threshold"])
    os.makedirs("reports", exist_ok=True)
    threshold_df.to_csv("reports/cat_threshold_tuning.csv", index=False)
    print("Saved: reports/cat_threshold_tuning.csv")
    print("Best threshold:", best_threshold)

    # Get feature names after preprocessing
    feature_names = clf.named_steps["prep"].get_feature_names_out()

    # Get feature importance scores from XGBoost
    importances = clf.named_steps["model"].feature_importances_

    # Create a table of features and their importance scores
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # predict on X test (val_proba already computed above during threshold tuning)
    test_proba = clf.predict_proba(X_test)[:, 1]
    # prediction threshold tuned above on the validation set
    val_pred  = (val_proba  >= best_threshold).astype(int)
    test_pred = (test_proba >= best_threshold).astype(int)

    # Confusion matrix values for validation and test sets
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_pred).ravel()
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_pred).ravel()

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

    # Store model evaluation results
    metrics = {
        "label_col": label_col,
        "group_col": group_col,
        "decision_threshold": best_threshold,
        "scale_pos_weight": float(scale_pos_weight),
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
    with open("reports/cat_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: reports/cat_metrics.json")
    # Save trained model pipeline
    joblib.dump(clf, "reports/cat_pipeline.joblib")
    print("Saved: reports/cat_pipeline.joblib")

    # Save patient-level predictions
    val_results.to_csv("reports/cat-val_predictions.csv", index=False)
    test_results.to_csv("reports/cat-test_predictions.csv", index=False)
    print("Saved: reports/cat-val_predictions.csv")
    print("Saved: reports/cat-test_predictions.csv")

    # Save split leakage check results
    with open("reports/cat-split_check.json", "w") as f:
        json.dump(leakage_report, f, indent=2)
    print("Saved: reports/cat-split_check.json")

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

    with open("reports/cat-confusion_matrix.json", "w") as f:
        json.dump(confusion_report, f, indent=2)
    print("Saved: reports/cat-confusion_matrix.json")

    # Save false positive and false negative cases
    val_false_positives.to_csv("reports/cat-val_false_positives.csv", index=False)
    val_false_negatives.to_csv("reports/cat-val_false_negatives.csv", index=False)
    test_false_positives.to_csv("reports/cat-test_false_positives.csv", index=False)
    test_false_negatives.to_csv("reports/cat-test_false_negatives.csv", index=False)
    
    print("Saved: reports/cat-val_false_positives.csv")
    print("Saved: reports/cat-val_false_negatives.csv")
    print("Saved: reports/cat-test_false_positives.csv")
    print("Saved: reports/cat-test_false_negatives.csv")

    # Save feature importance results
    feature_importance_df.to_csv("reports/cat_feature_importance.csv", index=False)
    print("Saved: reports/cat_feature_importance.csv")

    # Print top 10 most important features
    print("Top 10 features:")
    print(feature_importance_df.head(10))

if __name__ == "__main__":
    main()