# Train and evaluate a LightGBM model for CRRT prediction
import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# agg = noninteractive background so saved without display

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

from lightgbm import LGBMClassifier
from split import make_patient_level_split, get_Xy
from preprocessing import load_and_preprocess, FEATURE_COLS, TARGET_COL, RANDOM_SEED

# decision threshold (lower=more sensitive to catch more cases but potential more false positives)
# [REEVALUATE AFTER ROC CURVE]
DECISION_THRESHOLD = 0.4
THRESHOLD_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TUNING_N_ITER = 25

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

# Safely calculate ROC-AUC only if both classes are present
def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None

# Safely calculate PR-AUC only if both classes are present
def safe_prauc(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None

def get_metrics(y_true, y_proba, threshold):
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
    preprocessor = build_preprocessor(df)

    # Set up LightGBM model
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # randomized hyperparameter tuning on training split only
    clf = Pipeline([("prep", preprocessor), ("model", model)])
    cv = GroupKFold(n_splits=5)
    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
        "model__max_depth": [3, 4, 5, -1],
        "model__num_leaves": [15, 31, 63],
        "model__min_child_samples": [10, 20, 30],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_alpha": [0, 0.5, 1],
        "model__reg_lambda": [0, 1, 2, 5],
    }
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=TUNING_N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train, groups=train_df[group_col])
    clf = search.best_estimator_
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    best_cv_score = float(search.best_score_)

    # threshold tuning on validation set
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
    threshold_df.to_csv("reports/lightgbm_threshold_tuning.csv", index=False)
    print("Saved: reports/lightgbm_threshold_tuning.csv")

    # Get feature names after preprocessing
    feature_names = clf.named_steps["prep"].get_feature_names_out()

    # Global Feature Importance via SHAP (SHapley Additive exPlanations)
    # note: SHAP used to explain why ML model makes specific prediction, stems in game theory reduces black box

    # transform both splits so SHAP receives the same numeric matrix the model sees initialy
    X_test_transformed  = clf.named_steps["prep"].transform(X_test)
    # background dataset for permutationexplainer
    X_train_transformed = clf.named_steps["prep"].transform(X_train)

    # uses permutationexplainer bc calls predict_proba directly and never reads model internals so changes in model format don't matter too much
    explainer = shap.PermutationExplainer(
        clf.named_steps["model"].predict_proba,
        X_train_transformed,
    )
    # return 2 cols (n_samples, n_features, 2),[:, :, 1] extract shap for CRRT= 1
    shap_values = explainer(X_test_transformed).values[:, :, 1]
    # shap summary plot to reports folder
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig("reports/lightgbm_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("Saved: reports/lightgbm_shap_summary.png")

    # csv of raw shap values for frontend (row=patient, col=feature shap vals)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv("reports/lightgbm_shap_values_test.csv", index=False)
    print("Saved: reports/lightgbm_shap_values_test.csv")

    # Get feature importance scores from LightGBM
    importances = clf.named_steps["model"].feature_importances_

    # Create a table of features and their importance scores
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # predict on X validation + test
    test_proba = clf.predict_proba(X_test)[:, 1]
    # final model evaluation with tuned settings
    val_metrics = get_metrics(y_val, val_proba, best_threshold)
    test_metrics = get_metrics(y_test, test_proba, best_threshold)
    val_pred = val_metrics["pred"]
    test_pred = test_metrics["pred"]

    # Confusion matrix values for validation and test sets
    val_tn, val_fp, val_fn, val_tp = val_metrics["tn"], val_metrics["fp"], val_metrics["fn"], val_metrics["tp"]
    test_tn, test_fp, test_fn, test_tp = test_metrics["tn"], test_metrics["fp"], test_metrics["fn"], test_metrics["tp"]

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
        "best_params": best_params,
        "best_cv_score_average_precision": best_cv_score,
        "scale_pos_weight": float(scale_pos_weight),
        "rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "val_sensitivity": val_metrics["recall"],
        "test_sensitivity": test_metrics["recall"],
        "val_precision": val_metrics["precision"],
        "test_precision": test_metrics["precision"],
        "val_f1": val_metrics["f1"],
        "test_f1": test_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
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
    print("Best threshold:", best_threshold)

    with open("reports/lightgbm_best_params.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_cv_score_average_precision": best_cv_score,
            "best_threshold": best_threshold,
            "threshold_selection_summary": threshold_df.to_dict(orient="records"),
        }, f, indent=2)
    print("Saved: reports/lightgbm_best_params.json")
    # Save evaluation metrics
    with open("reports/lightgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved: reports/lightgbm_metrics.json")
    # Save trained model pipeline
    joblib.dump(clf, "reports/lightgbm_pipeline.joblib")
    print("Saved: reports/lightgbm_pipeline.joblib")

    # Save patient-level predictions
    val_results.to_csv("reports/lightgbm_val_predictions.csv", index=False)
    test_results.to_csv("reports/lightgbm_test_predictions.csv", index=False)
    print("Saved: reports/lightgbm_val_predictions.csv")
    print("Saved: reports/lightgbm_test_predictions.csv")

    # Save split leakage check results
    with open("reports/lightgbm_split_check.json", "w") as f:
        json.dump(leakage_report, f, indent=2)
    print("Saved: reports/lightgbm_split_check.json")

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

    with open("reports/lightgbm_confusion_matrix.json", "w") as f:
        json.dump(confusion_report, f, indent=2)
    print("Saved: reports/lightgbm_confusion_matrix.json")

    # Save false positive and false negative cases
    val_false_positives.to_csv("reports/lightgbm_val_false_positives.csv", index=False)
    val_false_negatives.to_csv("reports/lightgbm_val_false_negatives.csv", index=False)
    test_false_positives.to_csv("reports/lightgbm_test_false_positives.csv", index=False)
    test_false_negatives.to_csv("reports/lightgbm_test_false_negatives.csv", index=False)

    print("Saved: reports/lightgbm_val_false_positives.csv")
    print("Saved: reports/lightgbm_val_false_negatives.csv")
    print("Saved: reports/lightgbm_test_false_positives.csv")
    print("Saved: reports/lightgbm_test_false_negatives.csv")

    # Save feature importance results
    feature_importance_df.to_csv("reports/lightgbm_feature_importance.csv", index=False)
    print("Saved: reports/lightgbm_feature_importance.csv")

    # Print top 10 most important features
    print("Top 10 features:")
    print(feature_importance_df.head(10))

if __name__ == "__main__":
    main()
