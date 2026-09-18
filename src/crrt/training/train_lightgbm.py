# Train and evaluate a LightGBM model for CRRT prediction
#
# Every run is logged to MLflow and registered as a new (unpromoted) version
# of the "crrt-lightgbm" model. Hyperparameter search uses Optuna (TPE
# sampler) instead of RandomizedSearchCV, same rationale as train_xgb.py.
import os
import json
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# agg = noninteractive background so saved without display

import mlflow
import mlflow.sklearn
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GroupKFold

from lightgbm import LGBMClassifier
from src.crrt.data.split import make_patient_level_split, get_Xy
from src.crrt.features.preprocessing import load_and_preprocess, TARGET_COL, RANDOM_SEED
from src.crrt.training.common import build_preprocessor, verify_no_patient_leakage, get_metrics
from src.crrt.training.mlflow_utils import init_mlflow

optuna.logging.set_verbosity(optuna.logging.WARNING)

MLFLOW_MODEL_NAME = "crrt-lightgbm"

# decision threshold (lower=more sensitive to catch more cases but potential more false positives)
# [REEVALUATE AFTER ROC CURVE]
DECISION_THRESHOLD = 0.4
THRESHOLD_CANDIDATES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TUNING_N_TRIALS = 25


def make_objective(preprocessor, X_train, y_train, groups, scale_pos_weight):
    cv = GroupKFold(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, -1]),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(
            pipe, X_train, y_train, groups=groups, cv=cv,
            scoring="average_precision", n_jobs=-1,
        )
        return float(scores.mean())

    return objective


def main():
    os.makedirs("reports", exist_ok=True)
    init_mlflow()
    with mlflow.start_run(run_name="lightgbm"):
        _main()


def _main():
    artifact_paths = []
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

    mlflow.log_params({"data_path": os.getenv("BCQP_DATA_PATH", "data/synthetic_data.csv"), "label_col": label_col, "group_col": group_col})

    X_train, X_val, X_test, y_train, y_val, y_test = get_Xy(df, splits)
    # scaling positive bc minority class so it's not ignored
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.2f}  (train positives: {(y_train==1).sum()}, negatives: {(y_train==0).sum()})")

    # Create preprocessing pipeline using training data columns
    preprocessor = build_preprocessor(df)

    # Bayesian hyperparameter search on training split only (GroupKFold CV)
    objective = make_objective(preprocessor, X_train, y_train, train_df[group_col], scale_pos_weight)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=TUNING_N_TRIALS)

    best_params = study.best_params
    best_cv_score = float(study.best_value)
    print("Best params:", best_params)
    print("Best CV average precision:", best_cv_score)

    study.trials_dataframe().to_csv("reports/lightgbm_optuna_trials.csv", index=False)
    artifact_paths.append("reports/lightgbm_optuna_trials.csv")
    print("Saved: reports/lightgbm_optuna_trials.csv")

    best_model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
        **best_params,
    )
    clf = Pipeline([("prep", preprocessor), ("model", best_model)])
    clf.fit(X_train, y_train)

    mlflow.log_params({f"model__{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_average_precision", best_cv_score)

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
    artifact_paths.append("reports/lightgbm_threshold_tuning.csv")
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
    artifact_paths.append("reports/lightgbm_shap_summary.png")
    print("Saved: reports/lightgbm_shap_summary.png")

    # csv of raw shap values for frontend (row=patient, col=feature shap vals)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv("reports/lightgbm_shap_values_test.csv", index=False)
    artifact_paths.append("reports/lightgbm_shap_values_test.csv")
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

    mlflow.log_param("decision_threshold", best_threshold)
    mlflow.log_metrics({
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    })

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
    artifact_paths.append("reports/lightgbm_best_params.json")
    print("Saved: reports/lightgbm_best_params.json")
    # Save evaluation metrics
    with open("reports/lightgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    artifact_paths.append("reports/lightgbm_metrics.json")
    print("Saved: reports/lightgbm_metrics.json")
    # Save trained model pipeline (kept alongside the MLflow-registered copy)
    joblib.dump(clf, "reports/lightgbm_pipeline.joblib")
    print("Saved: reports/lightgbm_pipeline.joblib")

    # Save patient-level predictions
    val_results.to_csv("reports/lightgbm_val_predictions.csv", index=False)
    test_results.to_csv("reports/lightgbm_test_predictions.csv", index=False)
    artifact_paths.extend(["reports/lightgbm_val_predictions.csv", "reports/lightgbm_test_predictions.csv"])
    print("Saved: reports/lightgbm_val_predictions.csv")
    print("Saved: reports/lightgbm_test_predictions.csv")

    # Save split leakage check results
    with open("reports/lightgbm_split_check.json", "w") as f:
        json.dump(leakage_report, f, indent=2)
    artifact_paths.append("reports/lightgbm_split_check.json")
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
    artifact_paths.append("reports/lightgbm_confusion_matrix.json")
    print("Saved: reports/lightgbm_confusion_matrix.json")

    # Save false positive and false negative cases
    val_false_positives.to_csv("reports/lightgbm_val_false_positives.csv", index=False)
    val_false_negatives.to_csv("reports/lightgbm_val_false_negatives.csv", index=False)
    test_false_positives.to_csv("reports/lightgbm_test_false_positives.csv", index=False)
    test_false_negatives.to_csv("reports/lightgbm_test_false_negatives.csv", index=False)
    artifact_paths.extend([
        "reports/lightgbm_val_false_positives.csv",
        "reports/lightgbm_val_false_negatives.csv",
        "reports/lightgbm_test_false_positives.csv",
        "reports/lightgbm_test_false_negatives.csv",
    ])

    print("Saved: reports/lightgbm_val_false_positives.csv")
    print("Saved: reports/lightgbm_val_false_negatives.csv")
    print("Saved: reports/lightgbm_test_false_positives.csv")
    print("Saved: reports/lightgbm_test_false_negatives.csv")

    # Save feature importance results
    feature_importance_df.to_csv("reports/lightgbm_feature_importance.csv", index=False)
    artifact_paths.append("reports/lightgbm_feature_importance.csv")
    print("Saved: reports/lightgbm_feature_importance.csv")

    # Print top 10 most important features
    print("Top 10 features:")
    print(feature_importance_df.head(10))

    for path in artifact_paths:
        mlflow.log_artifact(path)

    mlflow.sklearn.log_model(
        clf, name="model", registered_model_name=MLFLOW_MODEL_NAME,
        serialization_format="cloudpickle",
    )
    print(f"Registered model version under '{MLFLOW_MODEL_NAME}' (stage=None).")

if __name__ == "__main__":
    main()
