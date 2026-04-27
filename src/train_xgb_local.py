# Train and evaluate an XGBoost model for CRRT prediction

import os
import json
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
from sklearn.model_selection import ParameterGrid, GroupKFold

from xgboost import XGBClassifier
from .split import make_patient_level_split
from .preprocessing import load_and_preprocess, RANDOM_SEED

MODEL_TARGET_COL = "crrt_25_48h"

MODEL_FEATURE_COLS = [
    "age",
    "sex",
    "tbsa_2nd_3rd",
    "admission_weight_kg",
    "initial_gcs",
    "carboxyhemoglobin",
    "initial_temp_c",
    "total_crystalloid_ml_first_24h",
    "total_colloid_ml_first_24h",
    "total_blood_products_units_first_24h",
    "total_urine_output_ml_first_24h",
    "inhalation_flag",
    "revised_baux_score",
    "hours_injury_to_admission",
    "late_admission_flag",
    "fluid_balance_24h",
    "fluid_overload_flag",
    "burn_severity_tier",
    "carboxyhemoglobin_risk_flag",
    "hypothermia_flag",
    "comorbidity_aki_risk_score",
]

THRESHOLD_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70]
CV_N_SPLITS = 5

# keep this smaller first so runtime stays manageable on 200 patients
TUNING_PARAM_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [100, 200, 300],
    "min_child_weight": [1, 3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 1],
    "reg_alpha": [0, 0.5],
}

def build_preprocessor(df: pd.DataFrame):
    available = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in available if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

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

def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None

def safe_prauc(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None

def get_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, proba),
        "pr_auc": safe_prauc(y_true, proba),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def mean_or_none(values):
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None

def main():
    data_path = os.getenv("BCQP_DATA_PATH", "data/synthetic_data.csv")
    print("Using data path =", data_path)
    df = load_and_preprocess(data_path)

    label_col = MODEL_TARGET_COL
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    group_col = "patient_id" if "patient_id" in df.columns else "record_id"
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")

    df = df[df[label_col].notna()].copy()

    approved_features = [c for c in MODEL_FEATURE_COLS if c in df.columns]
    missing_features = [c for c in MODEL_FEATURE_COLS if c not in df.columns]

    if len(approved_features) == 0:
        raise ValueError("No approved 24-hour predictors found in dataframe.")

    print("Target column:", label_col)
    print("Approved 24-hour predictors:", approved_features)

    if missing_features:
        print("Missing selected predictors:", missing_features)

    blocked_cols = [
        c for c in df.columns
        if c in {
            "crrt_first_24h",
            "crrt_within_48h",
            "urine_output_per_kg",
            "low_urine_output_flag",
        } or c.endswith("_25_48h")
    ]

    print("Blocked leakage/unreliable columns not used:", blocked_cols)

    splits = make_patient_level_split(
        df,
        group_col=group_col,
        val_size=0.10,
        test_size=0.20,
        seed=RANDOM_SEED,
    )

    train_df = df.iloc[splits.train_idx]
    val_df = df.iloc[splits.val_idx]
    test_df = df.iloc[splits.test_idx]

    leakage_report = verify_no_patient_leakage(train_df, val_df, test_df, group_col)
    print("Leakage check:", leakage_report)
    if leakage_report["leakage_found"]:
        raise RuntimeError("patient data leaked")

    dev_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    X_dev = dev_df[approved_features].copy()
    X_test = test_df[approved_features].copy()

    groups_dev = dev_df[group_col].copy()

    label_map = {
        "No": 0, "Yes": 1,
        "no": 0, "yes": 1,
        False: 0, True: 1,
        0: 0, 1: 1
    }

    y_dev = dev_df[label_col].map(label_map).astype(int)
    y_test = test_df[label_col].map(label_map).astype(int)

    pos_count = (y_dev == 1).sum()
    neg_count = (y_dev == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight:.2f} (dev positives: {pos_count}, negatives: {neg_count})")

    preprocessor = build_preprocessor(X_dev)
    cv = GroupKFold(n_splits=CV_N_SPLITS)

    os.makedirs("reports", exist_ok=True)

    tuning_rows = []
    fold_rows = []
    param_list = list(ParameterGrid(TUNING_PARAM_GRID))
    print(f"Total parameter combinations: {len(param_list)}")

    for i, params in enumerate(param_list, start=1):
        print(f"Tuning {i}/{len(param_list)}: {params}")

        threshold_metrics = {t: [] for t in THRESHOLD_CANDIDATES}

        for fold_num, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev, groups_dev), start=1):
            X_fold_train = X_dev.iloc[train_idx]
            X_fold_val = X_dev.iloc[val_idx]
            y_fold_train = y_dev.iloc[train_idx]
            y_fold_val = y_dev.iloc[val_idx]

            model = XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                reg_lambda=1.0,
                max_delta_step=2,
                **params,
            )

            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_fold_train, y_fold_train)
            fold_val_proba = pipe.predict_proba(X_fold_val)[:, 1]

            for threshold in THRESHOLD_CANDIDATES:
                fold_metrics = get_metrics(y_fold_val, fold_val_proba, threshold)

                threshold_metrics[threshold].append(fold_metrics)

                fold_rows.append({
                    **params,
                    "fold": fold_num,
                    "threshold": threshold,
                    "cv_accuracy": fold_metrics["accuracy"],
                    "cv_recall": fold_metrics["recall"],
                    "cv_precision": fold_metrics["precision"],
                    "cv_f1": fold_metrics["f1"],
                    "cv_roc_auc": fold_metrics["roc_auc"],
                    "cv_pr_auc": fold_metrics["pr_auc"],
                    "cv_tp": fold_metrics["tp"],
                    "cv_fp": fold_metrics["fp"],
                    "cv_tn": fold_metrics["tn"],
                    "cv_fn": fold_metrics["fn"],
                })

        for threshold in THRESHOLD_CANDIDATES:
            rows = threshold_metrics[threshold]

            summary_row = {
                **params,
                "threshold": threshold,
                "cv_accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
                "cv_recall_mean": float(np.mean([r["recall"] for r in rows])),
                "cv_precision_mean": float(np.mean([r["precision"] for r in rows])),
                "cv_f1_mean": float(np.mean([r["f1"] for r in rows])),
                "cv_roc_auc_mean": mean_or_none([r["roc_auc"] for r in rows]),
                "cv_pr_auc_mean": mean_or_none([r["pr_auc"] for r in rows]),
                "cv_tp_mean": float(np.mean([r["tp"] for r in rows])),
                "cv_fp_mean": float(np.mean([r["fp"] for r in rows])),
                "cv_tn_mean": float(np.mean([r["tn"] for r in rows])),
                "cv_fn_mean": float(np.mean([r["fn"] for r in rows])),
                "cv_accuracy_std": float(np.std([r["accuracy"] for r in rows])),
                "cv_recall_std": float(np.std([r["recall"] for r in rows])),
                "cv_precision_std": float(np.std([r["precision"] for r in rows])),
                "cv_f1_std": float(np.std([r["f1"] for r in rows])),
            }

            tuning_rows.append(summary_row)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv("reports/cv_fold_metrics.csv", index=False)

    tuning_df = pd.DataFrame(tuning_rows)

    tuning_df = tuning_df.sort_values(
        by=[
            "cv_fn_mean",
            "cv_recall_mean",
            "cv_precision_mean",
            "cv_f1_mean",
            "cv_pr_auc_mean",
            "cv_fp_mean",
            "cv_accuracy_mean",
            "cv_recall_std",
        ],
        ascending=[True, False, False, False, False, True, False, True]
    ).reset_index(drop=True)

    tuning_df.insert(0, "rank", np.arange(1, len(tuning_df) + 1))

    tuning_df.to_csv("reports/cv_tuning_summary.csv", index=False)
    tuning_df.head(10).to_csv("reports/cv_top_10_configs.csv", index=False)

    worst_df = tuning_df.sort_values(
        by=[
            "cv_fn_mean",
            "cv_recall_mean",
            "cv_precision_mean",
            "cv_f1_mean",
            "cv_pr_auc_mean",
            "cv_fp_mean",
            "cv_accuracy_mean",
        ],
        ascending=[False, True, True, True, True, False, True]
    ).reset_index(drop=True)

    worst_df.head(10).to_csv("reports/cv_bottom_10_configs.csv", index=False)

    best_row = tuning_df.iloc[0].to_dict()
    best_params = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "n_estimators": int(best_row["n_estimators"]),
        "min_child_weight": int(best_row["min_child_weight"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "gamma": float(best_row["gamma"]),
        "reg_alpha": float(best_row["reg_alpha"]),
    }
    best_threshold = float(best_row["threshold"])

    print("\nBest params:", best_params)
    print("Best threshold:", best_threshold)

    best_model = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        max_delta_step=2,
        **best_params,
    )

    best_pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", best_model)
    ])

    best_pipeline.fit(X_dev, y_dev)
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = get_metrics(y_test, test_proba, best_threshold)

    best_params_payload = {
        "target_column": label_col,
        "group_column": group_col,
        "selected_features": approved_features,
        "missing_features": missing_features,
        "cv_folds": CV_N_SPLITS,
        "ranking_rule": [
            "lowest cv_fn_mean",
            "highest cv_recall_mean",
            "highest cv_precision_mean",
            "highest cv_f1_mean",
            "highest cv_pr_auc_mean",
            "lowest cv_fp_mean",
            "highest cv_accuracy_mean",
            "lowest cv_recall_std"
        ],
        "best_params": best_params,
        "best_threshold": best_threshold,
        "best_cv_metrics": {
            "cv_accuracy_mean": best_row["cv_accuracy_mean"],
            "cv_recall_mean": best_row["cv_recall_mean"],
            "cv_precision_mean": best_row["cv_precision_mean"],
            "cv_f1_mean": best_row["cv_f1_mean"],
            "cv_roc_auc_mean": best_row["cv_roc_auc_mean"],
            "cv_pr_auc_mean": best_row["cv_pr_auc_mean"],
            "cv_tp_mean": best_row["cv_tp_mean"],
            "cv_fp_mean": best_row["cv_fp_mean"],
            "cv_tn_mean": best_row["cv_tn_mean"],
            "cv_fn_mean": best_row["cv_fn_mean"],
            "cv_accuracy_std": best_row["cv_accuracy_std"],
            "cv_recall_std": best_row["cv_recall_std"],
            "cv_precision_std": best_row["cv_precision_std"],
            "cv_f1_std": best_row["cv_f1_std"],
        },
    }

    with open("reports/best_params.json", "w") as f:
        json.dump(best_params_payload, f, indent=2)

    final_metrics = {
        "cv_accuracy_mean": best_row["cv_accuracy_mean"],
        "cv_recall_mean": best_row["cv_recall_mean"],
        "cv_precision_mean": best_row["cv_precision_mean"],
        "cv_f1_mean": best_row["cv_f1_mean"],
        "cv_roc_auc_mean": best_row["cv_roc_auc_mean"],
        "cv_pr_auc_mean": best_row["cv_pr_auc_mean"],
        "cv_tp_mean": best_row["cv_tp_mean"],
        "cv_fp_mean": best_row["cv_fp_mean"],
        "cv_tn_mean": best_row["cv_tn_mean"],
        "cv_fn_mean": best_row["cv_fn_mean"],
        "cv_accuracy_std": best_row["cv_accuracy_std"],
        "cv_recall_std": best_row["cv_recall_std"],
        "cv_precision_std": best_row["cv_precision_std"],
        "cv_f1_std": best_row["cv_f1_std"],

        "test_accuracy": test_metrics["accuracy"],
        "test_recall": test_metrics["recall"],
        "test_precision": test_metrics["precision"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_tp": test_metrics["tp"],
        "test_fp": test_metrics["fp"],
        "test_tn": test_metrics["tn"],
        "test_fn": test_metrics["fn"],
    }

    with open("reports/final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\nFINAL METRICS")
    print(final_metrics)
    print("Saved: reports/cv_fold_metrics.csv")
    print("Saved: reports/cv_tuning_summary.csv")
    print("Saved: reports/cv_top_10_configs.csv")
    print("Saved: reports/cv_bottom_10_configs.csv")
    print("Saved: reports/best_params.json")
    print("Saved: reports/final_metrics.json")

if __name__ == "__main__":
    main()