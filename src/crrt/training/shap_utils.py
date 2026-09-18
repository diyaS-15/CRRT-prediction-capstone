"""Shared SHAP explainability logic used by both training (global summary
plot over the test set, in train_xgb.py) and serving (per-patient
explanation, in serving/app.py) — one implementation, so a per-patient
explanation shown to a clinician can't silently drift from how the model was
actually explained during training/reporting.
"""
import numpy as np
import pandas as pd
import shap


def make_explainer(pipeline, background_transformed: np.ndarray) -> shap.PermutationExplainer:
    """PermutationExplainer wraps predict_proba directly rather than reading
    model internals, so the same code works unmodified across
    XGBoost/CatBoost/LightGBM pipelines."""
    return shap.PermutationExplainer(
        pipeline.named_steps["model"].predict_proba,
        background_transformed,
    )


def compute_shap_values(pipeline, background_transformed: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for the positive (CRRT=1) class. Shape: (n_rows, n_features)."""
    transformed = pipeline.named_steps["prep"].transform(X)
    explainer = make_explainer(pipeline, background_transformed)
    return explainer(transformed).values[:, :, 1]


def explain_instance(pipeline, background_transformed: np.ndarray, row_df: pd.DataFrame, feature_names) -> pd.Series:
    """Per-patient SHAP contribution for a single-row dataframe, indexed by feature name."""
    values = compute_shap_values(pipeline, background_transformed, row_df)
    return pd.Series(values[0], index=feature_names)
