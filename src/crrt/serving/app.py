import json
import os
from datetime import datetime, timedelta

import joblib
import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.crrt.features.preprocessing import FEATURE_COLS, engineer_features
from src.crrt.training.mlflow_utils import init_mlflow
from src.crrt.training.shap_utils import explain_instance

MLFLOW_MODEL_NAME = "crrt-xgb"
FALLBACK_MODEL_PATH = "reports/xgb_pipeline.joblib"
FALLBACK_METRICS_PATH = "reports/xgb_metrics.json"
SHAP_BACKGROUND_PATH = "reports/xgb_shap_background.joblib"
DECISION_THRESHOLD = 0.5

st.set_page_config(page_title="CRRT Risk Predictor", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
      h1 { margin-bottom: 0.1rem; }
      .crrt-subtitle { color: #6b7280; margin-top: 0; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load whichever model version is tagged "Production" in the MLflow
    Model Registry -- that's the explicit, auditable answer to "which model
    is live," set by promote_model.py rather than by whichever .joblib file
    happens to be sitting in reports/. Falls back to the local file only if
    the registry has no Production version yet (e.g. a fresh clone before
    promote_model.py has ever been run), so the app still works out of the
    box for a first-time demo.

    Returns (model, source_description, mlflow_run_id_or_None). Metadata is
    returned rather than written to st.session_state here: this function is
    @st.cache_resource'd and its body only runs once globally (shared across
    all sessions), so a session-state write inside it would only ever be
    visible to whichever session happened to trigger the cache miss -- every
    other session would see a stale/missing value. Returning it lets every
    caller set its own session state from the (possibly cached) result.
    """
    try:
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient
        init_mlflow()
        client = MlflowClient()
        production_versions = [
            v for v in client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
            if v.current_stage == "Production"
        ]
        if not production_versions:
            raise ValueError(f"No Production version registered for '{MLFLOW_MODEL_NAME}'")
        # sklearn.load_model (not pyfunc.load_model) so we get back the real
        # Pipeline object with .predict_proba, not just a .predict-only wrapper.
        model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")
        source = f"MLflow Registry: {MLFLOW_MODEL_NAME}/Production (v{production_versions[0].version})"
        return model, source, production_versions[0].run_id
    except Exception as registry_error:
        if not os.path.exists(FALLBACK_MODEL_PATH):
            raise FileNotFoundError(
                f"No Production model in the MLflow registry ({registry_error}) "
                f"and no fallback file at {FALLBACK_MODEL_PATH}."
            )
        source = f"local fallback file: {FALLBACK_MODEL_PATH} (no Production model registered yet)"
        return joblib.load(FALLBACK_MODEL_PATH), source, None


@st.cache_resource
def load_shap_background():
    if not os.path.exists(SHAP_BACKGROUND_PATH):
        return None
    return joblib.load(SHAP_BACKGROUND_PATH)


@st.cache_data
def load_serving_metrics(_model_run_id):
    """Pull the metrics of the model actually being served -- the MLflow run
    behind the Production version if we loaded from the registry, otherwise
    the local metrics.json written by the most recent training run. Used to
    show real recall/PR-AUC next to a prediction instead of just a bare
    probability (see the small-sample-size caveat rendered alongside it).
    """
    if _model_run_id:
        try:
            from mlflow.tracking import MlflowClient
            init_mlflow()
            run = MlflowClient().get_run(_model_run_id)
            m = run.data.metrics
            return {
                "test_sensitivity": m.get("test_sensitivity"),
                "test_pr_auc": m.get("test_pr_auc"),
                "test_tp": m.get("test_tp"),
                "test_fn": m.get("test_fn"),
                "source": "MLflow run metrics",
            }
        except Exception:
            pass
    if os.path.exists(FALLBACK_METRICS_PATH):
        with open(FALLBACK_METRICS_PATH) as f:
            m = json.load(f)
        return {
            "test_sensitivity": m.get("test_sensitivity"),
            "test_pr_auc": m.get("test_pr_auc"),
            "test_tp": m.get("test_tp"),
            "test_fn": m.get("test_fn"),
            "source": f"{FALLBACK_METRICS_PATH}",
        }
    return None


def build_raw_row(
    age,
    admission_weight_kg,
    tbsa_2nd,
    tbsa_3rd,
    inhalation_injury,
    hours_injury_to_admission,
    total_crystalloid_ml_first_24h,
    total_colloid_ml_first_24h,
    total_urine_output_ml_first_24h,
    initial_temp_c,
    carboxyhemoglobin,
    diabetes,
    hypertension,
    chronic_kidney_disease,
) -> pd.DataFrame:
    """Assemble a single-row dataframe matching the raw schema that
    features/preprocessing.py::engineer_features expects, so serving-time
    feature engineering is identical to training-time feature engineering.
    """
    admission_datetime = datetime.now()
    injury_datetime = admission_datetime - timedelta(hours=hours_injury_to_admission)

    comorbidity_parts = []
    if diabetes:
        comorbidity_parts.append("diabetes")
    if hypertension:
        comorbidity_parts.append("hypertension")
    if chronic_kidney_disease:
        comorbidity_parts.append("chronic kidney disease")
    comorbidity = ", ".join(comorbidity_parts) if comorbidity_parts else "none"

    raw_row = {
        "age": age,
        "tbsa_2nd_3rd": tbsa_2nd + tbsa_3rd,
        "inhalation_injury": "yes" if inhalation_injury else "no",
        "injury_datetime": injury_datetime,
        "admission_datetime": admission_datetime,
        "total_crystalloid_ml_first_24h": total_crystalloid_ml_first_24h,
        "total_colloid_ml_first_24h": total_colloid_ml_first_24h,
        "total_urine_output_ml_first_24h": total_urine_output_ml_first_24h,
        "admission_weight_kg": admission_weight_kg,
        "carboxyhemoglobin": carboxyhemoglobin,
        "initial_temp_c": initial_temp_c,
        "comorbidity": comorbidity,
    }
    return pd.DataFrame([raw_row])


def make_prediction(model, input_df: pd.DataFrame, threshold: float = 0.5):
    proba = model.predict_proba(input_df)[0][1]
    pred = 1 if proba >= threshold else 0
    return pred, proba


def numeric_field(label, default, lo, hi, help_text, is_int=False):
    """A validated free-text numeric input, replacing st.number_input's
    stepper (slower to use for values like fluid volumes where you're
    typing, not incrementing). Returns (value_or_None, error_or_None) so the
    caller can collect every error before showing them, rather than failing
    silently on the first bad field.
    """
    raw = st.text_input(label, value=str(default), help=help_text)
    raw = raw.strip()
    if not raw:
        return None, f"**{label}** is required."
    try:
        val = int(raw) if is_int else float(raw)
    except ValueError:
        return None, f"**{label}**: \"{raw}\" is not a valid number."
    if not (lo <= val <= hi):
        return None, f"**{label}**: {val} is outside the plausible range ({lo}–{hi})."
    return val, None


def render_shap_chart(contributions: pd.Series, top_n: int = 8):
    """Horizontal bar chart of this patient's top SHAP contributions --
    positive values pushed the prediction toward higher CRRT risk, negative
    values pushed it toward lower risk."""
    clean_names = {c: c.replace("num__", "").replace("_", " ") for c in contributions.index}
    top = contributions.reindex(contributions.abs().sort_values(ascending=False).index[:top_n])
    top = top.rename(index=clean_names).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = ["#d1495b" if v > 0 else "#2e86ab" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (→ higher CRRT risk)")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


st.title("CRRT Risk Prediction Tool")
st.markdown(
    '<p class="crrt-subtitle">Estimates the likelihood of CRRT need within 48h of admission '
    "for a burn patient, from first-24h clinical data.</p>",
    unsafe_allow_html=True,
)

try:
    model, model_source, model_run_id = load_model()
    st.session_state["model_source"] = model_source
    st.session_state["model_run_id"] = model_run_id
    st.caption(f"Serving model: {model_source}")
except Exception as e:
    st.error("Unable to load the model.")
    st.code(str(e))
    st.stop()

with st.form("patient_form"):
    errors = []

    with st.container(border=True):
        st.subheader("Demographics & Burn Injury")
        col1, col2, col3 = st.columns(3)
        with col1:
            age, err = numeric_field("Age (years)", 45, 0, 120, "Typical range: 0–120 years", is_int=True)
            errors.append(err)
        with col2:
            admission_weight_kg, err = numeric_field(
                "Weight (kg)", 70.0, 1, 300, "Typical adult: 40–120 kg"
            )
            errors.append(err)
        with col3:
            hours_injury_to_admission, err = numeric_field(
                "Hours: injury → admission", 2.0, 0, 72,
                "Most direct admissions: 0–6 hours; >6h is considered late presentation",
            )
            errors.append(err)

        col1, col2, col3 = st.columns(3)
        with col1:
            tbsa_2nd, err = numeric_field(
                "2nd Degree TBSA %", 10.0, 0, 100, "0–100% of total body surface area"
            )
            errors.append(err)
        with col2:
            tbsa_3rd, err = numeric_field(
                "3rd Degree TBSA %", 5.0, 0, 100, "0–100% of total body surface area"
            )
            errors.append(err)
        with col3:
            st.write("")
            inhalation_injury = st.checkbox("Inhalation Injury", help="Suspected or confirmed smoke inhalation injury")

    with st.container(border=True):
        st.subheader("Fluids & Vitals (First 24h)")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_crystalloid_ml_first_24h, err = numeric_field(
                "Crystalloid fluids (mL)", 5000.0, 0, 20000,
                "Parkland formula estimate: ~2–4 mL × kg × %TBSA",
            )
            errors.append(err)
        with col2:
            total_colloid_ml_first_24h, err = numeric_field(
                "Colloid fluids (mL)", 0.0, 0, 5000,
                "Often 0 in the first 24h; some protocols start colloids later",
            )
            errors.append(err)
        with col3:
            total_urine_output_ml_first_24h, err = numeric_field(
                "Urine output (mL)", 1000.0, 0, 10000,
                "Target ~0.5 mL/kg/hr (~840 mL/24h for a 70 kg adult)",
            )
            errors.append(err)

        col1, col2 = st.columns(2)
        with col1:
            initial_temp_c, err = numeric_field(
                "Initial temperature (°C)", 37.0, 30, 42,
                "Normal core temp: 36.5–37.5°C; <36°C = hypothermia",
            )
            errors.append(err)
        with col2:
            carboxyhemoglobin, err = numeric_field(
                "Carboxyhemoglobin %", 2.0, 0, 100,
                "Normal: <3% (nonsmoker), <10% (smoker); ≥25% = severe CO poisoning risk",
            )
            errors.append(err)

    with st.container(border=True):
        st.subheader("Comorbidities")
        col1, col2, col3 = st.columns(3)
        with col1:
            diabetes = st.checkbox("Diabetes")
        with col2:
            hypertension = st.checkbox("Hypertension")
        with col3:
            chronic_kidney_disease = st.checkbox("Chronic Kidney Disease")

    submitted = st.form_submit_button("Predict", type="primary")


if submitted:
    errors = [e for e in errors if e]
    if errors:
        st.error("Please fix the following before predicting:\n\n" + "\n\n".join(f"- {e}" for e in errors))
        st.stop()

    try:
        raw_row = build_raw_row(
            age=age,
            admission_weight_kg=admission_weight_kg,
            tbsa_2nd=tbsa_2nd,
            tbsa_3rd=tbsa_3rd,
            inhalation_injury=inhalation_injury,
            hours_injury_to_admission=hours_injury_to_admission,
            total_crystalloid_ml_first_24h=total_crystalloid_ml_first_24h,
            total_colloid_ml_first_24h=total_colloid_ml_first_24h,
            total_urine_output_ml_first_24h=total_urine_output_ml_first_24h,
            initial_temp_c=initial_temp_c,
            carboxyhemoglobin=carboxyhemoglobin,
            diabetes=diabetes,
            hypertension=hypertension,
            chronic_kidney_disease=chronic_kidney_disease,
        )

        # Same feature engineering used at training time, so serving can't drift from it.
        engineered = engineer_features(raw_row)
        input_df = engineered[FEATURE_COLS]

        pred, proba = make_prediction(model, input_df, DECISION_THRESHOLD)

        st.divider()
        result_col, caveat_col = st.columns([1, 1])

        with result_col:
            st.subheader("Prediction Result")
            st.metric("Predicted CRRT Risk Probability", f"{proba:.1%}")
            st.caption(f"Decision threshold: {DECISION_THRESHOLD:.2f}")
            if pred == 1:
                st.error("Higher CRRT Risk")
            else:
                st.success("Lower CRRT Risk")

        with caveat_col:
            st.subheader("Model Reliability")
            metrics = load_serving_metrics(st.session_state.get("model_run_id"))
            if metrics and metrics.get("test_sensitivity") is not None:
                n_pos = None
                if metrics.get("test_tp") is not None and metrics.get("test_fn") is not None:
                    n_pos = int(metrics["test_tp"] + metrics["test_fn"])
                sens_pct = f"{metrics['test_sensitivity']:.1%}"
                prauc_pct = f"{metrics['test_pr_auc']:.1%}" if metrics.get("test_pr_auc") is not None else "N/A"
                st.metric("Test Recall (Sensitivity)", sens_pct)
                st.metric("Test PR-AUC", prauc_pct)
                n_pos_note = f" (only {n_pos} positive cases in the test set)" if n_pos is not None else ""
                st.info(
                    "⚠️ This model is trained on a small dataset (~200 patients). "
                    f"Recall and precision estimates carry meaningful uncertainty{n_pos_note}. "
                    "Treat this prediction as a decision-support signal, not a clinical diagnosis."
                )
            else:
                st.warning("No evaluation metrics found for the serving model — treat this prediction with caution.")

        st.subheader("What drove this prediction")
        background = load_shap_background()
        if background is None:
            st.info("Per-patient SHAP explanation unavailable (no background sample found — re-run training to enable it).")
        else:
            with st.spinner("Computing SHAP explanation..."):
                feature_names = model.named_steps["prep"].get_feature_names_out()
                contributions = explain_instance(model, background, input_df, feature_names)
            render_shap_chart(contributions)
            st.caption(
                "Red bars push the prediction toward higher CRRT risk; blue bars push it toward lower risk. "
                "Computed with the same SHAP method used in training/reporting (src/crrt/training/shap_utils.py)."
            )

        with st.expander("Model input (engineered features)"):
            st.dataframe(input_df, width="stretch")

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
