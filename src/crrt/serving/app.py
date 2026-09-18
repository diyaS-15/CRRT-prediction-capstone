import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

from src.crrt.features.preprocessing import engineer_features, FEATURE_COLS
from src.crrt.training.mlflow_utils import init_mlflow

MLFLOW_MODEL_NAME = "crrt-xgb"
FALLBACK_MODEL_PATH = "reports/xgb_pipeline.joblib"
DECISION_THRESHOLD = 0.5

st.set_page_config(page_title="CRRT Risk Predictor", layout="centered")


@st.cache_resource
def load_model():
    """Load whichever model version is tagged "Production" in the MLflow
    Model Registry -- that's the explicit, auditable answer to "which model
    is live," set by promote_model.py rather than by whichever .joblib file
    happens to be sitting in reports/. Falls back to the local file only if
    the registry has no Production version yet (e.g. a fresh clone before
    promote_model.py has ever been run), so the app still works out of the
    box for a first-time demo.
    """
    try:
        import mlflow.sklearn
        init_mlflow()
        # sklearn.load_model (not pyfunc.load_model) so we get back the real
        # Pipeline object with .predict_proba, not just a .predict-only wrapper.
        model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")
        st.session_state["model_source"] = f"MLflow Registry: {MLFLOW_MODEL_NAME}/Production"
        return model
    except Exception as registry_error:
        if not os.path.exists(FALLBACK_MODEL_PATH):
            raise FileNotFoundError(
                f"No Production model in the MLflow registry ({registry_error}) "
                f"and no fallback file at {FALLBACK_MODEL_PATH}."
            )
        st.session_state["model_source"] = f"local fallback file: {FALLBACK_MODEL_PATH} (no Production model registered yet)"
        return joblib.load(FALLBACK_MODEL_PATH)


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

st.title("CRRT Risk Prediction Tool")
st.write("Enter patient information below and click **Predict**.")

try:
    model = load_model()
    st.caption(f"Serving model: {st.session_state.get('model_source', 'unknown')}")
except Exception as e:
    st.error("Unable to load the model.")
    st.code(str(e))
    st.stop()

with st.form("patient_form"):
    st.subheader("Patient Information")

    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    admission_weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)

    tbsa_2nd = st.number_input("2nd Degree TBSA %", min_value=0.0, max_value=100.0, value=10.0)
    tbsa_3rd = st.number_input("3rd Degree TBSA %", min_value=0.0, max_value=100.0, value=5.0)

    inhalation_injury = st.checkbox("Inhalation Injury")
    hours_injury_to_admission = st.number_input(
        "Hours from Injury to Admission",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

    total_crystalloid_ml_first_24h = st.number_input(
        "Total Crystalloid Fluids in First 24h (mL)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )
    total_colloid_ml_first_24h = st.number_input(
        "Total Colloid Fluids in First 24h (mL)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )
    total_urine_output_ml_first_24h = st.number_input(
        "Urine Output in First 24h (mL)",
        min_value=0.0,
        value=1000.0,
        step=50.0
    )

    initial_temp_c = st.number_input(
        "Initial Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=37.0,
        step=0.1
    )
    carboxyhemoglobin = st.number_input(
        "Carboxyhemoglobin %",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1
    )

    st.markdown("**Comorbidities**")
    diabetes = st.checkbox("Diabetes")
    hypertension = st.checkbox("Hypertension")
    chronic_kidney_disease = st.checkbox("Chronic Kidney Disease")

    submitted = st.form_submit_button("Predict")


if submitted:
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

        st.subheader("Model Input")
        st.dataframe(input_df, use_container_width=True)

        pred, proba = make_prediction(model, input_df, DECISION_THRESHOLD)

        st.subheader("Prediction Result")
        st.metric("Predicted Risk Probability", f"{proba:.3f}")
        st.write(f"Decision Threshold: **{DECISION_THRESHOLD:.2f}**")

        if pred == 1:
            st.error("Prediction: Higher CRRT Risk")
        else:
            st.success("Prediction: Lower CRRT Risk")

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
