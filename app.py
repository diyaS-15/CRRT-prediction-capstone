import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "reports/xgb_pipeline.joblib"
DECISION_THRESHOLD = 0.5

st.set_page_config(page_title="CRRT Risk Predictor", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_features(
    age,
    weight_kg,
    tbsa_2nd,
    tbsa_3rd,
    inhalation_injury,
    hours_injury_to_admission,
    fluid_intake_24h,
    fluid_output_24h,
    urine_output_24h,
    temperature_c,
    carboxyhemoglobin,
    baseline_creatinine,
    diabetes,
    hypertension,
    chronic_kidney_disease,
):
    tbsa_2nd_3rd = tbsa_2nd + tbsa_3rd

    if tbsa_2nd_3rd < 10:
        burn_severity_tier = 0
    elif tbsa_2nd_3rd < 20:
        burn_severity_tier = 1
    elif tbsa_2nd_3rd < 40:
        burn_severity_tier = 2
    else:
        burn_severity_tier = 3

    inhalation_flag = 1 if inhalation_injury else 0
    late_admission_flag = 1 if hours_injury_to_admission > 8 else 0

    fluid_balance_24h = fluid_intake_24h - fluid_output_24h
    fluid_overload_flag = 1 if fluid_balance_24h > 4000 else 0

    urine_output_per_kg = urine_output_24h / weight_kg if weight_kg > 0 else 0
    low_urine_output_flag = 1 if urine_output_per_kg < 0.5 else 0

    hypothermia_flag = 1 if temperature_c < 36.0 else 0
    carboxyhemoglobin_risk_flag = 1 if carboxyhemoglobin > 10 else 0

    comorbidity_aki_risk_score = (
        int(diabetes) +
        int(hypertension) +
        (2 * int(chronic_kidney_disease))
    )

    revised_baux_score = age + tbsa_2nd_3rd + (17 if inhalation_flag == 1 else 0)

    return {
        "age": age,
        "weight_kg": weight_kg,
        "tbsa_2nd": tbsa_2nd,
        "tbsa_3rd": tbsa_3rd,
        "tbsa_2nd_3rd": tbsa_2nd_3rd,
        "burn_severity_tier": burn_severity_tier,
        "inhalation_flag": inhalation_flag,
        "hours_injury_to_admission": hours_injury_to_admission,
        "late_admission_flag": late_admission_flag,
        "fluid_balance_24h": fluid_balance_24h,
        "fluid_overload_flag": fluid_overload_flag,
        "urine_output_per_kg": urine_output_per_kg,
        "low_urine_output_flag": low_urine_output_flag,
        "hypothermia_flag": hypothermia_flag,
        "carboxyhemoglobin_risk_flag": carboxyhemoglobin_risk_flag,
        "comorbidity_aki_risk_score": comorbidity_aki_risk_score,
        "baseline_creatinine": baseline_creatinine,
        "revised_baux_score": revised_baux_score,
    }


def make_prediction(model, input_df: pd.DataFrame, threshold: float = 0.5):
    proba = model.predict_proba(input_df)[0][1]
    pred = 1 if proba >= threshold else 0
    return pred, proba

st.title("CRRT Risk Prediction Tool")
st.write("Enter patient information below and click **Predict**.")

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the model.")
    st.code(str(e))
    st.stop()

with st.form("patient_form"):
    st.subheader("Patient Information")

    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)

    tbsa_2nd = st.number_input("2nd Degree TBSA %", min_value=0.0, max_value=100.0, value=10.0)
    tbsa_3rd = st.number_input("3rd Degree TBSA %", min_value=0.0, max_value=100.0, value=5.0)

    inhalation_injury = st.checkbox("Inhalation Injury")
    hours_injury_to_admission = st.number_input(
        "Hours from Injury to Admission",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

    fluid_intake_24h = st.number_input(
        "Fluid Intake in First 24h (mL)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )
    fluid_output_24h = st.number_input(
        "Total Fluid Output in First 24h (mL)",
        min_value=0.0,
        value=2500.0,
        step=100.0
    )
    urine_output_24h = st.number_input(
        "Urine Output in First 24h (mL)",
        min_value=0.0,
        value=1000.0,
        step=50.0
    )

    temperature_c = st.number_input(
        "Temperature (°C)",
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

    baseline_creatinine = st.number_input(
        "Baseline Creatinine",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    st.markdown("**Comorbidities**")
    diabetes = st.checkbox("Diabetes")
    hypertension = st.checkbox("Hypertension")
    chronic_kidney_disease = st.checkbox("Chronic Kidney Disease")

    submitted = st.form_submit_button("Predict")


if submitted:
    try:
        features = build_features(
            age=age,
            weight_kg=weight_kg,
            tbsa_2nd=tbsa_2nd,
            tbsa_3rd=tbsa_3rd,
            inhalation_injury=inhalation_injury,
            hours_injury_to_admission=hours_injury_to_admission,
            fluid_intake_24h=fluid_intake_24h,
            fluid_output_24h=fluid_output_24h,
            urine_output_24h=urine_output_24h,
            temperature_c=temperature_c,
            carboxyhemoglobin=carboxyhemoglobin,
            baseline_creatinine=baseline_creatinine,
            diabetes=diabetes,
            hypertension=hypertension,
            chronic_kidney_disease=chronic_kidney_disease,
        )

        input_df = pd.DataFrame([features])

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