import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# Load model, scaler, encoder metadata
@st.cache_resource
def load_artifacts():
    model = joblib.load("hypertension_rf_model.pkl")
    scaler = joblib.load("hypertension_scaler.pkl")
    with open("hypertension_label_encoders.json", "r") as f:
        enc_meta = json.load(f)
    return model, scaler, enc_meta

model, scaler, enc_meta = load_artifacts()

st.title("🩺 Hypertension Risk Prediction App")
st.write("Enter patient details below to predict whether they have hypertension.")

# 🔹 Helper: get classes for a column (if it's categorical)
def get_classes(col_name):
    if col_name in enc_meta:
        return enc_meta[col_name]
    return None

# --- INPUT FORM ---

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        salt_intake = st.number_input("Salt Intake (grams/day approx)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
        stress_score = st.slider("Stress Score (1–10)", min_value=1, max_value=10, value=5)

        sleep_duration = st.number_input("Sleep Duration (hours/day)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    with col2:
        # Categorical options from enc_meta if available, else fallback
        bp_hist_classes = get_classes("BP_History") or ["Normal", "Borderline", "High"]
        med_classes = get_classes("Medication") or ["Yes", "No"]
        fam_hist_classes = get_classes("Family_History") or ["Yes", "No"]
        ex_level_classes = get_classes("Exercise_Level") or ["Low", "Moderate", "High"]
        smoke_classes = get_classes("Smoking_Status") or ["Smoker", "Non-Smoker"]

        bp_history = st.selectbox("Blood Pressure History", bp_hist_classes)
        medication = st.selectbox("On Medication", med_classes)
        family_history = st.selectbox("Family History of Hypertension", fam_hist_classes)
        exercise_level = st.selectbox("Exercise Level", ex_level_classes)
        smoking_status = st.selectbox("Smoking Status", smoke_classes)

    submitted = st.form_submit_button("Predict Hypertension")

# --- PREPROCESS + PREDICT ---

def preprocess_input(data_dict, scaler, enc_meta):
    df_row = pd.DataFrame([data_dict])

    # Encode categorical columns
    for col, classes in enc_meta.items():
        if col == "Has_Hypertension":
            continue  # target, not input
        if col in df_row.columns:
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            df_row[col] = df_row[col].map(mapping)

    cols_order = [
        "Age",
        "Salt_Intake",
        "Stress_Score",
        "BP_History",
        "Sleep_Duration",
        "BMI",
        "Medication",
        "Family_History",
        "Exercise_Level",
        "Smoking_Status"
    ]

    df_row = df_row[cols_order]

    X_scaled = scaler.transform(df_row)
    return X_scaled

if submitted:
    # Prepare input dict
    input_data = {
        "Age": age,
        "Salt_Intake": salt_intake,
        "Stress_Score": stress_score,
        "BP_History": bp_history,
        "Sleep_Duration": sleep_duration,
        "BMI": bmi,
        "Medication": medication,
        "Family_History": family_history,
        "Exercise_Level": exercise_level,
        "Smoking_Status": smoking_status
    }

    try:
        X_scaled = preprocess_input(input_data, scaler, enc_meta)
        pred = model.predict(X_scaled)[0]

        # Decode prediction back to Yes/No
        target_classes = enc_meta["Has_Hypertension"]
        label = target_classes[int(pred)]

        if label == "Yes":
            st.error(f"⚠️ Prediction: HIGH RISK of Hypertension ({label})")
        else:
            st.success(f"✅ Prediction: NO Hypertension ({label})")

        st.write("**Encoded Prediction:**", int(pred))
    except Exception as e:
        st.write("❌ Error while predicting:", e)
