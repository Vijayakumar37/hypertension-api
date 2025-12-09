from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, encoder metadata
model = joblib.load("hypertension_rf_model.pkl")
scaler = joblib.load("hypertension_scaler.pkl")

with open("hypertension_label_encoders.json", "r") as f:
    enc_meta = json.load(f)

# Helper: encode input row like training
def preprocess_input(data_dict):
    """
    data_dict: Python dict with keys:
      Age, Salt_Intake, Stress_Score, BP_History, Sleep_Duration,
      BMI, Medication, Family_History, Exercise_Level, Smoking_Status
    """
    df_row = pd.DataFrame([data_dict])

    # Encode categorical columns using enc_meta (string -> integer)
    for col, classes in enc_meta.items():
        if col == "Has_Hypertension":
            continue  # target, not input
        if col in df_row.columns:
            mapping = {cls: idx for idx, cls in enumerate(classes)}
            df_row[col] = df_row[col].map(mapping)

    # Ensure correct column order
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

    # Scale numeric data
    X_scaled = scaler.transform(df_row)
    return X_scaled

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        X_scaled = preprocess_input(data)
        pred = model.predict(X_scaled)[0]

        # Decode prediction back to original Yes/No using enc_meta
        classes = enc_meta["Has_Hypertension"]
        label = classes[int(pred)]

        return jsonify({
            "prediction": label,
            "encoded_prediction": int(pred)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
