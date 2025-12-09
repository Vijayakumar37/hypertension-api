🩺 Hypertension Prediction Web App
An End-to-End Machine Learning + Streamlit Deployment Project

This project predicts whether a person is at risk of hypertension based on lifestyle, medical history, and demographic factors.
It includes data preprocessing, model training, artifact saving, and a fully deployed Streamlit UI.

🚀 Live Demo

🔗 Streamlit App: https://hypertension-api-3imvidcbqjnbn3wqsdnsql.streamlit.app/

🔗 GitHub Repo: https://github.com/Vijayakumar37/hypertension-api

📊 Project Features
✔ Machine Learning Pipeline

Data cleaning & preprocessing

Outlier removal using IQR

Label encoding of categorical variables

Feature scaling using StandardScaler

Training multiple models (Random Forest, SVM, Logistic Regression)

Selecting best model based on accuracy

Saving model, scaler, and encoders (.pkl + .json)

✔ Web App Features (Streamlit)

Clean and simple user interface

Dropdowns for BP History, Medication Type, and other factors

Real-time prediction

Risk message display (High Risk / Not at Risk)

Runs online with Streamlit Cloud

🧠 Input Features Used
Feature	Description
Age	Person’s age
Salt Intake	Daily salt consumption
Stress Score	Stress level (1–10)
BP History	Normal / Prehypertension / Hypertension
Sleep Duration	Hours slept per day
BMI	Body Mass Index
Medication	ACE inhibitor / Beta blocker / Diuretic / Other
Family History	Yes / No
Exercise Level	Low / Moderate / High
Smoking Status	Smoker / Non-smoker

🧰 Tech Stack

Python

Pandas, NumPy

Scikit-Learn

Joblib

Streamlit

GitHub

Streamlit Cloud (Deployment)

🛠 How to Run Locally
git clone https://github.com/<Vijayakumar37>/<hypertension-api>.git
cd hypertension
pip install -r requirements.txt
streamlit run hypertension_streamlit.py

📁 Project Structure
hypertension/
│── app.py   # Streamlit app
│── hypertension_rf_model.pkl
│── hypertension_scaler.pkl
│── hypertension_label_encoders.json
│── requirements.txt
│── README.md

🎯 Key Learnings

Building complete ML workflows

Training and evaluating ML models

Saving/loading model artifacts

Creating interactive UIs with Streamlit

Deploying ML apps to the cloud

Understanding healthcare analytics

🤝 Connect With Me

If you liked this project or want to collaborate, feel free to connect on LinkedIn:

🔗 https://github.com/Vijayakumar37

⭐ Show Your Support

If this project helped you, consider giving it a ⭐ on GitHub!
