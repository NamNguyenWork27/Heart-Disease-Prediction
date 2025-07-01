import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Prediction")
st.markdown("### Select Model and Enter Patient Information")

model_map = {
    "KNN": "model/heart_model_knn.pkl",
    "SVM": "model/heart_model_svm.pkl",
    "Naive Bayes": "model/heart_model_naive_bayes.pkl",
    "Decision Tree": "model/heart_model_decision_tree.pkl",
    "Random Forest": "model/heart_model_random_forest.pkl",
    "AdaBoost": "model/heart_model_adaboost.pkl",
    "Gradient Boosting": "model/heart_model_gradientboost.pkl",
    "XGBoost": "model/heart_model_xgboost.pkl",
    "Stacking": "model/heart_model_stacking.pkl"
}

model_choice = st.selectbox("🔍 Choose a model:", list(model_map.keys()))
model_path = os.path.join("", model_map[model_choice])

# 2. Patient input form
st.markdown("---")
st.subheader(" Patient Information")

age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Female - 0", "Male - 1"])
cp = st.selectbox("Chest Pain Type", ["Typical angina - 0", "Atypical angina - 1", "Non-anginal pain - 2", "Asymptomatic - 3"])
restbps = st.slider("Resting BP (Restbps)", 80, 200, 120)
chol = st.slider("Cholesterol (Chol)", 100, 600, 200)
fbs = st.radio("Fasting Blood Sugar > 120?", ["No - 0", "Yes - 1"])
restecg = st.selectbox("Resting ECG", ["Normal - 0", "ST-T abnormality - 1", "Left ventricular hypertrophy - 2"])
thalach = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.radio("Exercise Induced Angina?", ["No - 0", "Yes - 1"])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("ST Slope", ["Upsloping - 0", "Flat - 1", "Downsloping - 2"])
ca = st.slider("Number of Major Vessels", 0, 4, 0)
thal = st.selectbox("Thalassemia", ["Normal - 3", "Fixed defect - 1", "Reversible defect - 2"])

# 3. Convert to numerical
sex_val = 1 if sex == "Male" else 0
cp_val = ["Typical angina - 0", "Atypical angina - 1", "Non-anginal pain - 2", "Asymptomatic - 3"].index(cp)
fbs_val = 1 if fbs == "Yes" else 0
restecg_val = ["Normal - 0", "ST-T abnormality - 1", "Left ventricular hypertrophy - 2"].index(restecg)
exang_val = 1 if exang == "Yes" else 0
slope_val = ["Upsloping - 0", "Flat - 1", "Downsloping - 2"].index(slope)
thal_val = ["Normal - 3", "Fixed defect - 1", "Reversible defect - 2"].index(thal)

# 4. Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[
        age, sex_val, cp_val, restbps, chol, fbs_val,
        restecg_val, thalach, exang_val, oldpeak,
        slope_val, ca, thal_val
    ]], columns=[
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal"
    ])

    try:
        model = joblib.load(model_path)

        # Xac suat
        proba = model.predict_proba(input_data)[0][1]  # probability of heart disease
        percent = round(proba * 100, 2)

        # Ket qua logic
        if proba >= 0.5:
            verdict = " High Risk of Heart Disease"
            st.error(f"Prediction with **{model_choice}**: {verdict}")
        else:
            verdict = " Low Risk / No Heart Disease"
            st.success(f"Prediction with **{model_choice}**: {verdict}")

        st.markdown(f"**Probability of Heart Disease:** {percent}%")

        # Them gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            title={'text': "Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if proba >= 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Hien thi do chinh xac (neu co thong tin san)
        if model_choice == "Stacking":
            st.markdown("\U0001F50E *Accuracy on Training set: 88%*")
            st.markdown("\U0001F50E *Accuracy on Test set: 80%*")
        elif model_choice == "Random Forest":
            st.markdown("\U0001F50E *Accuracy ~85% (example)*")
        # ... Add more if desired

    except Exception as e:
        st.error(f"\u26a0\ufe0f Could not load model: {e}")
