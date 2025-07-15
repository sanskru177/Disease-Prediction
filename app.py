import streamlit as st
import pandas as pd
import joblib
import os

st.title("Disease Prediction App")

# Dropdowns
disease = st.selectbox("Select Disease", ["Heart Disease", "Diabetes", "Breast Cancer"])
model_name = st.selectbox("Select Model", ["logistic", "random_forest"])

def load_model(disease, model_name):
    prefix = disease.lower().replace(" ", "_")
    model_path = f"models/{prefix}_{model_name}.pkl"
    scaler_path = f"models/{prefix}_scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model files not found.")
        return None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def get_input_features(disease):
    if disease == "Heart Disease":
        return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    elif disease == "Diabetes":
        return ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    elif disease == "Breast Cancer":
        return ['radius_mean', 'texture_mean', 'perimeter_mean',
                'area_mean', 'smoothness_mean']

# Input fields
features = get_input_features(disease)
user_input = {}
for feat in features:
    user_input[feat] = st.number_input(f"Enter {feat}", value=0.0)

# Prediction
if st.button("Predict"):
    model, scaler = load_model(disease, model_name)
    if model is not None:
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        result = "Positive (Disease Detected)" if prediction[0] == 1 else "Negative (No Disease)"
        st.success(f"Prediction: {result}")
