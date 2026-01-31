# ATTRITION DETECTION SYSTEM - STREAMLIT APP

import streamlit as st
import pandas as pd
import joblib

# 1. LOAD TRAINED MODEL & SCALER
model = joblib.load("Attrition_detection_model.pkl")   # VotingClassifier model
scaler = joblib.load("scaler.pkl")                     # StandardScaler

# Feature list must match training
feature_names = ["MonthlyIncome"]

# 2. PAGE SETTINGS
st.set_page_config(
    page_title="Employee Attrition Detection System",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Employee Attrition Detection System")
st.write("Predict whether an employee is likely to **leave the company**.")

# 3. USER INPUT SECTION
st.subheader("Enter Employee Details")

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    step=500.0
)

# 4. PREDICTION BUTTON
if st.button("🔍 Predict Attrition"):

    # Create DataFrame in correct feature order
    input_data = pd.DataFrame(
        [[monthly_income]],
        columns=feature_names
    )

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # 5. DISPLAY RESULT
    st.subheader("Prediction Result")

    if prediction == "Yes":
        st.error("⚠️ Employee is likely to leave the company")
    else:
        st.success("✅ Employee is likely to stay in the company")

    st.metric("Attrition Probability", f"{probability:.2%}")
