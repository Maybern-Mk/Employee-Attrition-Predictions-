# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Features & Target
X = df[["MonthlyIncome", "Age", "JobSatisfaction", "Department", "Education", "Gender"]]
y = df["Attrition"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=42, stratify=y
)

# Preprocessing
numeric_features = ["MonthlyIncome", "Age", "JobSatisfaction", "Education"]
categorical_features = ["Department", "Gender"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)

# Models
final_model = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("dt", DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ],
    voting="soft"
)

final_model.fit(X_train_processed, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Attrition Predictor", layout="centered")

st.title("📉 Employee Attrition Detection System")
st.write("Predict whether an employee will leave the company.")

st.subheader("Enter Employee Details")

monthly_income = st.number_input("Monthly Income", min_value=0.0, step=500.0)
age = st.number_input("Age", min_value=18, max_value=65, step=1)
job_satisfaction = st.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])

department = st.selectbox(
    "Department",
    ["Sales", "Research & Development", "Human Resources"]
)

education = st.selectbox(
    "Education Level",
    [1, 2, 3, 4, 5]  # numeric since model expects number
)

gender = st.selectbox("Gender", ["Male", "Female"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Attrition"):
    input_data = pd.DataFrame({
        "MonthlyIncome": [monthly_income],
        "Age": [age],
        "JobSatisfaction": [job_satisfaction],
        "Department": [department],
        "Education": [education],
        "Gender": [gender]
    })

    processed = preprocessor.transform(input_data)
    prediction = final_model.predict(processed)[0]
    probability = final_model.predict_proba(processed)[0][1]

    st.subheader("Result")

    if prediction == "Yes":
        st.error(f"⚠️ Employee likely to leave\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Employee likely to stay\n\nProbability: {probability:.2%}")

# -------------------------------
# Run Instructions
# -------------------------------
# Save this file as app.py
# Run using: streamlit run app.py
