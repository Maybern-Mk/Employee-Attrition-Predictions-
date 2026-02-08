📉 Employee Attrition Detection System

This project builds a machine learning–based employee attrition prediction system using HR analytics data.
It predicts whether an employee is likely to leave the organization and provides a risk probability, helping HR teams take proactive retention measures.

The solution covers the entire ML lifecycle — from data exploration and preprocessing to model training, evaluation, deployment, and a Streamlit web app.

🧠 Problem Statement

Employee attrition leads to:

Increased hiring and training costs

Loss of experienced talent

Reduced team productivity

The goal of this project is to predict employee attrition early using historical HR data and machine learning models.

📂 Dataset

Dataset Name: IBM HR Analytics – Employee Attrition & Performance

File: WA_Fn-UseC_-HR-Employee-Attrition.csv

Target Variable

Attrition → Yes / No

Key Features Used

MonthlyIncome

Age

JobSatisfaction

Department

Education

Gender

🛠️ Tech Stack & Libraries

Python

Pandas, NumPy – data manipulation

Matplotlib, Seaborn – data visualization

Scikit-learn – ML models & preprocessing

Joblib – model persistence

Streamlit – interactive web application

🔍 Exploratory Data Analysis (EDA)

Dataset overview, shape, and statistical summary

Missing value detection and handling

Outlier detection using IQR method

Attrition distribution analysis

Feature-wise analysis vs Attrition

Correlation analysis with target variable

Boxplots and distribution plots for numeric features

⚙️ Data Preprocessing

Categorical Encoding

Label Encoding (for analysis)

One-Hot Encoding (for modeling)

Feature Scaling

StandardScaler for numeric features

ColumnTransformer

Ensures consistent preprocessing during training and inference

Train–Test Split

70% Training, 30% Testing (Stratified)

🤖 Machine Learning Models Used
Model	Purpose
Logistic Regression	Baseline classifier
Decision Tree	Rule-based learning
Random Forest	Ensemble learning
Support Vector Machine (SVM)	Non-linear classification
Voting Classifier (Final Model)	Soft-voting ensemble
📊 Model Evaluation Metrics

Each model is evaluated using:

Accuracy

Precision

Recall

F1-Score

ROC-AUC Score

The Voting Classifier was selected as the final model due to its balanced performance across all metrics.

🏆 Final Model

Model: Soft Voting Classifier

Components: Logistic Regression, Decision Tree, Random Forest

Saved as: Attrition_detection_model.pkl

Additional artifacts:

Feature names stored for consistency

Preprocessing pipeline reused during prediction

🔮 Prediction Pipeline

Accepts raw employee details

Automatically:

Handles missing values

Applies same preprocessing as training

Generates prediction + probability

Outputs:

Will Employee Leave? (Yes / No)

Probability of Attrition

🌐 Streamlit Web Application

An interactive Streamlit dashboard is included for real-time predictions.

Features:

User-friendly form for employee details

One-click attrition prediction

Clear risk probability display

HR-ready interface

Run the App
streamlit run app.py

📦 Installation
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit

🚀 How to Run the Project

Clone the repository

Install dependencies

Run the Jupyter notebook for training

Launch the Streamlit app for predictions

🎯 Project Highlights

✔ End-to-end ML pipeline
✔ Proper preprocessing with ColumnTransformer
✔ Multiple model comparison
✔ Ensemble learning
✔ Model persistence
✔ Real-world HR use case
✔ Deployment using Streamlit

📌 Future Improvements

Hyperparameter tuning

SHAP / feature importance explainability

Handling class imbalance

Integration with real HR systems

Cloud deployment
