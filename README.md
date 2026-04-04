# Employee Attrition Detection System

## Overview  
This project develops a machine learning-based system to predict employee attrition using HR analytics data.  

The model estimates the likelihood of an employee leaving the organization and provides a risk probability, enabling HR teams to take proactive retention measures. The project covers the complete machine learning lifecycle, including data analysis, preprocessing, model training, evaluation, and deployment through a Streamlit web application.

---

## Problem Statement  
Employee attrition leads to increased hiring costs, loss of experienced talent, and reduced productivity.  

The objective of this project is to predict employee attrition early using historical HR data and machine learning techniques to support data-driven decision-making.

---

## Dataset  
- Dataset: IBM HR Analytics – Employee Attrition & Performance  
- File: `WA_Fn-UseC_-HR-Employee-Attrition.csv`  

### Target Variable  
- `Attrition` (Yes / No)  

### Key Features  
- MonthlyIncome  
- Age  
- JobSatisfaction  
- Department  
- Education  
- Gender  

---

## Tools and Technologies  
- **Python**  
- **Pandas**, **NumPy** for data manipulation  
- **Matplotlib**, **Seaborn** for visualization  
- **Scikit-learn** for machine learning and preprocessing  
- **Joblib** for model persistence  
- **Streamlit** for deployment  

---

## Exploratory Data Analysis  
- Dataset structure and statistical summary  
- Missing value detection and handling  
- Outlier detection using IQR method  
- Attrition distribution analysis  
- Feature-wise analysis against attrition  
- Correlation analysis with target variable  
- Visualization using box plots and distribution plots  

---

## Data Preprocessing  

### Categorical Features  
- Label encoding for analysis  
- One-hot encoding for modeling  

### Numerical Features  
- Feature scaling using StandardScaler  

### Pipeline  
- Implemented using ColumnTransformer  
- Ensures consistent preprocessing during training and inference  

### Train-Test Split  
- 70% training, 30% testing  
- Stratified sampling for balanced distribution  

---

## Machine Learning Models  

| Model                  | Purpose                  |
|-----------------------|--------------------------|
| Logistic Regression   | Baseline model           |
| Decision Tree         | Rule-based learning      |
| Random Forest         | Ensemble model           |
| Support Vector Machine| Non-linear classification|
| Voting Classifier     | Final ensemble model     |

---

## Model Evaluation  

### Metrics Used  
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC Score  

### Result  
The Voting Classifier demonstrated the most balanced performance across all evaluation metrics and was selected as the final model.

---

## Final Model  
- Model: Soft Voting Classifier  
- Components: Logistic Regression, Decision Tree, Random Forest  
- Saved as: `Attrition_detection_model.pkl`  

### Additional Artifacts  
- Stored feature names for consistency  
- Reusable preprocessing pipeline for predictions  

---

## Prediction Pipeline  
- Accepts raw employee input data  
- Applies preprocessing automatically  
- Generates:  
  - Attrition prediction (Yes / No)  
  - Probability score  

---

## Streamlit Application  

An interactive web application is included for real-time predictions.

### Features  
- User-friendly input form  
- One-click prediction  
- Probability-based risk output  
- HR-focused interface  

### Run the App  
streamlit run app.py

---

## Installation  
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit

---

## How to Run  

### 1. Setup  
- Clone the repository  
- Install dependencies  

### 2. Model Training  
- Run the Jupyter notebook  

### 3. Deployment  
- Launch the Streamlit application  

---

## Project Highlights  
- End-to-end machine learning pipeline  
- Structured preprocessing using ColumnTransformer  
- Multiple model comparison and evaluation  
- Ensemble learning using Voting Classifier  
- Model persistence with Joblib  
- Real-world HR analytics use case  
- Deployment using Streamlit  

---

## Business Value  
- Enables early identification of at-risk employees  
- Supports HR teams in improving retention strategies  
- Reduces hiring and training costs  
- Enhances workforce planning and decision-making  

---

## Use Case  
This system can be integrated into HR analytics platforms to monitor employee risk levels and support proactive retention initiatives.

---

## Future Enhancements  
- Hyperparameter tuning for improved performance  
- Model explainability using SHAP or feature importance  
- Handling class imbalance  
- Integration with real HR management systems  
- Cloud deployment for scalability  

---

## Author  
**Mrudul Paku**  
Data Analytics | Machine Learning | HR Analytics | Data Science  

