import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
 
# Page Configuration
st.set_page_config(page_title="BMW Price Tier Classifier", layout="wide")
 
st.title("🚗 BMW High-End vs Standard Classifier")
st.markdown("### BITS Pilani ML Assignment 2 - Model Deployment")
 
# --- SIDEBAR: DATA UPLOAD (Requirement 91) ---
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload BMW CSV for Testing", type="csv")
 
# --- SIDEBAR: MODEL SELECTION (Requirement 92) ---
st.sidebar.header("2. Choose Model")
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
)
 
# Map display names to the exact .pkl filenames created in the training script
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}
 
def preprocess_bmw_data(df):
    """Processes BMW features and sets Target based on MSRP median."""
    data = df.copy()
    # Define Target: 1 if MSRP > $55,695 (Median), 0 otherwise
    if 'Target' not in data.columns and 'MSRP_USD' in data.columns:
        data['Target'] = (data['MSRP_USD'] > 55695).astype(int)
    # Fill missing values for fuel economy
    if 'Fuel_Economy_City_mpg' in data.columns:
        data['Fuel_Economy_City_mpg'] = data['Fuel_Economy_City_mpg'].fillna(data['Fuel_Economy_City_mpg'].median())
    if 'Fuel_Economy_Highway_mpg' in data.columns:
        data['Fuel_Economy_Highway_mpg'] = data['Fuel_Economy_Highway_mpg'].fillna(data['Fuel_Economy_Highway_mpg'].median())
 
    # Label Encoding for categorical columns used in training
    le = LabelEncoder()
    cat_cols = ['Series', 'Body_Type', 'Engine_Type', 'Drivetrain', 'Transmission']
    for col in cat_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].astype(str))
    # The 12 features used during training
    features = [
        'Year', 'Displacement_L', 'Cylinders', 'Horsepower', 'Torque_lb_ft',
        '0_60_mph_sec', 'Top_Speed_mph', 'Fuel_Economy_City_mpg', 
        'Fuel_Economy_Highway_mpg', 'Seating_Capacity', 'Series', 'Body_Type'
    ]
    available_features = [f for f in features if f in data.columns]
    return data[available_features], data['Target'] if 'Target' in data.columns else None
 
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.success("Test dataset loaded successfully!")
    with st.expander("Preview Test Data"):
        st.dataframe(df_test.head())
 
    try:
        # Load the selected model from the /model directory
        model_path = os.path.join("model", model_map[model_choice])
        model = joblib.load(model_path)
        # Preprocess
        X_test, y_true = preprocess_bmw_data(df_test)
        # Scaling
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        # --- REQUIREMENT (c): DISPLAY EVALUATION METRICS (Requirement 93) ---
        st.subheader(f"📊 Evaluation Metrics: {model_choice}")
        if y_true is not None:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
            m2.metric("AUC Score", f"{roc_auc_score(y_true, y_prob):.3f}")
            m3.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
            m4.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
            m5.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")
            m6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.3f}")
            # --- REQUIREMENT (d): CONFUSION MATRIX & REPORT (Requirement 94) ---
            st.divider()
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
                st.pyplot(fig)
            with col_r:
                st.subheader("Classification Report")
                st.code(classification_report(y_true, y_pred))
        else:
            st.warning("Column 'MSRP_USD' not found. Evaluation metrics cannot be calculated.")
    except FileNotFoundError:
        st.error(f"Model file '{model_map[model_choice]}' not found in the /model/ folder.")
    except Exception as e:
        st.error(f"Processing Error: {e}")
else:
    st.info("💡 Please upload the BMW CSV file to begin.")
