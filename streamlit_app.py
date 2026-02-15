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
 
# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="BMW ML Analytics", page_icon="🚗", layout="wide")
 
# Custom CSS for a clean look
st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
    """, unsafe_base64=True)
 
st.title("🚗 BMW High-End vs Standard Classifier")
st.write("An end-to-end ML deployment for BITS Pilani Assignment 2")
 
# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("📋 Input Controls")
    uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv") # Requirement (a) [cite: 91]
    st.divider()
    model_choice = st.selectbox(
        "Select Classification Model",
        ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
    ) # Requirement (b) 
    st.info("The models classify cars based on an MSRP threshold of $55,695.")
 
# Map display names to .pkl files
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}
 
def preprocess_data(df):
    data = df.copy()
    if 'Target' not in data.columns and 'MSRP_USD' in data.columns:
        data['Target'] = (data['MSRP_USD'] > 55695).astype(int)
    le = LabelEncoder()
    for col in ['Series', 'Body_Type', 'Engine_Type', 'Drivetrain', 'Transmission']:
        if col in data.columns:
            data[col] = le.fit_transform(data[col].astype(str))
    features = ['Year', 'Displacement_L', 'Cylinders', 'Horsepower', 'Torque_lb_ft',
                '0_60_mph_sec', 'Top_Speed_mph', 'Fuel_Economy_City_mpg', 
                'Fuel_Economy_Highway_mpg', 'Seating_Capacity', 'Series', 'Body_Type']
    return data[features], data['Target'] if 'Target' in data.columns else None
 
# --- MAIN CONTENT ---
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "📈 Visual Analysis", "📋 Data Preview"])
 
    with tab3:
        st.dataframe(df_test, use_container_width=True)
 
    try:
        # Load Model
        model_path = os.path.join("model", model_map[model_choice])
        model = joblib.load(model_path)
        # Preprocess
        X_test, y_true = preprocess_data(df_test)
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        if y_true is not None:
            with tab1:
                st.subheader(f"Results: {model_choice}") # Requirement (c) 
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy Score", f"{accuracy_score(y_true, y_pred):.3f}")
                m2.metric("AUC Score", f"{roc_auc_score(y_true, y_prob):.3f}")
                m3.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")
                m4, m5, m6 = st.columns(3)
                m4.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
                m5.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
                m6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.3f}")
 
            with tab2:
                col_l, col_r = st.columns(2) # Requirement (d) [cite: 94]
                with col_l:
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                with col_r:
                    st.write("**Detailed Classification Report**")
                    st.code(classification_report(y_true, y_pred))
        else:
            st.warning("Ground truth labels missing for metrics calculation.")
 
    except Exception as e:
        st.error(f"Error loading {model_choice}: {e}")
else:
    st.info("👋 Welcome! Please upload your BMW dataset to begin analysis.")
