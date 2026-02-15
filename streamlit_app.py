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
 
# Custom UI Styling
st.markdown("""
<style>
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        border: 1px solid #e1e4e8;
    }
</style>
    """, unsafe_allow_html=True)
 
st.title("🚗 BMW High-End vs Standard Classifier")
st.write("BITS Pilani Assignment 2: End-to-End ML Deployment")
 
# --- TOP SECTION: DATA UPLOADER ---
st.subheader("Step 1: Data Acquisition")
uploaded_file = st.file_uploader("Upload BMW dataset (CSV) for testing", type="csv")
 
# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_choice = st.selectbox(
        "Choose Classification Model",
        ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
    )
    st.divider()
    st.info("💡 Threshold: 'High-End' is MSRP > $55,695.")
 
# Filename Mapping
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}
 
def preprocess_data(df):
    """Robust preprocessing that handles missing values for all features."""
    data = df.copy()
    # 1. Target definition
    if 'Target' not in data.columns and 'MSRP_USD' in data.columns:
        data['Target'] = (data['MSRP_USD'] > 55695).astype(int)
    # 2. Features list
    features = ['Year', 'Displacement_L', 'Cylinders', 'Horsepower', 'Torque_lb_ft',
                '0_60_mph_sec', 'Top_Speed_mph', 'Fuel_Economy_City_mpg', 
                'Fuel_Economy_Highway_mpg', 'Seating_Capacity', 'Series', 'Body_Type']
    # 3. Global Imputation (Fixes the NaN error)
    for col in features:
        if col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                # Fill numeric NaNs with median
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val if not pd.isna(median_val) else 0)
            else:
                # Fill categorical NaNs with 'Unknown'
                data[col] = data[col].fillna("Unknown")
        else:
            # If feature is missing entirely, create it with 0s
            data[col] = 0
    # 4. Encoding categorical columns used in features
    le = LabelEncoder()
    for col in ['Series', 'Body_Type']:
        data[col] = le.fit_transform(data[col].astype(str))
    return data[features], data['Target'] if 'Target' in data.columns else None
 
# --- MAIN CONTENT ---
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
 
    try:
        # Load Model
        model_path = os.path.join("model", model_map[model_choice])
        model = joblib.load(model_path)
        # Preprocess (Cleans all NaNs)
        X_test, y_true = preprocess_data(df_test)
        # Scaling
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📈 Analysis", "📋 Preview"])
 
        with tab1:
            if y_true is not None:
                st.subheader(f"Performance: {model_choice}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
                c2.metric("AUC Score", f"{roc_auc_score(y_true, y_prob):.3f}")
                c3.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")
                c4, c5, c6 = st.columns(3)
                c4.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
                c5.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
                c6.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.3f}")
            else:
                st.warning("MSRP_USD column missing; metrics cannot be calculated.")
 
        with tab2:
            if y_true is not None:
                l, r = st.columns(2)
                with l:
                    st.write("**Confusion Matrix**")
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
                    st.pyplot(fig)
                with r:
                    st.write("**Report**")
                    st.code(classification_report(y_true, y_pred))
 
        with tab3:
            st.dataframe(df_test.head(10), use_container_width=True)
 
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👆 Please upload the BMW CSV file to begin.")
