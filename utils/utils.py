import os
import joblib
import streamlit as st

@st.cache_resource
def load_model(choice):
    model_map = {
        "XGBoost": "models/xgb_model.pkl",
        "Random Forest": "models/rf_model.pkl",
        "Logistic Regression": "models/logistic_model.pkl"
    }

    path = model_map[choice]

    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Run main.py first.")
        st.stop()

    return joblib.load(path)
