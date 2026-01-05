import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="centered"
)

# ---------------- SIDEBAR: MODEL SELECTION ----------------
st.sidebar.subheader("🤖 Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["XGBoost", "Random Forest", "Logistic Regression"]
)

# ---------------- LOAD MODEL (DYNAMIC) ----------------
@st.cache_resource
def load_model(choice):
    if choice == "XGBoost":
        return joblib.load("xgb_model.pkl")
    elif choice == "Random Forest":
        return joblib.load("rf_model.pkl")
    else:
        return joblib.load("logistic_model.pkl")

model = load_model(model_choice)

# ---------------- TITLE ----------------
st.title("🎬 Movie Success Predictor")
st.write(
    "Predict whether a movie will be successful using IMDb metadata, "
    "multiple ML models, and explainability."
)

st.caption(f"🔍 Currently using: **{model_choice}**")

# ---------------- SIDEBAR INPUTS ----------------
runtime_minutes = st.sidebar.slider(
    "Runtime (minutes)", 60, 240, 120
)

num_votes = st.sidebar.slider(
    "Number of Votes", 0, 500000, 1000, step=100
)

average_rating = st.sidebar.slider(
    "IMDb Rating", 0.0, 10.0, 7.0, step=0.1
)

# ---------------- GENRE INPUTS ----------------
st.sidebar.subheader("🎭 Genres")

available_genres = [
    "Action", "Comedy", "Drama", "Romance",
    "Thriller", "Horror", "Adventure",
    "Crime", "Sci-Fi", "Fantasy"
]

selected_genres = st.sidebar.multiselect(
    "Select Genres",
    available_genres
)

# ---------------- FEATURE ORDER (MUST MATCH TRAINING) ----------------
FEATURE_COLUMNS = [
    "runtimeMinutes",
    "numVotes",
    "averageRating",
    *available_genres
]

# ---------------- PREDICTION ----------------
if st.button("🎯 Predict Success"):

    genre_features = {g: 0 for g in available_genres}
    for g in selected_genres:
        genre_features[g] = 1

    input_data = {
        "runtimeMinutes": runtime_minutes,
        "numVotes": num_votes,
        "averageRating": average_rating,
        **genre_features
    }

    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    probability = model.predict_proba(input_df)[0][1]

    # ---------------- RESULT UI ----------------
    st.subheader("📊 Prediction Result")

    st.progress(min(int(probability * 100), 100))
    st.metric("Success Probability", f"{probability:.2%}")

    if probability > 0.75:
        st.success("🔥 Very strong success potential!")
    elif probability > 0.5:
        st.info("👍 Moderate success potential")
    else:
        st.warning("❌ Low success potential")

    if model_choice == "Logistic Regression":
        st.info("Note: Logistic Regression is used as a baseline model.")

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("🧠 Why this prediction?")

    if model_choice == "XGBoost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

    elif model_choice == "Random Forest":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

    else:
        st.info("SHAP explanations are not shown for Logistic Regression.")
        shap_values = None

    if shap_values is not None:
        shap_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Impact": shap_values[0]
        }).sort_values(by="Impact", key=abs, ascending=False)

        st.dataframe(shap_df)
        st.bar_chart(shap_df.set_index("Feature")["Impact"])

    # ---------------- OVERALL FEATURES ----------------
st.subheader("🌍 Overall Feature Importance")

if model_choice in ["XGBoost", "Random Forest"]:
    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df)

elif model_choice == "Logistic Regression":
    coef = model.named_steps["clf"].coef_[0]

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Coefficient": coef
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    st.dataframe(importance_df)
