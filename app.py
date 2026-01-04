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

# ---------------- LOAD MODEL ----------------
model = joblib.load("movie_success_model.pkl")

# ---------------- TITLE ----------------
st.title("🎬 Movie Success Predictor")
st.write(
    "Predict whether a movie will be successful using IMDb metadata, "
    "an XGBoost model, and SHAP explainability."
)

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🎛 Movie Inputs")

runtime_minutes = st.sidebar.slider(
    "Runtime (minutes)",
    min_value=60,
    max_value=240,
    value=120
)

num_votes = st.sidebar.slider(
    "Number of Votes",
    min_value=0,
    max_value=500000,
    value=1000,
    step=100
)

average_rating = st.sidebar.slider(
    "IMDb Rating",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
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

# ---------------- FEATURE NAMES (ORDER MATTERS) ----------------
feature_names = [
    "Runtime",
    "Votes",
    "Rating",
    *available_genres
]

# ---------------- PREDICTION ----------------
if st.button("🎯 Predict Success"):

    # Encode genres (must match training order)
    genre_input = [1 if g in selected_genres else 0 for g in available_genres]

    X = np.array([[
        runtime_minutes,
        num_votes,
        average_rating,
        *genre_input
    ]])

    # Predict probability
    probability = model.predict_proba(X)[0][1]

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

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("🧠 Why this prediction? (SHAP Explanation)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values[0]
    }).sort_values(
        by="Impact",
        key=abs,
        ascending=False
    )

    st.dataframe(shap_df)

    # Optional visual chart
    st.bar_chart(
        shap_df.set_index("Feature")["Impact"]
    )

# ---------------- GLOBAL FEATURE IMPORTANCE ----------------
st.subheader("🌍 Overall Feature Importance (Model-Level)")

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df)
