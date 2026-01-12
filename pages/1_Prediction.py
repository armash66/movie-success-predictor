# =================================================
# IMPORTS
# =================================================
import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import shap

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="Prediction", layout="centered")
st.title("🎯 Movie Success Prediction")

# =================================================
# LOAD ARTIFACTS (ONCE)
# =================================================
@st.cache_resource
def load_artifacts():
    with open("models/feature_schema.json", "r") as f:
        features = json.load(f)

    xgb_clf = joblib.load("models/xgb_classifier.pkl")
    rf_clf = joblib.load("models/rf_model.pkl")
    log_clf = joblib.load("models/logistic_model.pkl")
    reg_model = joblib.load("models/xgb_regressor.pkl")

    explainer = shap.TreeExplainer(reg_model)

    return features, xgb_clf, rf_clf, log_clf, reg_model, explainer


FEATURES, xgb_clf, rf_clf, log_clf, reg_model, explainer = load_artifacts()

# =================================================
# CONSTANTS
# =================================================
GENRES = [
    "Action", "Comedy", "Drama", "Romance", "Thriller",
    "Horror", "Adventure", "Crime", "Sci-Fi", "Fantasy"
]

# =================================================
# INPUT UI
# =================================================
st.subheader("🎬 Movie Details")

runtime = st.number_input("Runtime (minutes)", 40, 300, 120)
rating = st.slider("Average Rating", 1.0, 10.0, 6.9, 0.1)
votes = st.number_input("Number of Votes", min_value=0, value=3000)
genres = st.multiselect("Genres", GENRES)

st.subheader("👥 Cast & Crew")
num_cast = st.number_input("Total Cast", min_value=0, value=8)
num_lead_cast = st.number_input("Lead Cast", min_value=0, value=3)
num_directors = st.number_input("Directors", min_value=0, value=1)
num_writers = st.number_input("Writers", min_value=0, value=2)

st.subheader("🌍 Market Reach")
num_languages = st.number_input("Languages", min_value=1, value=1)
num_regions = st.number_input("Regions Released", min_value=1, value=1)
is_franchise = st.checkbox("Part of a Franchise")

# =================================================
# PREDICTION MODE
# =================================================
st.subheader("⚙️ Prediction Settings")

prediction_mode = st.radio(
    "Prediction Type",
    ["Smooth Success Score (Recommended)", "Classification Probability"],
    index=0
)

model_choice = None
if prediction_mode == "Classification Probability":
    model_choice = st.selectbox(
        "Select Classification Model",
        ["XGBoost Classifier", "Random Forest", "Logistic Regression"]
    )

# =================================================
# FEATURE VECTOR BUILDER
# =================================================
def build_feature_vector():
    row = {}

    row["runtimeMinutes"] = runtime
    row["numVotes"] = votes
    row["averageRating"] = rating

    for g in GENRES:
        row[g] = 1 if g in genres else 0

    lead_cast = min(num_lead_cast, num_cast)
    crew_size = max(num_cast + num_directors + num_writers, 1)

    row["num_cast"] = num_cast
    row["num_lead_cast"] = lead_cast
    row["num_directors"] = num_directors
    row["num_writers"] = num_writers
    row["crew_size"] = crew_size
    row["cast_to_crew_ratio"] = num_cast / crew_size

    row["num_languages"] = num_languages
    row["num_regions"] = num_regions
    row["is_franchise"] = int(is_franchise)

    X = np.array([row.get(f, 0) for f in FEATURES]).reshape(1, -1)
    return X

# =================================================
# PREDICT
# =================================================
if st.button("🚀 Predict"):

    # ---------- ALWAYS COMPUTE ----------
    X = build_feature_vector()
    smooth_score = float(reg_model.predict(X)[0])

    st.subheader("📊 Prediction Result")

    # ---------- SMOOTH MODE ----------
    if prediction_mode == "Smooth Success Score (Recommended)":
        st.metric("Smooth Success Strength", f"{smooth_score:.3f}")
        st.progress(int(max(0, min(1, smooth_score)) * 100))

        st.caption("Stable, continuous success-strength signal.")

        # ---------- SHAP ----------
        st.subheader("🧠 Why this prediction? (Model-driven)")
        shap_values = explainer.shap_values(X)[0]

        shap_df = pd.DataFrame({
            "feature": FEATURES,
            "impact": shap_values
        }).assign(abs=lambda d: d["impact"].abs()) \
          .sort_values("abs", ascending=False)

        helped = shap_df[shap_df["impact"] > 0].head(4)
        hurt = shap_df[shap_df["impact"] < 0].head(4)

        def label(f):
            return f.replace("_", " ").title()

        if not helped.empty:
            st.markdown("**✅ What helped:**")
            for f in helped["feature"]:
                st.markdown(f"- {label(f)}")

        if not hurt.empty:
            st.markdown("**⚠️ What hurt:**")
            for f in hurt["feature"]:
                st.markdown(f"- {label(f)}")

    # ---------- CLASSIFICATION MODE ----------
    else:
        if model_choice == "XGBoost Classifier":
            prob = float(xgb_clf.predict_proba(X)[0][1])
        elif model_choice == "Random Forest":
            prob = float(rf_clf.predict_proba(X)[0][1])
        else:
            prob = float(log_clf.predict_proba(X)[0][1])

        st.metric(f"{model_choice} Probability", f"{prob:.2%}")
        st.caption("Classifier probability (may snap near boundary).")

    # =================================================
    # TRUST & CONFIDENCE
    # =================================================
    st.subheader("🛡️ Trust & Confidence")

    probs = [
        float(xgb_clf.predict_proba(X)[0][1]),
        float(rf_clf.predict_proba(X)[0][1]),
        float(log_clf.predict_proba(X)[0][1]),
    ]

    spread = max(probs) - min(probs)
    warnings = []

    if spread < 0.20:
        st.success("✅ Overall Confidence: HIGH")
    elif spread < 0.35:
        st.info("🟡 Overall Confidence: MEDIUM")
    else:
        st.warning("🔴 Overall Confidence: LOW")
        warnings.append("Strong disagreement between models.")

    if 0.45 <= smooth_score <= 0.55:
        warnings.append("Prediction is near decision boundary.")

    if votes < 2000:
        warnings.append("Limited audience votes.")

    if num_regions <= 1:
        warnings.append("Limited regional release.")

    if warnings:
        st.markdown("**⚠️ Notes:**")
        for w in warnings:
            st.markdown(f"- {w}")
