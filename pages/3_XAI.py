import streamlit as st
import json
import joblib
import numpy as np
import shap
import pandas as pd

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(page_title="XAI", layout="centered")
st.title("🧠 Explainable AI (Why this score?)")

st.caption(
    "This page explains the **smooth success score** using SHAP.\n"
    "It does NOT explain probability or decision thresholds."
)

# =================================================
# LOAD ARTIFACTS (ONCE)
# =================================================
@st.cache_resource
def load_artifacts():
    with open("models/feature_schema.json") as f:
        features = json.load(f)

    reg = joblib.load("models/xgb_regressor.pkl")

    explainer = shap.TreeExplainer(reg)

    return features, reg, explainer

FEATURES, reg_model, explainer = load_artifacts()

GENRES = [
    "Action","Comedy","Drama","Romance","Thriller",
    "Horror","Adventure","Crime","Sci-Fi","Fantasy"
]

# =================================================
# INPUTS (SAME AS PREDICTION)
# =================================================
st.subheader("🎬 Movie Configuration")

runtime = st.number_input("Runtime (minutes)", 40, 300, 120)
rating = st.slider("Average Rating", 1.0, 10.0, 6.9, 0.1)
votes = st.number_input("Number of Votes", 0, value=3000)

genres = st.multiselect("Genres", GENRES)

num_cast = st.number_input("Cast", 0, value=8)
num_lead_cast = st.number_input("Lead Cast", 0, value=3)
num_directors = st.number_input("Directors", 0, value=1)
num_writers = st.number_input("Writers", 0, value=2)

num_languages = st.number_input("Languages", 1, value=1)
num_regions = st.number_input("Regions", 1, value=1)

is_franchise = st.checkbox("Franchise")

# =================================================
# FEATURE VECTOR (PURE, MATCHES TRAINING)
# =================================================
def build_vector():
    row = {}

    row["runtimeMinutes"] = runtime
    row["numVotes"] = votes
    row["averageRating"] = rating

    for g in GENRES:
        row[g] = 1 if g in genres else 0

    num_lead = min(num_lead_cast, num_cast)
    crew_size = max(num_cast + num_directors + num_writers, 1)

    row["num_cast"] = num_cast
    row["num_lead_cast"] = num_lead
    row["num_directors"] = num_directors
    row["num_writers"] = num_writers
    row["crew_size"] = crew_size
    row["cast_to_crew_ratio"] = num_cast / crew_size

    row["num_languages"] = num_languages
    row["num_regions"] = num_regions
    row["is_franchise"] = int(is_franchise)

    X = np.array([row.get(f, 0) for f in FEATURES]).reshape(1, -1)
    return X, row

# =================================================
# EXPLAIN
# =================================================
if st.button("Explain Prediction"):
    X, row_dict = build_vector()

    # Predict score
    score = reg_model.predict(X)[0]

    # SHAP values
    shap_values = explainer.shap_values(X)[0]

    shap_df = pd.DataFrame({
        "feature": FEATURES,
        "value": [row_dict.get(f, 0) for f in FEATURES],
        "impact": shap_values
    })

    shap_df["abs_impact"] = shap_df["impact"].abs()
    shap_df = shap_df.sort_values("abs_impact", ascending=False)

    # Split helpful vs harmful
    helped = shap_df[shap_df["impact"] > 0].head(5)
    hurt = shap_df[shap_df["impact"] < 0].head(5)

    st.subheader("📊 Explained Result")
    st.metric("Smooth Success Score", f"{score:.3f}")

    # ---------------- HELPED ----------------
    st.markdown("### ✅ Features that helped")
    if helped.empty:
        st.write("No strong positive contributors.")
    else:
        for _, r in helped.iterrows():
            st.write(
                f"- **{r['feature']}** increased the score "
                f"(impact: +{r['impact']:.3f})"
            )

    # ---------------- HURT ----------------
    st.markdown("### ⚠️ Features that hurt")
    if hurt.empty:
        st.write("No strong negative contributors.")
    else:
        for _, r in hurt.iterrows():
            st.write(
                f"- **{r['feature']}** decreased the score "
                f"(impact: {r['impact']:.3f})"
            )

    # ---------------- FULL TABLE ----------------
    with st.expander("🔍 Full SHAP details"):
        st.dataframe(
            shap_df[["feature", "value", "impact"]].reset_index(drop=True),
            use_container_width=True
        )

    st.caption(
        "Positive impact → increases success strength\n"
        "Negative impact → decreases success strength\n"
        "This explanation is local to THIS movie only."
    )
