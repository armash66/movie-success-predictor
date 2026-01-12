import streamlit as st
import json
import joblib
import shap
import numpy as np
import pandas as pd

# ===============================
# LOAD MODELS & FEATURES
# ===============================
with open("models/feature_schema.json") as f:
    FEATURES = json.load(f)

clf = joblib.load("models/xgb_classifier.pkl")

GENRES = [
    "Action", "Comedy", "Drama", "Romance",
    "Thriller", "Horror", "Adventure",
    "Crime", "Sci-Fi", "Fantasy"
]

# ===============================
# PAGE HEADER
# ===============================
st.title("🧠 Explainable AI (XAI)")
st.markdown("""
This page explains **why the model made a prediction**  
using **SHAP values** (industry-standard explainability).
""")

# ===============================
# INPUTS (SAME AS PREDICTION)
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    runtime = st.number_input("Runtime", 60, 240, 120)
    votes = st.number_input("Votes", 0, 1_000_000, 1000, step=100)

with col2:
    rating = st.slider("Rating", 0.0, 10.0, 7.0, 0.1)
    languages = st.number_input("Languages", 1, 20, 1)

with col3:
    regions = st.number_input("Regions", 1, 50, 1)
    franchise = st.checkbox("Franchise")

genres = st.multiselect("Genres", GENRES)

# ===============================
# BUILD INPUT
# ===============================
row = {
    "runtimeMinutes": runtime,
    "numVotes": votes,
    "averageRating": rating,
    "num_languages": languages,
    "num_regions": regions,
    "is_franchise": int(franchise),
    "num_cast": 10,
    "num_lead_cast": 3,
    "num_directors": 1,
    "num_writers": 1,
    "crew_size": 20,
    "cast_to_crew_ratio": 0.5
}

for g in GENRES:
    row[g] = int(g in genres)

X = pd.DataFrame([row])[FEATURES]

# ===============================
# PREDICTION
# ===============================
prob = clf.predict_proba(X)[0][1]

st.metric("🎯 Success Probability", f"{prob:.1%}")

# ===============================
# SHAP EXPLANATION
# ===============================
st.subheader("🔍 Feature Contributions")

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

shap_df = pd.DataFrame({
    "Feature": FEATURES,
    "Impact": shap_values[0]
}).sort_values(
    by="Impact",
    key=abs,
    ascending=False
)

# ===============================
# INTERPRETATION (HUMAN READABLE)
# ===============================
st.subheader("🧠 Why this prediction happened")

positive = shap_df[shap_df["Impact"] > 0].head(5)
negative = shap_df[shap_df["Impact"] < 0].head(5)

st.markdown("### ✅ Factors helping success")
for _, r in positive.iterrows():
    st.write(f"➕ **{r['Feature']}** → +{r['Impact']:.3f}")

st.markdown("### ❌ Factors hurting success")
for _, r in negative.iterrows():
    st.write(f"➖ **{r['Feature']}** → {r['Impact']:.3f}")

# ===============================
# TABLE + BAR VISUAL
# ===============================
st.subheader("📊 Full SHAP Breakdown")
st.dataframe(shap_df, use_container_width=True)

st.bar_chart(
    shap_df.set_index("Feature")["Impact"]
)

st.info("""
SHAP values show **direction + strength** of each feature.
Positive = helps success  
Negative = hurts success
""")
