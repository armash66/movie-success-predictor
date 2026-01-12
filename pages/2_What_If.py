import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd

# ===============================
# LOAD MODEL & FEATURES
# ===============================
with open("models/feature_schema.json") as f:
    FEATURES = json.load(f)

reg = joblib.load("models/xgb_regressor.pkl")

GENRES = [
    "Action", "Comedy", "Drama", "Romance",
    "Thriller", "Horror", "Adventure",
    "Crime", "Sci-Fi", "Fantasy"
]

# ===============================
# PAGE TITLE
# ===============================
st.title("🔄 What-If Simulation (Real-Time)")

st.markdown("""
This page shows **how small changes affect success**.
No thresholds. No jumps. Just smooth intelligence.
""")

# ===============================
# BASE INPUT
# ===============================
st.subheader("🎬 Base Movie Setup")

col1, col2, col3 = st.columns(3)

with col1:
    runtime = st.slider("Runtime", 60, 240, 120)
    votes = st.slider("Votes", 0, 500_000, 2000, step=500)

with col2:
    rating = st.slider("Rating", 0.0, 10.0, 6.8, 0.1)
    languages = st.slider("Languages", 1, 10, 1)

with col3:
    regions = st.slider("Regions", 1, 20, 1)
    franchise = st.checkbox("Franchise")

genres = st.multiselect("Genres", GENRES)

# ===============================
# BUILD BASE VECTOR
# ===============================
def build_row(rating_delta=0, vote_delta=0):
    row = {
        "runtimeMinutes": runtime,
        "numVotes": votes + vote_delta,
        "averageRating": rating + rating_delta,
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

    return pd.DataFrame([row])[FEATURES]

# ===============================
# BASE SCORE
# ===============================
base_df = build_row()
base_score = reg.predict(base_df)[0]

st.metric("🎯 Base Success Score", f"{base_score:.2f}")

# ===============================
# WHAT-IF ANALYSIS
# ===============================
st.subheader("📈 Impact of Small Changes")

scenarios = {
    "Rating +0.3": build_row(rating_delta=0.3),
    "Rating +0.5": build_row(rating_delta=0.5),
    "Votes +5k": build_row(vote_delta=5000),
    "Votes +20k": build_row(vote_delta=20000),
}

results = []

for name, df_s in scenarios.items():
    new_score = reg.predict(df_s)[0]
    delta = new_score - base_score
    results.append([name, new_score, delta])

impact_df = pd.DataFrame(
    results,
    columns=["Scenario", "New Score", "Change"]
)

impact_df["Change"] = impact_df["Change"].apply(
    lambda x: f"{x:+.3f}"
)

st.dataframe(impact_df, use_container_width=True)

# ===============================
# INTERPRETATION
# ===============================
st.subheader("🧠 Interpretation")

if rating < 6.9:
    st.warning("📉 Rating is the biggest bottleneck right now.")
else:
    st.success("⭐ Rating is already helping the model.")

if votes < 5000:
    st.warning("👥 Audience reach is limiting growth.")
else:
    st.success("📣 Votes contribute positively.")

st.info("All changes above are **continuous**, not threshold-based.")
