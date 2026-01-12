import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd

# ===============================
# LOAD MODELS & SCHEMA
# ===============================
with open("models/feature_schema.json") as f:
    FEATURES = json.load(f)

clf = joblib.load("models/xgb_classifier.pkl")
reg = joblib.load("models/xgb_regressor.pkl")

GENRES = [
    "Action", "Comedy", "Drama", "Romance",
    "Thriller", "Horror", "Adventure",
    "Crime", "Sci-Fi", "Fantasy"
]

# ===============================
# PAGE TITLE
# ===============================
st.title("🎯 Movie Success Prediction")

st.markdown(
    "Enter movie attributes below. Results update **in real time**."
)

# ===============================
# INPUTS (MAIN SCREEN, NOT SIDEBAR)
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    runtime = st.number_input("Runtime (minutes)", 60, 240, 120)
    votes = st.number_input("Expected Votes", 0, 1_000_000, 1000, step=100)

with col2:
    rating = st.slider("Expected IMDb Rating", 0.0, 10.0, 7.0, 0.1)
    languages = st.number_input("Number of Languages", 1, 20, 1)

with col3:
    regions = st.number_input("Number of Regions", 1, 50, 1)
    franchise = st.checkbox("Part of a Franchise")

selected_genres = st.multiselect(
    "Genres",
    GENRES
)

# ===============================
# BUILD INPUT VECTOR
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
    row[g] = int(g in selected_genres)

input_df = pd.DataFrame([row])[FEATURES]

# ===============================
# PREDICTIONS (REAL TIME)
# ===============================
prob = clf.predict_proba(input_df)[0][1]
score = reg.predict(input_df)[0]

# ===============================
# OUTPUT
# ===============================
st.subheader("📊 Prediction Results")

st.metric("Success Probability", f"{prob:.1%}")
st.metric("Smooth Success Score", f"{score:.2f}")

# ===============================
# REAL-TIME INSIGHT (NO THRESHOLDS)
# ===============================
st.subheader("🧠 What this movie has / lacks")

signals = []

if rating < 6.5:
    signals.append("❌ Rating expectation is weak")
else:
    signals.append("✅ Strong rating expectation")

if votes < 5000:
    signals.append("❌ Low audience reach")
else:
    signals.append("✅ Good audience reach")

if not franchise:
    signals.append("❌ No franchise boost")
else:
    signals.append("✅ Franchise advantage")

if languages == 1:
    signals.append("❌ Limited international exposure")
else:
    signals.append("✅ International reach potential")

for s in signals:
    st.write(s)
