import streamlit as st
import json
import joblib
import numpy as np

st.set_page_config(page_title="What-If Simulation", layout="centered")
st.title("🧪 What-If Simulation (Cliff-Free)")

@st.cache_resource
def load_artifacts():
    with open("models/feature_schema.json") as f:
        features = json.load(f)

    reg = joblib.load("models/xgb_regressor.pkl")
    return features, reg

FEATURES, reg_model = load_artifacts()

GENRES = [
    "Action","Comedy","Drama","Romance","Thriller",
    "Horror","Adventure","Crime","Sci-Fi","Fantasy"
]

# ---------------- BASELINE ----------------
st.subheader("Baseline Movie")

rating = st.slider("Rating", 1.0, 10.0, 6.9, 0.1)
votes = st.number_input("Votes", 0, value=3000)
runtime = st.number_input("Runtime", 40, 300, 120)

genres = st.multiselect("Genres", GENRES)

num_cast = st.number_input("Cast", 0, value=8)
num_lead_cast = st.number_input("Lead Cast", 0, value=3)
num_directors = st.number_input("Directors", 0, value=1)
num_writers = st.number_input("Writers", 0, value=2)

num_languages = st.number_input("Languages", 1, value=1)
num_regions = st.number_input("Regions", 1, value=1)

is_franchise = st.checkbox("Franchise")

# ---------------- FEATURE BUILDER ----------------
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

    return np.array([row.get(f, 0) for f in FEATURES]).reshape(1, -1)

# ---------------- WHAT-IF ----------------
baseline_score = reg_model.predict(build_vector())[0]

st.divider()
st.subheader("🔄 Change ONE parameter")

rating_delta = st.slider("Change Rating By", -1.0, 1.0, 0.0, 0.1)

new_rating = max(1.0, min(10.0, rating + rating_delta))
rating = new_rating  # override only here

new_score = reg_model.predict(build_vector())[0]
delta = new_score - baseline_score

# ---------------- OUTPUT ----------------
st.metric("Baseline Score", f"{baseline_score:.4f}")
st.metric("New Score", f"{new_score:.4f}")
st.metric("Δ Change (THIS IS WHAT MATTERS)", f"{delta:+.4f}")

st.caption(
    "No probability • No thresholds • Pure regressor delta\n"
    "This WILL NOT snap at 6.9 → 7.0"
)
