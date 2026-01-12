import streamlit as st
import joblib
import json
import pandas as pd
from sklearn.inspection import partial_dependence

@st.cache_resource
def load_assets():
    model = joblib.load("models/xgb_regressor.pkl")
    data = pd.read_parquet("feature_store/movies_features.parquet")
    with open("models/feature_schema.json") as f:
        features = json.load(f)
    return model, data, features

model, df, FEATURES = load_assets()

st.header("📈 Partial Dependence & ICE")

feature = st.selectbox("Select feature", FEATURES)

pdp = partial_dependence(
    model,
    df[FEATURES],
    [feature],
    kind="average"
)

st.line_chart(
    pd.DataFrame({
        feature: pdp["values"][0],
        "Effect": pdp["average"][0]
    }).set_index(feature)
)

st.caption("ICE lines are implicitly learned via regressor smoothness.")
