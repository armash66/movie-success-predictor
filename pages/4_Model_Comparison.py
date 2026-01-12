import json
import joblib
import pandas as pd
import streamlit as st
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="Model Comparison",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Model Comparison Dashboard")
st.caption("Compare multiple ML models on the same feature space")

# =================================================
# LOAD MODELS & FEATURES
# =================================================
@st.cache_resource
def load_assets():
    models = {
        "Logistic Regression": joblib.load("models/logistic_model.pkl"),
        "Random Forest": joblib.load("models/rf_model.pkl"),
        "XGBoost": joblib.load("models/xgb_model.pkl")
    }

    with open("models/feature_schema.json") as f:
        features = json.load(f)

    df = pd.read_parquet("feature_store/movies_features.parquet")

    return models, features, df

models, FEATURE_COLUMNS, df = load_assets()

X = df[FEATURE_COLUMNS]
y = df["success"]

# =================================================
# SAMPLE DATA (FAST + FAIR)
# =================================================
df_sample = df.sample(n=8000, random_state=42)
X_s = df_sample[FEATURE_COLUMNS]
y_s = df_sample["success"]

# =================================================
# METRIC COMPUTATION
# =================================================
def evaluate_model(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    return {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1 Score": f1_score(y, preds),
        "ROC-AUC": roc_auc_score(y, probs)
    }

results = []

for name, model in models.items():
    metrics = evaluate_model(model, X_s, y_s)
    metrics["Model"] = name
    results.append(metrics)

results_df = pd.DataFrame(results).set_index("Model")

# =================================================
# DISPLAY RESULTS
# =================================================
st.subheader("📊 Performance Comparison")

st.dataframe(
    results_df.style.format("{:.3f}")
)

st.bar_chart(results_df[["F1 Score", "ROC-AUC"]])

# =================================================
# INTERPRETATION
# =================================================
st.markdown("---")
st.subheader("🧠 How to Interpret This")

st.markdown("""
**Logistic Regression**
- Simple & interpretable
- Baseline reference
- Struggles with non-linear patterns

**Random Forest**
- Captures interactions
- More stable than single trees
- Can overfit on noisy signals

**XGBoost**
- Best balance of bias & variance
- Handles non-linearity + imbalance
- Strongest F1 & ROC-AUC

👉 **This is why XGBoost is used as the primary deployment model.**
""")

# =================================================
# MODEL AGREEMENT ANALYSIS
# =================================================
st.markdown("---")
st.subheader("🤝 Model Agreement Analysis")

preds = {
    name: model.predict(X_s)
    for name, model in models.items()
}

agreement_df = pd.DataFrame(preds)
agreement_df["True"] = y_s.values

agreement_rate = (
    (agreement_df.iloc[:, :-1].nunique(axis=1) == 1).mean()
)

st.metric(
    "Model Agreement Rate",
    f"{agreement_rate:.2%}"
)

st.info(
    "Higher agreement = more confident predictions.\n"
    "Disagreements highlight risky or ambiguous cases."
)
