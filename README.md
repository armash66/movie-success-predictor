🎬 Movie Success Prediction System

Production-grade ML + XAI Platform using IMDb Data

A deployable, explainable machine learning system that predicts movie success using IMDb metadata, audience signals, and talent features — with smooth predictions, model switching, and trust awareness.

🚀 Project Overview

This project predicts whether a movie is likely to be successful and explains why — without unstable threshold behavior.

Instead of relying on a single binary model, the system separates:

Decision models (classification)

Reasoning models (regression)

Explanation models (XAI / SHAP)

Trust signals (model disagreement & data quality)

This architecture avoids common pitfalls like probability snapping and misleading explanations.

🧠 Key Design Principles
1️⃣ No Hard Prediction Jumps

Binary classifiers tend to jump sharply near thresholds (e.g., rating 6.9 → 7.0).

Solution:

Use a regression model for the main UI

Use classifiers only when explicitly requested

2️⃣ Separation of Concerns (Critical)

Each model has a clear role:

Model	Purpose
Logistic Regression	Interpretable baseline
Random Forest	Non-linear baseline
XGBoost Classifier	Binary decision model
XGBoost Regressor	Smooth success score (primary UX)

This prevents mixing incompatible outputs.

3️⃣ Explainability Without Lying

Explanations are:

Local (one prediction at a time)

Model-driven (SHAP)

Aligned with regression output

No global SHAP misuse. No heuristic explanations pretending to be model logic.

4️⃣ Trust Is Explicit

The system does not blindly trust one model.

Instead, it:

Compares multiple classifiers

Detects disagreement

Flags low-confidence predictions

Warns when data is insufficient

📊 Data Sources

IMDb datasets (TSV format):

title_basics.tsv

title_ratings.tsv

title_principals.tsv

title_akas.tsv

Processed into a single feature store (~240k movies).

🧬 Feature Engineering
Core Numeric

runtimeMinutes

numVotes

averageRating

Genres (One-Hot)

Action, Comedy, Drama, Romance, Thriller

Horror, Adventure, Crime, Sci-Fi, Fantasy

Talent & Crew

num_cast

num_lead_cast

num_directors

num_writers

crew_size

cast_to_crew_ratio

Market Reach

num_languages

num_regions

Business Signal

is_franchise

Feature schema is saved to:

models/feature_schema.json

🎯 Targets
Binary Target (Decision)
success = (averageRating ≥ 7) AND (numVotes ≥ 1000)


Used only for classification models.

Continuous Target (Smooth Reasoning)
success_score = 0.7 * normalized_rating + 0.3 * normalized_votes


Used for:

Prediction UI

What-If simulation

XAI explanations

🖥️ Application Architecture (Streamlit)
app.py
pages/
 ├── 1_Prediction.py
 ├── 2_What_If.py
 └── 3_XAI.py
models/
feature_store/
datasets/

app.py

Navigation only

No ML logic

🔮 Prediction Page

Smooth Success Score (default, cliff-free)

Classification mode with model switching

SHAP-based explanations

Trust & confidence warnings

🔁 What-If Simulation

Regressor-only

Real-time sliders

Delta-based impact

No thresholds, no jumps

🧠 XAI Page

SHAP TreeExplainer

Local explanation for one movie

Top features helping vs hurting

Full SHAP table (transparent)

🛡️ Trust & Confidence Layer

The system warns users when predictions are risky:

Model disagreement

Boundary proximity

Low audience votes

Limited regional or language reach

Confidence levels:

High

Medium

Low

This makes the system honest and production-ready.

⚙️ Model Switching (While Predicting)

Users can explicitly choose:

Prediction Type

Smooth Success Score (recommended)

Classification Probability

Classification Models

XGBoost

Random Forest

Logistic Regression

No silent switching. No averaging incompatible models.

🧪 How to Run
pip install -r requirements.txt
streamlit run app.py