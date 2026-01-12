# =================================================
# MOVIE SUCCESS PREDICTOR — TRAINING PIPELINE
# =================================================

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# =================================================
# CONFIG
# =================================================

DATA_DIR = "datasets"
MODEL_DIR = "models"
FEATURE_STORE_DIR = "feature_store"

RANDOM_STATE = 42
SAMPLE_SIZE = 240_000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)

# =================================================
# 1️⃣ LOAD DATASETS (ONCE)
# =================================================

print("📦 Loading IMDb datasets...")

basics = pd.read_csv(f"{DATA_DIR}/title_basics.tsv", sep="\t", low_memory=False)
ratings = pd.read_csv(f"{DATA_DIR}/title_ratings.tsv", sep="\t")
principals = pd.read_csv(f"{DATA_DIR}/title_principals.tsv", sep="\t", low_memory=False)

akas_path = f"{DATA_DIR}/title_akas.tsv"
akas = pd.read_csv(akas_path, sep="\t", low_memory=False) if os.path.exists(akas_path) else None

# =================================================
# 2️⃣ BASE FILTERING
# =================================================

basics = basics[basics["titleType"] == "movie"]
df = basics.merge(ratings, on="tconst", how="inner")

# =================================================
# 3️⃣ TALENT FEATURES
# =================================================

principals = principals[principals["tconst"].isin(df["tconst"])]
principals = principals[
    principals["category"].isin(["actor", "actress", "director", "writer"])
]

talent = principals.groupby("tconst").agg(
    num_cast=("category", lambda x: ((x == "actor") | (x == "actress")).sum()),
    num_lead_cast=("ordering", lambda x: (x <= 3).sum()),
    num_directors=("category", lambda x: (x == "director").sum()),
    num_writers=("category", lambda x: (x == "writer").sum()),
    crew_size=("category", "count"),
).reset_index()

talent["cast_to_crew_ratio"] = (
    talent["num_cast"] / talent["crew_size"].replace(0, np.nan)
)

df = df.merge(talent, on="tconst", how="left")

talent_cols = [
    "num_cast",
    "num_lead_cast",
    "num_directors",
    "num_writers",
    "crew_size",
    "cast_to_crew_ratio",
]

df[talent_cols] = df[talent_cols].fillna(0)

# =================================================
# 4️⃣ LANGUAGE & REGION FEATURES
# =================================================

if akas is not None:
    if "titleId" in akas.columns:
        akas = akas.rename(columns={"titleId": "tconst"})

    akas = akas[akas["tconst"].isin(df["tconst"])]

    lang_region = akas.groupby("tconst").agg(
        num_languages=("language", "nunique"),
        num_regions=("region", "nunique"),
    ).reset_index()

    df = df.merge(lang_region, on="tconst", how="left")
else:
    df["num_languages"] = 1
    df["num_regions"] = 1

df[["num_languages", "num_regions"]] = df[
    ["num_languages", "num_regions"]
].fillna(1)

# =================================================
# 5️⃣ FRANCHISE SIGNAL
# =================================================

df["is_franchise"] = df["originalTitle"].duplicated().astype(int)

# =================================================
# 6️⃣ CLEANING & SAMPLING
# =================================================

df = df.replace("\\N", np.nan)

df = df.dropna(
    subset=["runtimeMinutes", "genres", "averageRating", "numVotes"]
)

df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

df["runtimeMinutes"] = df["runtimeMinutes"].astype(int)
df["numVotes"] = df["numVotes"].astype(int)

# =================================================
# 7️⃣ TARGETS
# =================================================

df["success"] = (
    (df["averageRating"] >= 7.0) &
    (df["numVotes"] >= 1000)
).astype(int)

rating_norm = df["averageRating"] / 10.0
votes_norm = np.log1p(df["numVotes"]) / np.log1p(df["numVotes"].max())

df["success_score"] = 0.7 * rating_norm + 0.3 * votes_norm

print("\n🎯 Class distribution:")
print(df["success"].value_counts(normalize=True))

# =================================================
# 8️⃣ GENRE FEATURES
# =================================================

GENRE_FEATURES = [
    "Action", "Comedy", "Drama", "Romance",
    "Thriller", "Horror", "Adventure",
    "Crime", "Sci-Fi", "Fantasy",
]

for g in GENRE_FEATURES:
    df[g] = df["genres"].str.contains(g, na=False).astype(int)

# =================================================
# 9️⃣ FEATURE SCHEMA (SOURCE OF TRUTH)
# =================================================

FEATURE_COLUMNS = (
    ["runtimeMinutes", "numVotes", "averageRating"]
    + GENRE_FEATURES
    + talent_cols
    + ["num_languages", "num_regions", "is_franchise"]
)

X = df[FEATURE_COLUMNS]
y_class = df["success"]
y_reg = df["success_score"]

with open(f"{MODEL_DIR}/feature_schema.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f, indent=2)

df.to_parquet(
    f"{FEATURE_STORE_DIR}/movies_features.parquet",
    index=False
)

print(f"\n📊 Total features: {len(FEATURE_COLUMNS)}")

# =================================================
# 🔟 MODEL DEFINITIONS
# =================================================

models = {
    "logistic_model": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced"
        )),
    ]),
    "rf_model": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "xgb_classifier": XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=y_class.value_counts()[0] / y_class.value_counts()[1],
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    ),
}

xgb_regressor = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=RANDOM_STATE,
)

# =================================================
# 1️⃣1️⃣ CROSS-VALIDATION (CLASSIFIER)
# =================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n🔄 Cross-validation (F1 score):")
for name, model in models.items():
    scores = cross_val_score(model, X, y_class, cv=cv, scoring="f1")
    print(f"{name}: F1 = {scores.mean():.3f}")

# =================================================
# 1️⃣2️⃣ TRAIN & SAVE MODELS
# =================================================

print("\n🚀 Training final models...")

for name, model in models.items():
    model.fit(X, y_class)
    joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
    print(f"✅ Saved {name}.pkl")

xgb_regressor.fit(X, y_reg)
joblib.dump(xgb_regressor, f"{MODEL_DIR}/xgb_regressor.pkl")
print("✅ Saved xgb_regressor.pkl")

# =================================================
# 1️⃣3️⃣ HOLDOUT EVALUATION (XGB CLASSIFIER)
# =================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.2,
    stratify=y_class,
    random_state=RANDOM_STATE,
)

clf = models["xgb_classifier"]
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n📈 XGBoost Classifier Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("\n🏁 TRAINING COMPLETE")
print("• xgb_classifier.pkl → decision")
print("• xgb_regressor.pkl → smooth success")
print("• feature_schema.json → inference contract")
print("• movies_features.parquet → cached feature store")
