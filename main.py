import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------- LOAD DATA ----------------
basics = pd.read_csv("title_basics.tsv", sep="\t", low_memory=False)
ratings = pd.read_csv("title_ratings.tsv", sep="\t")

basics = basics[basics["titleType"] == "movie"]
df = basics.merge(ratings, on="tconst")

# ---------------- CLEAN DATA ----------------
df = df.replace("\\N", np.nan).infer_objects(copy=False)

df = df.dropna(subset=[
    "runtimeMinutes",
    "genres",
    "averageRating",
    "numVotes"
])

# Sample for speed
df = df.sample(n=200_000, random_state=42)

df["runtimeMinutes"] = df["runtimeMinutes"].astype(int)
df["numVotes"] = df["numVotes"].astype(int)

# ---------------- TARGET ----------------
df["success"] = (
    (df["averageRating"] >= 7) &
    (df["numVotes"] >= 1000)
).astype(int)

print("Class distribution:")
print(df["success"].value_counts())

# ---------------- GENRE FEATURES ----------------
numeric_features = ["runtimeMinutes", "numVotes", "averageRating"]

top_genres = [
    "Action", "Comedy", "Drama", "Romance",
    "Thriller", "Horror", "Adventure",
    "Crime", "Sci-Fi", "Fantasy"
]

for genre in top_genres:
    df[genre] = df["genres"].str.contains(genre, na=False).astype(int)

X = df[numeric_features + top_genres]
y = df["success"]

# =================================================
# 🔄 MODEL COMPARISON + CROSS-VALIDATION
# =================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )
}

print("\n🔍 Model comparison using 5-fold cross-validation:")
for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1"
    )
    print(f"{name} | Mean F1 score: {scores.mean():.3f}")

# =================================================
# ✅ FINAL MODEL (XGBoost → TRAIN & DEPLOY)
# =================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

final_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

final_model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = final_model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# ✅ TRAIN & SAVE ALL MODELS
# ===============================

# Logistic Regression (with scaling)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])
logistic_model.fit(X, y)
joblib.dump(logistic_model, "logistic_model.pkl")

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X, y)
joblib.dump(rf_model, "rf_model.pkl")

# XGBoost (final model)
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]),
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X, y)
joblib.dump(xgb_model, "xgb_model.pkl")

print("\n✅ All models saved successfully:")
print("• logistic_model.pkl")
print("• rf_model.pkl")
print("• xgb_model.pkl")
