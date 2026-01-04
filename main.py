import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ---------------- LOAD DATA ----------------
basics = pd.read_csv("title_basics.tsv", sep="\t", low_memory=False)
ratings = pd.read_csv("title_ratings.tsv", sep="\t")

basics = basics[basics["titleType"] == "movie"]
df = basics.merge(ratings, on="tconst")

# ---------------- CLEAN DATA ----------------
df = df.replace("\\N", np.nan)

df = df.dropna(subset=[
    "runtimeMinutes",
    "genres",
    "averageRating",
    "numVotes"
])

# sample for speed
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

# ---------------- TRAIN / TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- XGBOOST MODEL ----------------
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

model = XGBClassifier(
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

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "movie_success_model.pkl")
print("\n✅ XGBoost model saved as movie_success_model.pkl")
