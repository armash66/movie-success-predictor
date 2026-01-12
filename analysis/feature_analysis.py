import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ===============================
# LOAD FEATURE STORE
# ===============================
df = pd.read_parquet("feature_store/movies_features.parquet")

print("📦 Loaded rows:", df.shape[0])
print("📊 Total columns:", df.shape[1])

# ===============================
# DEFINE FEATURES SAFELY
# ===============================
DESIRED_FEATURES = [
    # Core
    "runtimeMinutes", "numVotes", "averageRating",

    # Genres
    "Action", "Comedy", "Drama", "Romance", "Thriller",
    "Horror", "Adventure", "Crime", "Sci-Fi", "Fantasy",

    # Talent size
    "num_cast", "num_directors", "num_writers",
    "crew_size", "cast_to_crew_ratio",

    # Talent experience
    "avg_known_titles", "max_known_titles",
    "cast_experience", "num_veterans",

    # Creative structure
    "num_directors_crew", "num_writers_crew",

    # Market reach
    "num_regions", "num_languages",

    # Franchise
    "is_franchise"
]

TARGET = "success"

# Keep only features that actually exist
FEATURE_COLUMNS = [c for c in DESIRED_FEATURES if c in df.columns]

# Log missing features (IMPORTANT)
missing = set(DESIRED_FEATURES) - set(FEATURE_COLUMNS)
if missing:
    print("\n⚠️ Missing features (skipped safely):")
    for m in missing:
        print(" -", m)

print("\n✅ Using", len(FEATURE_COLUMNS), "features for analysis")

X = df[FEATURE_COLUMNS]
y = df[TARGET]

# ===============================
# TRAIN XGBOOST (FAST)
# ===============================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model trained for feature analysis")

# ===============================
# XGBOOST FEATURE IMPORTANCE
# ===============================
xgb_imp = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n🔝 Top 15 XGBoost Importances:")
print(xgb_imp.head(15))

# ===============================
# PERMUTATION IMPORTANCE
# ===============================
print("\n🔄 Computing permutation importance...")

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    scoring="f1",
    random_state=42
)

perm_imp = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": perm.importances_mean
}).sort_values(by="importance", ascending=False)

print("\n🔝 Top 15 Permutation Importances:")
print(perm_imp.head(15))

# ===============================
# LOW SIGNAL FEATURES
# ===============================
low_signal = perm_imp[perm_imp["importance"] < 0.001]["feature"].tolist()

print("\n⚠️ Low-signal feature candidates:")
print(low_signal)

print("\n✅ Feature analysis completed successfully.")
