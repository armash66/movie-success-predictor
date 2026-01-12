import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "datasets"
OUT_DIR = "feature_store"
os.makedirs(OUT_DIR, exist_ok=True)

print("🚀 BUILDING IMDb FEATURE STORE (ALL 7 DATASETS)")
print("⏳ This will take time — run ONCE only\n")

# =================================================
# 1️⃣ LOAD CORE DATA
# =================================================
basics = pd.read_csv(f"{DATA_DIR}/title_basics.tsv", sep="\t", low_memory=False)
ratings = pd.read_csv(f"{DATA_DIR}/title_ratings.tsv", sep="\t")

basics = basics[basics["titleType"] == "movie"]
df = basics.merge(ratings, on="tconst")

print("✔ Loaded basics + ratings")

# =================================================
# 2️⃣ CAST & CREW (title.principals)
# =================================================
principals = pd.read_csv(f"{DATA_DIR}/title_principals.tsv", sep="\t", low_memory=False)
principals = principals[principals["tconst"].isin(df["tconst"])]

roles = ["actor", "actress", "director", "writer", "producer"]
principals = principals[principals["category"].isin(roles)]

talent = principals.groupby("tconst").agg(
    num_cast=("category", lambda x: sum(x.isin(["actor", "actress"]))),
    num_directors=("category", lambda x: sum(x == "director")),
    num_writers=("category", lambda x: sum(x == "writer")),
    crew_size=("category", "count"),
).reset_index()

df = df.merge(talent, on="tconst", how="left").fillna(0)
print("✔ Talent size features added")

# =================================================
# 3️⃣ TALENT EXPERIENCE (name.basics)
# =================================================
names = pd.read_csv(f"{DATA_DIR}/name_basics.tsv", sep="\t", low_memory=False)
names = names.replace("\\N", np.nan)

names["num_known_titles"] = names["knownForTitles"].apply(
    lambda x: len(x.split(",")) if isinstance(x, str) else 0
)

names["is_veteran"] = names["birthYear"].apply(
    lambda x: 1 if pd.notnull(x) and int(x) <= 1985 else 0
)

person_feats = names[["nconst", "num_known_titles", "is_veteran"]]

principals = principals.merge(person_feats, on="nconst", how="left").fillna(0)

exp_feats = principals.groupby("tconst").agg(
    avg_known_titles=("num_known_titles", "mean"),
    max_known_titles=("num_known_titles", "max"),
    cast_experience=("num_known_titles", "sum"),
    num_veterans=("is_veteran", "sum")
).reset_index()

df = df.merge(exp_feats, on="tconst", how="left").fillna(0)
print("✔ Talent experience features added")

# =================================================
# 4️⃣ CREATIVE STRUCTURE (title.crew)
# =================================================
crew = pd.read_csv(f"{DATA_DIR}/title_crew.tsv", sep="\t")
crew = crew.replace("\\N", np.nan)

crew["num_directors_crew"] = crew["directors"].apply(
    lambda x: len(x.split(",")) if isinstance(x, str) else 0
)

crew["num_writers_crew"] = crew["writers"].apply(
    lambda x: len(x.split(",")) if isinstance(x, str) else 0
)

df = df.merge(
    crew[["tconst", "num_directors_crew", "num_writers_crew"]],
    on="tconst",
    how="left"
).fillna(0)

print("✔ Creative structure features added")

# =================================================
# 5️⃣ MARKET REACH (title.akas)
# =================================================
akas = pd.read_csv(f"{DATA_DIR}/title_akas.tsv", sep="\t", low_memory=False)
akas = akas[akas["titleId"].isin(df["tconst"])]

market = akas.groupby("titleId").agg(
    num_regions=("region", "nunique"),
    num_languages=("language", "nunique")
).reset_index().rename(columns={"titleId": "tconst"})

df = df.merge(market, on="tconst", how="left").fillna(0)
print("✔ Market reach features added")

# =================================================
# 6️⃣ FRANCHISE POWER (title.episode)
# =================================================
episodes = pd.read_csv(f"{DATA_DIR}/title_episode.tsv", sep="\t")
episodes = episodes.replace("\\N", np.nan)

episodes["is_franchise"] = 1
df = df.merge(
    episodes[["tconst", "is_franchise"]],
    on="tconst",
    how="left"
).fillna({"is_franchise": 0})

print("✔ Franchise features added")

# =================================================
# 7️⃣ FINAL CLEANING + TARGET
# =================================================
df = df.replace("\\N", np.nan)

df = df.dropna(subset=["runtimeMinutes", "genres", "averageRating", "numVotes"])

df["runtimeMinutes"] = df["runtimeMinutes"].astype(int)
df["numVotes"] = df["numVotes"].astype(int)

df["success"] = (
    (df["averageRating"] >= 7) &
    (df["numVotes"] >= 1000)
).astype(int)

# =================================================
# 8️⃣ GENRE FEATURES
# =================================================
GENRES = [
    "Action","Comedy","Drama","Romance","Thriller",
    "Horror","Adventure","Crime","Sci-Fi","Fantasy"
]

for g in GENRES:
    df[g] = df["genres"].str.contains(g, na=False).astype(int)

# =================================================
# SAVE FEATURE STORE
# =================================================
out_path = f"{OUT_DIR}/movies_features.parquet"
df.to_parquet(out_path, index=False)

print("\n✅ FEATURE STORE BUILT SUCCESSFULLY")
print("📦 Rows:", len(df))
print("📁 Saved to:", out_path)
print("\n🚀 You will NEVER load raw IMDb TSVs again.")
