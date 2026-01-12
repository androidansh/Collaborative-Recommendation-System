import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Program started")

# ---------------- LOAD DATA ----------------
finalDf = pd.read_csv("dataset.csv")

# ---------------- IMDB WEIGHTED RATING ----------------
movie_stats = finalDf.groupby(["movieId", "title"]).agg(
    v=("rating", "count"),
    R=("rating", "mean")
).reset_index()

C = movie_stats["R"].mean()
m = movie_stats["v"].quantile(0.90)

movie_stats["weighted_rating"] = (
    (movie_stats["v"] / (movie_stats["v"] + m)) * movie_stats["R"]
    + (m / (movie_stats["v"] + m)) * C
)

logging.info("IMDB weighted ratings computed")

# ---------------- SPARSE CF MATRIX ----------------
df = finalDf.drop(["title"], axis=1)

df["movie_index"] = df["movieId"].astype("category").cat.codes
df["user_index"] = df["userId"].astype("category").cat.codes

movie_user_sparse = csr_matrix(
    (df["rating"], (df["movie_index"], df["user_index"])),
    shape=(df["movie_index"].nunique(), df["user_index"].nunique())
)

movieId_to_index = dict(zip(df["movieId"], df["movie_index"]))
index_to_movieId = dict(zip(df["movie_index"], df["movieId"]))

logging.info("Collaborative filtering matrix ready")

# ---------------- GENRE TF-IDF MATRIX ----------------
movie_genres = finalDf[["movieId", "genres"]].drop_duplicates().reset_index(drop=True)

tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z\-]+")
genre_matrix = tfidf.fit_transform(movie_genres["genres"])

genre_id_to_row = dict(zip(movie_genres["movieId"], movie_genres.index))

logging.info("Genre TF-IDF matrix created")

# ---------------- CF SIMILARITY ----------------
def get_cf_similar_movies(movie_id, top_k=80):
    if movie_id not in movieId_to_index:
        return []

    idx = movieId_to_index[movie_id]
    vec = movie_user_sparse[idx]

    if vec.nnz == 0:
        return []

    sim = cosine_similarity(vec, movie_user_sparse).flatten()
    top_idx = sim.argsort()[-top_k-1:-1][::-1]

    return [(index_to_movieId[i], float(sim[i])) for i in top_idx]

# ---------------- GENRE SIMILARITY ----------------
def get_genre_similar_movies(movie_id, top_k=80):
    if movie_id not in genre_id_to_row:
        return []

    row = genre_id_to_row[movie_id]
    sim = cosine_similarity(genre_matrix[row], genre_matrix).flatten()

    top_idx = sim.argsort()[-top_k-1:-1][::-1]
    return [(movie_genres.iloc[i]["movieId"], float(sim[i])) for i in top_idx]

# ---------------- HYBRID RECOMMENDER ----------------
def recommend_genre_aware_hybrid(movie_ids, top_n=10,
                                 alpha=0.5, beta=0.3, gamma=0.2):

    logging.info("Inside genre-aware hybrid recommender")

    scores = {}

    for mid in movie_ids:

        # CF similarity
        for m, s in get_cf_similar_movies(mid):
            if m not in movie_ids:
                scores[m] = scores.get(m, 0) + alpha * s

        # Genre similarity
        for m, s in get_genre_similar_movies(mid):
            if m not in movie_ids:
                scores[m] = scores.get(m, 0) + beta * s

    if not scores:
        logging.warning("CF + Genre failed → IMDB fallback")
        return movie_stats.sort_values(
            by="weighted_rating", ascending=False
        )[["movieId", "title", "weighted_rating"]].head(top_n)

    # Convert to DataFrame
    score_df = pd.DataFrame(scores.items(), columns=["movieId", "hybrid_score"])

    # Merge IMDB confidence
    rec = movie_stats.merge(score_df, on="movieId")

    rec["final_score"] = rec["hybrid_score"] + gamma * rec["weighted_rating"]

    rec = rec.sort_values("final_score", ascending=False)

    # return rec[["movieId", "title", "weighted_rating", "final_score"]].head(top_n)
    return rec[["movieId"]].head(top_n)


# ---------------- RUN ----------------
ids = [217465, 109487, 134130]   # watched movies

titles = movie_stats[movie_stats["movieId"].isin(ids)]["title"].tolist()
print(f"\n🎬 Recommendations for movies: {titles}\n")

recommendations = recommend_genre_aware_hybrid(ids)
titles = movie_stats[movie_stats["movieId"].isin(recommendations)]["title"].tolist()
print(recommendations)
