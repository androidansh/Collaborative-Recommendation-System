import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Program started")

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

finalDf = pd.read_csv('final_df.csv')

movie_stats = finalDf.groupby(['movieId', 'title']).agg(
    v = ('rating','count'),
    R = ('rating','mean')
).reset_index()
C = movie_stats['R'].mean()
m = movie_stats['v'].quantile(0.90)
movie_stats['weighted_rating'] = ((movie_stats['v']/(movie_stats['v']+m)) * movie_stats['R']+ (m/(movie_stats['v']+m)) * C)

df = finalDf.drop(['title'],axis=1)
movie_cat = df['movieId'].astype('category')
user_cat  = df['userId'].astype('category')

df['movie_index'] = movie_cat.cat.codes
df['user_index']  = user_cat.cat.codes

movie_user_sparse = csr_matrix((df['rating'], (df['movie_index'], df['user_index'])), shape=(df['movie_index'].nunique(), df['user_index'].nunique()))

movieId_to_index = dict(zip(df['movieId'], df['movie_index']))
index_to_movieId = dict(zip(df['movie_index'], df['movieId']))

logging.info("Basic data requirements completed")
def get_similar_movies(movie_id, top_k=50):
    movie_index = movieId_to_index[movie_id]
    movie_vec = movie_user_sparse[movie_index]

    sim_scores = cosine_similarity(movie_vec, movie_user_sparse).flatten()

    # get top K (excluding itself)
    top_idx = sim_scores.argsort()[-top_k-1:-1][::-1]

    results = [(index_to_movieId[i], float(sim_scores[i])) for i in top_idx]
    return results
def recommend_for_cold_user_multi(movie_ids, top_n=10):
    logging.info("Inside the recommendation function")
    combined_scores = {}
    for mid in movie_ids:
        neighbors = get_similar_movies(mid, top_k=80)
        for m, score in neighbors:
            if m in movie_ids:
                continue      # don't recommend watched movies
            combined_scores[m] = combined_scores.get(m, 0) + score

    # if nothing found → fallback
    if not combined_scores:
        print("No CF neighbors found → IMDB fallback")
        return movie_stats.sort_values(by="weighted_rating", ascending=False)[['movieId','title','weighted_rating']].head(top_n)
    logging.info("finished similar movie search")
    # convert to sorted list
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_ids = [m for m, _ in ranked[:top_n]]
    # Convert to dataframe to merge scores + weighted rating
    score_df = pd.DataFrame(ranked[:top_n], columns=['movieId','cf_score'])
    rec = movie_stats[movie_stats['movieId'].isin(ranked_ids)].merge(score_df, on='movieId').sort_values(by='cf_score', ascending=False)
    return rec[['movieId','title','weighted_rating','cf_score']]


logging.info("Calling recommendation")
ids = [109487, 79132,134130]
a = recommend_for_cold_user_multi(ids)
logging.info("Output")
titles = movie_stats[movie_stats['movieId'].isin(ids)]['title'].tolist()
print(f'Movie Recommendation for movies: {titles}')
print(a)