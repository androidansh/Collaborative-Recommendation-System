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

def get_similar_movies(movie_id, top_k=20):
    movie_index = movieId_to_index[movie_id]
    movie_vec = movie_user_sparse[movie_index]
    sim_scores = cosine_similarity(movie_vec, movie_user_sparse).flatten()
    top_idx = sim_scores.argsort()[-top_k-1:-1][::-1]
    similar_movie_ids = [index_to_movieId[i] for i in top_idx]
    return similar_movie_ids

def recommend_for_cold_user(movie_id, top_n=15):
    similar_movies = get_similar_movies(movie_id)
    candidates = movie_stats[movie_stats['movieId'].isin(similar_movies)]
    # print(candidates) 'v','R',
    candidates = candidates.sort_values(by='weighted_rating', ascending=False)
    # print(f"Candid")
    return candidates[['movieId','title','weighted_rating']].head(top_n)

a = recommend_for_cold_user(103688)
print(a)

