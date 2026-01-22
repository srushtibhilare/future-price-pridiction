from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_similarity(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_similar_reviews(
    review_unique_id,
    df,
    cosine_sim,
    top_n=5
):
    idx = df.index[df['review_unique_id'] == review_unique_id][0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]
    review_indices = [i[0] for i in sim_scores]

    return df.iloc[review_indices].copy()
