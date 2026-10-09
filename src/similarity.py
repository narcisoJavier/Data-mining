"""
Proximity and Similarity / Dissimilarity Analysis Module.
Calculates Cosine Similarity, Jaccard Similarity, and Euclidean Distance
across social media comments and extracts the most similar and dissimilar pairs.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def compute_jaccard_similarity_matrix(token_series: pd.Series) -> np.ndarray:
    """Computes pairwise Jaccard similarity across token lists: |A ∩ B| / |A ∪ B|."""
    token_sets = [set(t) for t in token_series]
    n = len(token_sets)
    jaccard_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            set_i, set_j = token_sets[i], token_sets[j]
            union = set_i.union(set_j)
            if not union:
                sim = 0.0
            else:
                sim = len(set_i.intersection(set_j)) / len(union)
            jaccard_mat[i, j] = sim
            jaccard_mat[j, i] = sim
    return jaccard_mat

def analyze_proximity_measures(df: pd.DataFrame, text_col: str = 'text_for_lda', token_col: str = 'tokens_for_lda', top_n: int = 5):
    """
    Computes Cosine Similarity, Euclidean Distance, and Jaccard Similarity matrices.
    Returns matrices and formatted DataFrames showing the top similar and dissimilar pairs.
    """
    # 1. TF-IDF Representation
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df[text_col])

    # 2. Distance and Similarity Matrices
    cosine_sim_mat = cosine_similarity(tfidf_matrix)
    euclidean_dist_mat = euclidean_distances(tfidf_matrix)
    jaccard_sim_mat = compute_jaccard_similarity_matrix(df[token_col])

    # 3. Find top similar and dissimilar pairs (excluding self-pairs)
    n = len(df)
    results = []
    
    for i in range(n):
        for j in range(i + 1, n):
            results.append({
                'index_1': i,
                'index_2': j,
                'author_1': df.iloc[i].get('username', f'User_{i}'),
                'author_2': df.iloc[j].get('username', f'User_{j}'),
                'text_1': df.iloc[i]['comment_text'],
                'text_2': df.iloc[j]['comment_text'],
                'cosine_similarity': float(cosine_sim_mat[i, j]),
                'jaccard_similarity': float(jaccard_sim_mat[i, j]),
                'euclidean_distance': float(euclidean_dist_mat[i, j])
            })

    pairs_df = pd.DataFrame(results)

    # Top similar by Cosine
    top_similar = pairs_df.sort_values(by='cosine_similarity', ascending=False).head(top_n).copy()
    # Top dissimilar by Euclidean Distance
    top_dissimilar = pairs_df.sort_values(by='euclidean_distance', ascending=False).head(top_n).copy()

    return {
        'tfidf_matrix': tfidf_matrix,
        'cosine_sim_mat': cosine_sim_mat,
        'euclidean_dist_mat': euclidean_dist_mat,
        'jaccard_sim_mat': jaccard_sim_mat,
        'top_similar': top_similar,
        'top_dissimilar': top_dissimilar,
        'pairs_df': pairs_df
    }
