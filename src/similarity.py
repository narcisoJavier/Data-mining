"""
Proximity and Similarity / Dissimilarity Analysis Module.
Vectorized calculation of Cosine Similarity, Jaccard Similarity, and Euclidean Distance
optimized for datasets with thousands of records.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def compute_pairwise_jaccard(tokens1: list, tokens2: list) -> float:
    """Computes Jaccard similarity between two token lists: |A ∩ B| / |A ∪ B|."""
    s1, s2 = set(tokens1), set(tokens2)
    union = s1.union(s2)
    if not union:
        return 0.0
    return len(s1.intersection(s2)) / len(union)

def analyze_proximity_measures(df: pd.DataFrame, text_col: str = 'text_for_lda', token_col: str = 'tokens_for_lda', top_n: int = 5, max_sample: int = 1000):
    """
    Computes vectorized Cosine Similarity, Euclidean Distance, and Jaccard Similarity.
    Efficiently extracts top similar and dissimilar pairs without combinatorial memory bottleneck.
    """
    # Sample up to max_sample records if dataset is large to maintain interactive performance
    if len(df) > max_sample:
        sample_df = df.sample(n=max_sample, random_state=42).reset_index(drop=True)
    else:
        sample_df = df.reset_index(drop=True)

    # 1. TF-IDF Representation
    sample_df[text_col] = sample_df[text_col].fillna('')
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(sample_df[text_col])

    # 2. Vectorized Cosine Similarity
    cosine_sim_mat = cosine_similarity(tfidf_matrix)

    # Upper triangle indices (excluding self-comparisons i == j)
    n = sample_df.shape[0]
    r_idx, c_idx = np.triu_indices(n, k=1)
    cos_values = cosine_sim_mat[r_idx, c_idx]

    # Find Top Similar Pairs (highest cosine similarity < 0.999 to filter exact duplicate reposts)
    non_exact = cos_values < 0.999
    valid_cos_values = np.where(non_exact, cos_values, -1.0)
    top_sim_positions = np.argsort(valid_cos_values)[-top_n:][::-1]

    top_similar_records = []
    for pos in top_sim_positions:
        i, j = r_idx[pos], c_idx[pos]
        tok_i = sample_df.iloc[i][token_col]
        tok_j = sample_df.iloc[j][token_col]
        jacc = compute_pairwise_jaccard(tok_i, tok_j)
        euc = float(np.linalg.norm(tfidf_matrix[i].toarray() - tfidf_matrix[j].toarray()))
        
        top_similar_records.append({
            'index_1': i,
            'index_2': j,
            'author_1': sample_df.iloc[i].get('username', f'User_{i}'),
            'author_2': sample_df.iloc[j].get('username', f'User_{j}'),
            'text_1': sample_df.iloc[i]['comment_text'],
            'text_2': sample_df.iloc[j]['comment_text'],
            'cosine_similarity': float(cos_values[pos]),
            'jaccard_similarity': float(jacc),
            'euclidean_distance': euc
        })

    # Find Top Dissimilar Pairs (lowest cosine similarity / highest euclidean distance)
    top_dissim_positions = np.argsort(cos_values)[:top_n]
    top_dissimilar_records = []
    for pos in top_dissim_positions:
        i, j = r_idx[pos], c_idx[pos]
        tok_i = sample_df.iloc[i][token_col]
        tok_j = sample_df.iloc[j][token_col]
        jacc = compute_pairwise_jaccard(tok_i, tok_j)
        euc = float(np.linalg.norm(tfidf_matrix[i].toarray() - tfidf_matrix[j].toarray()))
        
        top_dissimilar_records.append({
            'index_1': i,
            'index_2': j,
            'author_1': sample_df.iloc[i].get('username', f'User_{i}'),
            'author_2': sample_df.iloc[j].get('username', f'User_{j}'),
            'text_1': sample_df.iloc[i]['comment_text'],
            'text_2': sample_df.iloc[j]['comment_text'],
            'cosine_similarity': float(cos_values[pos]),
            'jaccard_similarity': float(jacc),
            'euclidean_distance': euc
        })

    top_similar = pd.DataFrame(top_similar_records)
    top_dissimilar = pd.DataFrame(top_dissimilar_records)

    return {
        'tfidf_matrix': tfidf_matrix,
        'cosine_sim_mat': cosine_sim_mat,
        'top_similar': top_similar,
        'top_dissimilar': top_dissimilar
    }
