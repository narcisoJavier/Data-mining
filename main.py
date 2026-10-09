"""
Main Execution Script for Midterm Summative Activity.
Performs end-to-end execution:
1. Data Ingestion & Preprocessing
2. Proximity Analysis (Cosine, Jaccard, Euclidean)
3. Topic Modeling (LDA, LSA, BERTopic) + Coherence Evaluation
4. Rule-Based Sentiment Analysis (90-10 Split + 10-Fold CV + Iterative Improvement)
"""

import sys
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from src.config import BASE_DIR, DATA_RAW, DATA_PROCESSED_DIR
from src.preprocess import run_preprocessing_pipeline
from src.similarity import analyze_proximity_measures
from src.topic_modeling import train_lda_model, train_lsa_model, train_bertopic_model
from src.sentiment_analysis import (
    split_dataset_90_10,
    run_10fold_cross_validation,
    evaluate_on_unseen_data
)

def main():
    print("=" * 80)
    print("       TAGALOG NLP MIDTERM SUMMATIVE PIPELINE")
    print("=" * 80)

    # 1. Locate Dataset
    custom_dataset_path = BASE_DIR / "data" / "raw" / "dataset.csv"
    if custom_dataset_path.exists():
        data_path = custom_dataset_path
        print(f"[*] Found user uploaded dataset at: {data_path}")
    else:
        data_path = DATA_RAW
        print(f"[*] Using sample dataset at: {data_path}")

    # Detect delimiter (comma or tab)
    try:
        df = pd.read_csv(data_path)
    except Exception:
        df = pd.read_csv(data_path, sep='\t')

    print(f"[*] Initial Loaded Records: {len(df)}")
    print(f"[*] Columns detected: {list(df.columns)}")

    # 2. Automated Preprocessing
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean = run_preprocessing_pipeline(df, text_col='comment_text')
    
    clean_output_path = DATA_PROCESSED_DIR / "preprocessed_dataset.csv"
    df_clean.to_csv(clean_output_path, index=False, encoding='utf-8')
    print(f"[OK] Saved preprocessed dataset to: {clean_output_path}\n")

    # 3. Proximity Measures
    print("=" * 80)
    print("       1. PROXIMITY & SIMILARITY / DISSIMILARITY ANALYSIS")
    print("=" * 80)
    prox_results = analyze_proximity_measures(df_clean, top_n=3)
    
    print("\n>>> Top Most Similar Comment Pairs (Cosine Similarity):")
    sim_display = prox_results['top_similar'][['author_1', 'author_2', 'cosine_similarity', 'jaccard_similarity', 'text_1', 'text_2']]
    print(tabulate(sim_display, headers='keys', tablefmt='grid', showindex=False))

    print("\n>>> Top Most Dissimilar Comment Pairs (Euclidean Distance):")
    dissim_display = prox_results['top_dissimilar'][['author_1', 'author_2', 'euclidean_distance', 'cosine_similarity', 'text_1', 'text_2']]
    print(tabulate(dissim_display, headers='keys', tablefmt='grid', showindex=False))

    # 4. Topic Modeling
    print("\n" + "=" * 80)
    print("       2. TOPIC MODELING & COHERENCE EVALUATION")
    print("=" * 80)
    print("[*] Training Latent Dirichlet Allocation (LDA) with 3 topics...")
    lda_res = train_lda_model(df_clean['tokens_for_lda'], num_topics=3)
    print(f"[OK] LDA Model Coherence Score (C_v): {lda_res['coherence_cv']:.4f}")
    print("\n>>> Extracted LDA Topics & Keywords:")
    print(tabulate(lda_res['topic_table'], headers='keys', tablefmt='grid'))

    print("\n[*] Training Latent Semantic Analysis (LSA) with 3 topics...")
    lsa_res = train_lsa_model(df_clean['tokens_for_lda'], num_topics=3)
    print(f"[OK] LSA Model Coherence Score (C_v): {lsa_res['coherence_cv']:.4f}")
    print("\n>>> Extracted LSA Topics & Keywords:")
    print(tabulate(lsa_res['topic_table'], headers='keys', tablefmt='grid'))

    print("\n[*] Training BERTopic with Multilingual Sentence Transformers...")
    bert_res = train_bertopic_model(df_clean['text_for_bert'].tolist(), num_topics=3)
    if bert_res['status'] == 'success':
        print("[OK] BERTopic successfully fitted.")
        print(tabulate(bert_res['topic_info'][['Topic', 'Count', 'Name']].head(5), headers='keys', tablefmt='grid'))
    else:
        print(f"[!] BERTopic note: {bert_res.get('message')}")

    # Topic Modeling Comparison Summary
    topic_comp = pd.DataFrame({
        'Model Algorithm': ['LDA (Latent Dirichlet Allocation)', 'LSA (Latent Semantic Analysis)'],
        'Number of Topics': [3, 3],
        'Coherence Score (C_v)': [f"{lda_res['coherence_cv']:.4f}", f"{lsa_res['coherence_cv']:.4f}"],
        'Remarks': ['Probabilistic generative model; distinct thematic clusters', 'Linear SVD dimensionality reduction; captures latent concepts']
    })
    print("\n>>> Topic Model Comparison Table:")
    print(tabulate(topic_comp, headers='keys', tablefmt='grid', showindex=False))

    # 5. Rule-Based Sentiment Analysis
    print("\n" + "=" * 80)
    print("       3. RULE-BASED SENTIMENT ANALYSIS & ITERATIVE IMPROVEMENT")
    print("=" * 80)
    
    # 90-10 Split
    dev_df, unseen_df = split_dataset_90_10(df_clean)
    print(f"[*] Dataset Split: 90% Development Set ({len(dev_df)} rows), 10% Unseen Validation Set ({len(unseen_df)} rows)")

    # 10-Fold Cross-Validation
    print("[*] Running 10-Fold Cross-Validation on Development Set...")
    cv_df, summary_df = run_10fold_cross_validation(dev_df)
    
    print("\n>>> 10-Fold Cross-Validation Performance Comparison (Per Fold):")
    print(tabulate(cv_df, headers='keys', tablefmt='grid', showindex=False))

    print("\n>>> Summary Performance: Baseline (V1) vs Improved Model (V2):")
    print(tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False))

    # Unseen Validation
    print("\n[*] Evaluating on 10% Unseen Validation Set...")
    unseen_preds, dist_df = evaluate_on_unseen_data(unseen_df)
    print("\n>>> Unseen Data Sentiment Distribution:")
    print(tabulate(dist_df, headers='keys', tablefmt='grid', showindex=False))

    print("\n" + "=" * 80)
    print("       PIPELINE EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
