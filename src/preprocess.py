"""
Automated Multi-Stage Preprocessing Pipeline for Tagalog/Taglish Social Media Text.
Generates dedicated columns tailored for:
1. BERTopic (preserves syntax, grammar, casing)
2. Latent Dirichlet Allocation (LDA) / Latent Semantic Analysis (LSA) (tokens, bigrams, stopword-free)
3. Rule-Based Sentiment Analysis (negation markers, slang normalized, demojized cues)
"""

import re
import string
import pandas as pd
import emoji
from gensim.models.phrases import Phrases, Phraser
from src.config import (
    ALL_STOPWORDS,
    TAGALOG_SLANG_DICT
)

def clean_social_noise(text: str) -> str:
    """Removes URLs, mentions, HTML entities, and excessive whitespaces."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode quotes and dashes
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    text = text.replace('—', '-').replace('–', '-')
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove HTML entities like &amp; &lt; &#39;
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    # Normalize excessive character repetitions (e.g., 'soooobrang' -> 'soobrang')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text: str) -> str:
    """Replaces internet abbreviations and slang with standardized forms."""
    words = text.split()
    normalized = [TAGALOG_SLANG_DICT.get(w.lower(), w) for w in words]
    return " ".join(normalized)

def preprocess_for_bertopic(text: str) -> str:
    """Preprocesses text for Transformer embeddings while keeping sentence syntax intact."""
    cleaned = clean_social_noise(text)
    # Strip emojis to avoid model tokenizer artifact warnings
    cleaned = emoji.replace_emoji(cleaned, replace='')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def preprocess_for_sentiment(text: str) -> str:
    """Preprocesses text for rule-based sentiment: converts emojis to tokens and normalizes slang."""
    cleaned = clean_social_noise(text)
    cleaned = normalize_slang(cleaned)
    # Demojize emojis to descriptive word tokens (e.g. 😡 -> :enraged_face: -> enraged_face)
    demojized = emoji.demojize(cleaned, delimiters=(" ", " "))
    demojized = re.sub(r'[:_]', ' ', demojized)
    return re.sub(r'\s+', ' ', demojized).strip()

def preprocess_for_lda_tokens(text: str) -> list:
    """Lowercased, punctuation-free, and stopword-free tokenization for LDA/LSA/Proximity."""
    cleaned = clean_social_noise(text)
    cleaned = normalize_slang(cleaned)
    cleaned = emoji.replace_emoji(cleaned, replace='')
    # Remove punctuations and digits
    cleaned = re.sub(f'[{re.escape(string.punctuation)}0-9]', ' ', cleaned.lower())
    # Tokenize words longer than 2 characters
    tokens = [w for w in cleaned.split() if len(w) > 2]
    # Remove combined Tagalog + English + domain stopwords
    tokens = [w for w in tokens if w not in ALL_STOPWORDS]
    return tokens

def run_preprocessing_pipeline(df: pd.DataFrame, text_col: str = 'comment_text') -> pd.DataFrame:
    """
    Executes the full automated preprocessing pipeline on the input DataFrame.
    Returns DataFrame with new dedicated columns:
      - text_for_bert
      - text_for_sentiment
      - tokens_for_lda
      - text_for_lda
    """
    print("--- [Step 1/4] Deduplication and Noise Filtering ---")
    initial_count = len(df)
    # Deduplicate based on text column
    df = df.drop_duplicates(subset=[text_col]).copy()
    
    # Filter out empty or ultra-short comments (e.g. comments consisting only of an URL or mention)
    df['temp_clean'] = df[text_col].apply(clean_social_noise)
    df = df[df['temp_clean'].str.strip().str.len() > 3].copy()
    df.drop(columns=['temp_clean'], inplace=True)
    print(f"Dataset filtered from {initial_count} to {len(df)} non-empty unique records.")

    print("--- [Step 2/4] Generating Contextual Sentences for BERTopic ---")
    df['text_for_bert'] = df[text_col].apply(preprocess_for_bertopic)

    print("--- [Step 3/4] Generating Emotion & Negation Cues for Sentiment Analysis ---")
    df['text_for_sentiment'] = df[text_col].apply(preprocess_for_sentiment)

    print("--- [Step 4/4] Generating BoW Tokens & Collocation Phrases for LDA/LSA ---")
    raw_tokens = df[text_col].apply(preprocess_for_lda_tokens).tolist()
    
    # Build Bigrams / Phrases (e.g., 'confidential_funds', 'justice_system', 'sara_duterte')
    phrases = Phrases(raw_tokens, min_count=2, threshold=5)
    bigram_phraser = Phraser(phrases)
    df['tokens_for_lda'] = [bigram_phraser[t] for t in raw_tokens]
    df['text_for_lda'] = df['tokens_for_lda'].apply(lambda x: " ".join(x))

    print("Preprocessing completed successfully.")
    return df
