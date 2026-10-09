"""
Topic Modeling Module: Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA),
and BERTopic with automated Topic Coherence (C_v) evaluation.
"""

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from src.config import ALL_STOPWORDS

def train_lda_model(tokens_series: pd.Series, num_topics: int = 3, random_state: int = 42):
    """
    Trains Gensim Latent Dirichlet Allocation (LDA) model and evaluates Coherence score (C_v).
    """
    # 1. Build Dictionary & Corpus
    dictionary = Dictionary(tokens_series)
    dictionary.filter_extremes(no_below=1, no_above=0.85)
    corpus = [dictionary.doc2bow(text) for text in tokens_series]

    # 2. Train LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=15,
        alpha='auto',
        per_word_topics=True
    )

    # 3. Compute Coherence Score (C_v)
    coherence_cv = CoherenceModel(
        model=lda,
        texts=tokens_series.tolist(),
        dictionary=dictionary,
        coherence='c_v'
    ).get_coherence()

    # 4. Extract Top Keywords
    topic_keywords = {}
    for topic_id in range(num_topics):
        words_weights = lda.show_topic(topic_id, topn=8)
        topic_keywords[f"Topic {topic_id + 1}"] = [f"{w} ({weight:.3f})" for w, weight in words_weights]

    df_topics = pd.DataFrame(topic_keywords)

    return {
        'model': lda,
        'dictionary': dictionary,
        'corpus': corpus,
        'coherence_cv': coherence_cv,
        'topic_table': df_topics
    }

def train_lsa_model(tokens_series: pd.Series, num_topics: int = 3, random_state: int = 42):
    """
    Trains Latent Semantic Analysis (LSA / LSI) model and calculates Coherence score (C_v).
    """
    dictionary = Dictionary(tokens_series)
    dictionary.filter_extremes(no_below=1, no_above=0.85)
    corpus = [dictionary.doc2bow(text) for text in tokens_series]

    lsi = LsiModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_seed=random_state
    )

    coherence_cv = CoherenceModel(
        model=lsi,
        texts=tokens_series.tolist(),
        dictionary=dictionary,
        coherence='c_v'
    ).get_coherence()

    topic_keywords = {}
    for topic_id in range(num_topics):
        words_weights = lsi.show_topic(topic_id, topn=8)
        topic_keywords[f"Topic {topic_id + 1}"] = [f"{w} ({weight:.3f})" for w, weight in words_weights]

    df_topics = pd.DataFrame(topic_keywords)

    return {
        'model': lsi,
        'dictionary': dictionary,
        'corpus': corpus,
        'coherence_cv': coherence_cv,
        'topic_table': df_topics
    }

def train_bertopic_model(texts: list, num_topics: int = 3):
    """
    Trains BERTopic model with multilingual sentence transformer embeddings
    and custom Tagalog+English stopwords passed to CountVectorizer.
    Includes fallback if torch/bertopic are not installed yet.
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        # Pass custom stopwords into CountVectorizer to keep c-TF-IDF keyword labels clean
        vectorizer_model = CountVectorizer(stop_words=list(ALL_STOPWORDS), min_df=1)
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            nr_topics=num_topics,
            calculate_probabilities=True,
            verbose=False
        )

        topics, probs = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()
        return {
            'model': topic_model,
            'topics': topics,
            'probs': probs,
            'topic_info': topic_info,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'topic_info': pd.DataFrame()
        }
