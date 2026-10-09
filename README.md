# Tagalog Social Media Topic Modeling & Sentiment Analysis Pipeline
### Midterm Summative Activity — Natural Language Processing & Advanced Data Analytics

This repository provides an automated, end-to-end NLP workflow specifically designed for **Tagalog and code-switched (Taglish) social media data** (e.g., X/Twitter comments on Philippine Senate proceedings and the VP Sara Duterte impeachment issue).

It fulfills all requirements of the **Midterm Summative Activity**, featuring:
1. **Automated Multi-Stage Preprocessing** (preserves syntax for Transformers, normalizes Tagalog slang & emojis, extracts bigrams, and strips stop words for Bag-of-Words).
2. **Proximity & Similarity / Dissimilarity Analysis** (Cosine Similarity, Jaccard Similarity, Euclidean Distance).
3. **Topic Modeling & Evaluation** (Latent Dirichlet Allocation, Latent Semantic Analysis, and BERTopic with Topic Coherence $C_v$).
4. **Rule-Based Sentiment Analysis with Iterative Improvement** (Baseline V1 vs. Domain-Improved V2, 90–10 train/validation holdout split, 10-fold cross-validation, and performance metrics).

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/
│   │   ├── sample_dataset.csv          <- Bundled sample X/Twitter data (ready to test)
│   │   └── dataset.csv                 <- [DROP YOUR FULL 200+ SCRAPED CSV HERE]
│   └── processed/
│       └── preprocessed_dataset.csv    <- Generated multi-representation dataset
├── notebooks/
│   └── Midterm_Summative_Activity.ipynb <- Deliverable Jupyter Notebook with documentation
├── src/
│   ├── __init__.py
│   ├── config.py                       <- Stop words, slang dictionaries, iterative lexicons
│   ├── preprocess.py                   <- Automated multi-stage Tagalog text preprocessing
│   ├── similarity.py                   <- Cosine, Jaccard, and Euclidean proximity metrics
│   ├── topic_modeling.py               <- LDA, LSA, BERTopic & Coherence (C_v) evaluation
│   └── sentiment_analysis.py           <- Rule-based models, 90-10 split, 10-fold CV
├── main.py                             <- CLI runner executing the end-to-end pipeline
├── requirements.txt                    <- Dependency list
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/narcisoJavier/Data-mining.git
cd Data-mining
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. How to Input Your Scraped CSV Dataset
To analyze your full scraped dataset ($\ge 200$ rows):
1. Copy your CSV file into:
   ```text
   data/raw/dataset.csv
   ```
2. Ensure your CSV contains the following standard scraped columns:
   `source`, `platform`, `post_url`, `post_title`, `post_author`, `comment_text`, `username`, `likes`, `timestamp`
3. If `data/raw/dataset.csv` is not present, the pipeline automatically falls back to `data/raw/sample_dataset.csv`.

### 4. Run the Pipeline
To execute the automated pipeline from the terminal and display all formatted summary tables:
```bash
python main.py
```

To run the interactive notebook with graphs and report sections:
```bash
jupyter notebook notebooks/Midterm_Summative_Activity.ipynb
```

---

## 🧠 Why Multi-Stage Preprocessing for Tagalog?

Standard preprocessing often breaks when applied across different NLP models simultaneously:
* **LDA & LSA** require aggressive stop word removal (Tagalog: *ang, mga, sa, ng, na* + English: *the, is, in*), lowercasing, and bigrams (`confidential_funds`).
* **BERTopic** relies on sentence transformers and grammatical context. Stripping stop words or lemmatizing degrades BERT embedding quality.
* **Rule-Based Sentiment** requires negations (`hindi`, `wala`, `ayaw`), intensifiers (`sobra`, `talaga`), and emojis (😡, 🤬, 🙏) to determine emotional polarity.

Our pipeline automatically constructs dedicated representation columns:
* `text_for_bert` $\rightarrow$ Clean sentences with punctuation and grammar preserved.
* `tokens_for_lda` $\rightarrow$ Filtered tokens with collocations/bigrams for Gensim LDA/LSA.
* `text_for_sentiment` $\rightarrow$ Normalized slang, demojized tokens, and negation cues.

---

## 📊 Evaluation & Alignment with Activity Rubrics

| Activity Criterion | Implemented Component | Location in Code / Notebook |
| :--- | :--- | :--- |
| **Data Exploration & Preprocessing** | Automated deduplication, cleaning, and multi-representation generation | `src/preprocess.py` |
| **Similarity / Dissimilarity** | Cosine similarity, Jaccard similarity, and Euclidean distance matrices; top similar/dissimilar pair extraction | `src/similarity.py` |
| **Topic Modeling** | Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA), and BERTopic ($K \ge 3$ topics) | `src/topic_modeling.py` |
| **Topic Model Evaluation** | Topic Coherence ($C_v$) score computation across topics | `src/topic_modeling.py` |
| **Rule-Based Sentiment Analysis** | Positive, Negative, Neutral rule classification with 2-word lookahead negations and emotion weighting | `src/sentiment_analysis.py` |
| **Data Partitioning** | 90% Development set vs. 10% Unseen Validation Holdout | `src/sentiment_analysis.py` |
| **Cross-Validation** | 10-Fold Cross-Validation on the 90% development set | `src/sentiment_analysis.py` |
| **Iterative Improvement** | Quantitative comparison between Baseline Lexicon (V1) and Domain-Expanded Model (V2) | `src/sentiment_analysis.py` |
| **Systematic Tabular Output** | Preformatted tables showing Scripts, Outputs, and Remarks | `main.py` & Notebook Section 7 |

---

## 📝 Preparing Deliverables for Submission

According to the guidelines:
1. **Documentation PDF:** Save as `GROUP NAME.pdf` using 8.5 x 13 (long bond paper), Bookman Old Style 11, line spacing 1.15, full justification. Include all document sections (A to I).
2. **Jupyter Notebook:** Save as `GROUP NAME.ipynb` (use `notebooks/Midterm_Summative_Activity.ipynb` as the master template).
3. **Dataset:** Save as CSV in the appendix (`data/processed/preprocessed_dataset.csv`).
4. **Peer Evaluation Form:** Submit the consolidated group peer evaluation form alongside your PDF and notebook.
