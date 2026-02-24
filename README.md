```md
# Itemset Mining & NLP Pipeline: Amazon Review Intelligence

> A high-performance, modular NLP pipeline for extracting semantic relationships, sentiment signals, and lexical corrections from large-scale Amazon review data.

---

## Overview

This project implements an end-to-end natural language processing (NLP) and machine learning pipeline designed to:

- Discover statistically significant word associations  
- Identify sentiment-discriminative features  
- Perform efficient, scalable spell correction  

The system emphasizes **modularity, computational efficiency, and reproducibility**, leveraging sparse representations and automated workflows via a `Makefile`.

---

## Key Features

- Statistical Word Association Mining (PMI)  
- Chi-Square-Based Sentiment Feature Selection  
- Optimized Spell Correction (Jaccard + Levenshtein)  
- Multi-model ML benchmarking pipeline  
- Reproducible build system using Makefile  
- Efficient handling of large text corpora via sparse matrices  

---

## Architecture

```
data/
├── raw/                # Immutable source data
└── processed/          # Generated features, models, outputs

src/
├── word_association.py
├── feature_selection.py
└── spell_correction.py

Makefile
README.md
```

---

## Core Components

### 1. Word Association Mining (PMI)

**Location:** `src/word_association.py`

- Computes **Pointwise Mutual Information (PMI)** to detect meaningful word pair relationships  
- Sliding window of **±5 tokens** for contextual co-occurrence  
- Filters low-signal pairs using a **minimum frequency threshold (≥50)**  
- Outputs semantically meaningful associations (e.g., `"customer" ↔ "service"`)  

**Why it matters:**  
Captures latent structure in language beyond simple frequency counts.

---

### 2. Sentiment Feature Selection (Chi-Square)

**Location:** `src/feature_selection.py`

- Uses **χ² (Chi-square statistic)** to identify top 100 sentiment-bearing unigrams  
- Binary label mapping:
  - `1` → Positive sentiment  
  - `0` → Negative sentiment  
- Applies stop-word filtering to remove non-informative tokens  
- Produces interpretable feature sets:
  - Positive: `"excellent"`, `"amazing"`  
  - Negative: `"disappointed"`, `"waste"`  

**Why it matters:**  
Improves downstream model performance by isolating statistically relevant signals.

---

### 3. Intelligent Spell Correction

**Location:** `src/spell_correction.py`

- Hybrid similarity approach:
  - **Jaccard Similarity (n-grams)**  
  - **Levenshtein Edit Distance**  
- Performance optimization:
  - Jaccard: **O(m + n)**  
  - Levenshtein: **O(m × n)**  
- Supports **2–5 gram tokenization**  
  - Empirically, **bigrams (2-grams)** provide best trade-off  
- Dictionary scope: Wiktionary subset (terms starting with "a")  

**Why it matters:**  
Balances accuracy and computational efficiency for real-world NLP systems.

---

## Machine Learning Pipeline

The pipeline evaluates multiple models for sentiment classification:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Naive Bayes  

### Output

- `model_comparison_report.csv`  
  - Sorted by **F1-score**  
  - Enables objective model benchmarking and selection  

---

## Installation

### Prerequisites

- Python **3.13+**

### Required Libraries

```bash
pip install scikit-learn pandas numpy xgboost scipy
```

---

## Usage

### Run Full Pipeline

```bash
make run
```

Executes:

* Dependency setup
* Data preprocessing
* Feature extraction
* Model training & evaluation

---

### Reset Environment

```bash
make clean
```

Removes all generated artifacts in `data/processed/`.

---

## Design Principles

* **Modularity** — Each component is independently testable and extensible
* **Efficiency** — Sparse matrices and optimized algorithms reduce computational overhead
* **Reproducibility** — Deterministic pipeline execution via Makefile
* **Scalability** — Designed for large-scale text corpora

---

## Example Applications

* Review sentiment analysis at scale
* Keyword and phrase discovery for product insights
* Data cleaning and preprocessing pipelines
* Feature engineering for NLP classification tasks

---

## Future Enhancements

* Expand dictionary coverage beyond "a"-prefixed entries
* Integrate deep learning models (e.g., transformers)
* Add real-time inference API
* Incorporate phrase-level sentiment modeling

---
