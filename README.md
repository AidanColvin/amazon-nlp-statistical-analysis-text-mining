# Amazon NLP: Itemset Mining & Statistical Text Analysis

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A high-performance, modular NLP pipeline for extracting semantic relationships, sentiment signals, and lexical corrections from large-scale Amazon review data.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Core Components](#core-components)
- [Models & Results](#models--results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Pipeline](#running-the-pipeline)
- [Design Principles](#design-principles)
- [Future Enhancements](#future-enhancements)

---

## Overview

This project implements an end-to-end NLP and machine learning pipeline designed to mine semantic structure, sentiment signals, and lexical patterns from Amazon product reviews at scale.

Three analytical components drive the system: statistical word association mining via PMI, chi-square sentiment feature selection, and a hybrid spell correction engine. A multi-model ML benchmark evaluates classifier performance across all engineered features.

---

## Dataset

The dataset is hosted on Google Drive and must be downloaded before running the pipeline.

📂 [Amazon Review Dataset](https://drive.google.com/drive/folders/1ZDa2qHMPxQSXxg-Bn-LnoXwi41ZoZI-r)

Place the raw files into `data/raw/` before executing any scripts. This directory is created automatically on first run.

---

## Repository Structure

```
.
├── src/
│   ├── download_data.py              # Dataset download utility
│   ├── data_loader.py                # Review loading & label handling
│   ├── text_processing.py            # Tokenization & preprocessing
│   ├── convert_to_parquet.py         # CSV → Parquet conversion
│   ├── build_features.py             # Feature matrix construction
│   ├── word_association.py           # PMI-based word association mining
│   ├── feature_selection.py          # Chi-square sentiment feature selection
│   ├── spell_correction.py           # Hybrid Jaccard + Levenshtein correction
│   ├── classifiers.py                # Classifier definitions
│   ├── train_logistic_regression.py
│   ├── train_svm.py
│   ├── train_random_forest.py
│   ├── train_gradient_boosting.py
│   ├── train_naive_bayes.py
│   ├── train_xgboost.py
│   ├── train_models.py               # Unified training runner
│   └── compare_models.py             # Leaderboard & F1 comparison
├── data/
│   ├── raw/                          # Downloaded source files (not tracked)
│   └── processed/                    # Auto-generated pipeline artifacts
├── main.py
├── Makefile
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Core Components

### 1. Word Association Mining — `word_association.py`

Computes **Pointwise Mutual Information (PMI)** over a sliding ±5 token window to detect meaningful word pair co-occurrences. Pairs below a minimum frequency threshold of 50 are filtered as low-signal noise, surfacing associations such as `"customer" ↔ "service"` that reflect genuine semantic structure beyond raw frequency counts.

---

### 2. Sentiment Feature Selection — `feature_selection.py`

Applies **χ² (Chi-square)** statistics to identify the top 100 sentiment-discriminative unigrams from the review corpus. Labels are mapped to binary values (1 = positive, 0 = negative) and stop words are removed prior to scoring. Representative outputs include positive signals like `"excellent"` and `"amazing"`, and negative signals like `"disappointed"` and `"waste"`.

---

### 3. Spell Correction — `spell_correction.py`

A hybrid similarity engine combining **Jaccard Similarity (n-gram)** and **Levenshtein Edit Distance** for balanced accuracy and speed. Jaccard operates at O(m + n) for fast candidate filtering; Levenshtein at O(m × n) for precise final ranking. Bigrams (2-grams) provide the best empirical accuracy-efficiency trade-off. Dictionary scope covers a Wiktionary subset of "a"-prefixed terms.

---

## Models & Results

All classifiers are evaluated on chi-square selected features and ranked by **F1-score** in `data/processed/model_comparison_report.csv`.

| Model | Metric |
|---|:---:|
| Logistic Regression | F1-ranked |
| Support Vector Machine | F1-ranked |
| Random Forest | F1-ranked |
| Gradient Boosting | F1-ranked |
| Naive Bayes | F1-ranked |
| XGBoost (HistGradientBoosting) | F1-ranked |

> Results are written to `data/processed/model_comparison_report.csv` on each run.

---

## Getting Started

### Prerequisites

- Python 3.13 or higher

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AidanColvin/amazon-nlp-statistical-analysis-text-mining.git
   cd amazon-nlp-statistical-analysis-text-mining
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:

   ```bash
   pip install scikit-learn pandas numpy xgboost scipy gdown
   ```

### Running the Pipeline

**Download the dataset first:**

```bash
python3 src/download_data.py
```

**Run the full pipeline:**

```bash
make run
```

This executes dependency setup, data preprocessing, feature extraction, model training, and evaluation in sequence. All outputs are saved to `data/processed/`.

**Reset the environment:**

```bash
make clean
```

Removes all generated artifacts in `data/processed/`.

---

## Design Principles

- **Modularity** — Each component is independently testable and extensible
- **Efficiency** — Sparse matrix representations and optimized algorithms minimize overhead
- **Reproducibility** — Deterministic pipeline execution via `Makefile`
- **Scalability** — Designed for large-scale text corpora

---

## Future Enhancements

- Expand spell correction dictionary beyond "a"-prefixed entries
- Integrate transformer-based models (e.g., BERT, RoBERTa)
- Add a real-time inference API
- Incorporate phrase-level and aspect-based sentiment modeling
