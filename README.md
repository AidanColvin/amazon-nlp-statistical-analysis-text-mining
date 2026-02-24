## # Itemset Mining: Amazon Review Text Analysis

### ## Project Overview

This project implements a modular machine learning and natural language processing (NLP) pipeline. It utilizes Amazon review data to perform word association mining, sentiment-based feature selection, and string-similarity-based spelling correction. The system is designed for high performance using sparse matrices and automated workflows via a `Makefile`.

---

### ## Core Modules

#### ### 1. Statistical Word Association

Located in `src/word_association.py`, this module identifies significant relationships between words using **Pointwise Mutual Information (PMI)**.

* **Text Normalization**: Filters specific punctuation and standardizes casing.
* **Contextual Analysis**: Analyzes word pairs within a 5-word sliding window.
* **Threshold Filtering**: Only processes pairs with a minimum frequency of 50 occurrences to ensure statistical significance.
* **Output**: Identifies functional pairings (e.g., "customer service") and named entities.

#### ### 2. Sentiment Feature Selection

Located in `src/feature_selection.py`, this module extracts the top 100 unigrams most indicative of sentiment using the **Chi-square ($\chi^2$)** statistic.

* **Label Mapping**: Correlates words with binary sentiment labels ($1$ for positive, $0$ for negative).
* **Noise Reduction**: Utilizes stop-word filtering to remove non-informative "mysterious" words like "was" or "after."
* **Association Logic**: Segregates high-value features into positive indicators (e.g., "excellent") and negative indicators (e.g., "disappointed").

#### ### 3. Intelligent Spell Correction

Located in `src/spell_correction.py`, this system provides orthographic corrections by comparing out-of-vocabulary strings against a dictionary.

* **Algorithm Comparison**: Evaluates results using both **Jaccard Similarity** (via n-gram sets) and **Levenshtein Edit Distance**.
* **Optimization**: Uses n-gram approximation ($O(m + n)$ complexity) to provide faster results than traditional edit distance ($O(m \times n)$).
* **N-gram Analysis**: Supports bigrams through 5-grams; testing indicates bigrams (2-grams) offer the best balance of flexibility and similarity scoring.
* **Scope**: Primary dictionary focuses on Wiktionary entries starting with the letter "a."

---

### ## Installation & Usage

#### ### Prerequisites

Ensure you have Python 3.13+ installed. The environment should include `scikit-learn`, `pandas`, `numpy`, `xgboost`, and `scipy`.

#### ### Execution

The entire pipeline is automated. Use the following commands in the terminal:

* **Run Pipeline**: `make run` — Installs dependencies, builds features, and trains models.
* **Clean Environment**: `make clean` — Removes processed data files to reset the pipeline.

---

### ## Technical Specifications

* **Data Hierarchy**: Original data is stored in `data/raw/` to ensure immutability. All generated outputs and models are stored in `data/processed/`.
* **Model Selection**: Includes Logistic Regression, SVM, Random Forest, XGBoost, Gradient Boosting, and Naive Bayes.
* **Performance Tracking**: Generates a `model_comparison_report.csv` sorted by F1-Score to verify model accuracy.

---

**Would you like me to add a section to the README that explains the specific results of your model performance report?**