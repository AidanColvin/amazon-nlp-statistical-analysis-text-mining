# Itemset Mining: Amazon Review Text Analysis

## Overview
This repository contains three programming tasks. They focus on word association mining, feature selection for sentiment analysis, and spelling correction. All tasks use a collection of Amazon reviews and string similarity techniques.

---

## Question 1: Word Association

This module finds the top 100 word associations in a collection of Amazon reviews. It measures these associations using **pointwise mutual information (PMI)**.

**Implemented Guidelines:**
* Removes selected punctuation like commas and quotation marks.
* Considers only ordered word pairs that appear within a fixed-size text window of 5 consecutive words.
* Includes only word pairs that appear at least 50 times.
* Computes the frequency of each word across the reviews.
* Computes the frequency of each word pair across the reviews.

---

## Question 2: Feature Selection

This module finds 100 single words (unigrams) most associated with sentiment labels. It uses the **Chi-square** statistic to identify these associations.

**Details:**
* Uses the label `1` for positive sentiment.
* Uses the label `0` for negative sentiment.
* Identifies which sentiment each word is strongly associated with.
* Flags "mysterious" words that do not clearly associate with either positive or negative sentiment.

---

## Question 3: Spell Correction

This module implements a spelling correction system based on string similarity. It matches out-of-vocabulary strings to the closest word in a provided dictionary.

**Background and Approximation:**
A natural measure of string similarity is Levenshtein Edit Distance. The time complexity of edit distance is O(m * n) for strings of length m and n. This computation becomes expensive for long strings. To reduce computational cost, this system represents strings using overlapping n-grams.



Comparing n-gram sets provides high similarity accuracy while reducing the time complexity to O(m + n).

**Task Requirements:**
* Uses a Wiktionary dictionary containing all words starting with "a".
* Returns the top 10 most similar dictionary words for any input string using **Jaccard similarity**.
* Processes specific input strings: `abreviation`, `abstrictiveness`, `accanthopterigious`, `artifitial inteligwnse`, and `agglumetation`.
* Compares n-gram approximation results with Levenshtein edit distance results.
* Experiments with different n-gram lengths, including bigrams (2-grams), trigrams (3-grams), 4-grams, and 5-grams.