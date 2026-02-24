import os
import math
from text_processing import extract_label_and_text, process_text

def update_word_frequencies(words, word_freq):
    # Takes words (list) and frequency dict (dict), updates document counts, returns nothing
    unique_words = set(words)
    for word in unique_words:
        word_freq[word] = word_freq.get(word, 0) + 1

def update_pair_frequencies(words, window_size, pair_freq):
    # Takes words (list), window size (int), and frequency dict (dict), updates pair counts, returns nothing
    unique_pairs = set()
    for i in range(len(words)):
        for j in range(i + 1, min(i + window_size, len(words))):
            pair = (words[i], words[j])
            unique_pairs.add(pair)
            
    for pair in unique_pairs:
        pair_freq[pair] = pair_freq.get(pair, 0) + 1

def calculate_pmi_score(pair_count, total_reviews, word1_count, word2_count):
    # Takes counts (ints), calculates pointwise mutual information, returns score (float)
    numerator = pair_count * total_reviews
    denominator = word1_count * word2_count
    if numerator > 0 and denominator > 0:
        return math.log2(numerator / denominator)
    return 0.0

def run_word_association(input_file, window_size, min_pair_freq):
    # Takes file path (str) and thresholds (ints), computes PMI for all pairs, returns top PMI results (list)
    word_freq = {}
    pair_freq = {}
    total_reviews = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        f.readline() # Skip header
        for line in f:
            label, text = extract_label_and_text(line)
            if text is None:
                continue
            
            total_reviews += 1
            words = process_text(text)
            update_word_frequencies(words, word_freq)
            update_pair_frequencies(words, window_size, pair_freq)

    pmi_results = {}
    for pair, freq in pair_freq.items():
        if freq >= min_pair_freq:
            w1, w2 = pair
            score = calculate_pmi_score(freq, total_reviews, word_freq[w1], word_freq[w2])
            if score > 0:
                pmi_results[pair] = score

    return sorted(pmi_results.items(), key=lambda x: x[1], reverse=True)[:100]