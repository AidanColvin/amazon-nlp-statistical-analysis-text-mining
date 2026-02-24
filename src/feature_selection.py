from text_processing import extract_label_and_text, process_text

def get_word_frequencies_by_label(input_file):
    # Takes file path (str), counts word occurrences per sentiment, returns positive and negative frequency dicts (tuple)
    pos_freq = {}
    neg_freq = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        f.readline() # Skip header
        for line in f:
            label, text = extract_label_and_text(line)
            if text is None:
                continue
                
            words = set(process_text(text))
            for word in words:
                if label == '1':
                    pos_freq[word] = pos_freq.get(word, 0) + 1
                elif label == '0':
                    neg_freq[word] = neg_freq.get(word, 0) + 1
                    
    return pos_freq, neg_freq

def calculate_chi_square_score(a, b, c, d, n):
    # Takes contingency counts (ints), calculates chi-square statistic, returns score (float)
    numerator = n * ((a * d) - (b * c))**2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator > 0:
        return numerator / denominator
    return 0.0

def run_feature_selection(input_file, total_pos, total_neg, total_reviews):
    # Takes file path (str) and corpus stats (ints), finds top sentiment words, returns sorted chi-square scores (list)
    pos_freq, neg_freq = get_word_frequencies_by_label(input_file)
    all_words = set(pos_freq.keys()) | set(neg_freq.keys())
    chi2_scores = {}

    for word in all_words:
        a = pos_freq.get(word, 0)
        b = neg_freq.get(word, 0)
        
        if (a + b) < 10:
            continue
            
        c = total_pos - a
        d = total_neg - b
        
        score = calculate_chi_square_score(a, b, c, d, total_reviews)
        chi2_scores[word] = score

    return sorted(chi2_scores.items(), key=lambda x: x[1], reverse=True)[:100]