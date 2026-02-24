def chunk_word_into_letter_ngrams(word, n):
    # Takes word (str) and n (int), generates overlapping n-grams, returns n-grams (set)
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i : i+n])
    return set(ngrams)

def calculate_jaccard_similarity(set_a, set_b):
    # Takes two sets, calculates intersection over union, returns Jaccard score (float)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / union

def calculate_edit_distance(s1, s2):
    # Takes two strings, calculates Levenshtein distance iteratively, returns distance (int)
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
    return dp[m][n]

def find_top_jaccard_matches(target_word, dictionary, n, top_k=10):
    # Takes target word (str), dictionary (list), n-gram size (int), finds best matches, returns matches (list)
    scores = []
    target_ngrams = chunk_word_into_letter_ngrams(target_word, n)
    
    if not target_ngrams:
        return []

    for word in dictionary:
        if abs(len(word) - len(target_word)) > 4:
            continue
            
        word_ngrams = chunk_word_into_letter_ngrams(word, n)
        if not word_ngrams: 
            continue
            
        score = calculate_jaccard_similarity(target_ngrams, word_ngrams)
        if score > 0:
            scores.append((word, score))
            
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def find_top_edit_distance_matches(target_word, dictionary, top_k=10):
    # Takes target word (str), dictionary (list), finds closest words by edit distance, returns matches (list)
    scores = []
    for word in dictionary:
        if abs(len(word) - len(target_word)) > 5:
            continue
            
        dist = calculate_edit_distance(target_word, word)
        scores.append((word, dist))
        
    return sorted(scores, key=lambda x: x[1])[:top_k]