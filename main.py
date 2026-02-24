import os
from src.text_processing import load_dictionary
from src.word_association import run_word_association
from src.feature_selection import run_feature_selection
from src.spell_correction import find_top_jaccard_matches, find_top_edit_distance_matches

INPUT_FILE = './data/raw/amazon_reviews.txt'
DICT_FILE = './data/raw/enwiktionary.a.list'
OUTPUT_DIR = './data/processed'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Takes nothing, executes the full data mining pipeline, returns nothing
    
    print("--- 1. Word Association (PMI) ---")
    pmi_results = run_word_association(INPUT_FILE, window_size=5, min_pair_freq=50)
    with open(os.path.join(OUTPUT_DIR, 'pmi_top_100.txt'), 'w', encoding='utf-8') as f:
        for pair, score in pmi_results:
            f.write(f"{pair[0]} {pair[1]}\t{score:.4f}\n")
    print("PMI results saved.")

    print("\n--- 2. Feature Selection (Chi-Square) ---")
    # Hardcoded corpus stats for brevity (replace with dynamic counts as needed)
    total_reviews = 10000 
    total_pos = 5000
    total_neg = 5000
    
    chi2_results = run_feature_selection(INPUT_FILE, total_pos, total_neg, total_reviews)
    with open(os.path.join(OUTPUT_DIR, 'chi2_top_100.txt'), 'w', encoding='utf-8') as f:
        for word, score in chi2_results:
            f.write(f"{word}\t{score:.4f}\n")
    print("Chi-Square results saved.")

    print("\n--- 3. Spell Correction ---")
    dictionary = load_dictionary(DICT_FILE)
    misspelled_words = ["abreviation", "abstrictiveness", "accanthopterigious", "artifitial inteligwnse", "agglumetation"]
    
    if dictionary:
        with open(os.path.join(OUTPUT_DIR, 'spell_correction.txt'), 'w', encoding='utf-8') as f:
            for word in misspelled_words:
                f.write(f"\nTarget: {word}\n")
                jaccard_matches = find_top_jaccard_matches(word, dictionary, n=2)
                f.write(f"Jaccard Bigram Top Match: {jaccard_matches[0][0] if jaccard_matches else 'None'}\n")
                
                edit_matches = find_top_edit_distance_matches(word, dictionary)
                f.write(f"Edit Dist Top Match: {edit_matches[0][0] if edit_matches else 'None'}\n")
        print("Spell Correction results saved.")

if __name__ == "__main__":
    main()