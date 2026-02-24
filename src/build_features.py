import os
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

INPUT_FILE = './data/raw/amazon_reviews.txt'
OUTPUT_DIR = './data/processed'

def parse_dataset(filepath):
    # Takes filepath (str), extracts valid labels and text, returns labels (list) and texts (list)
    labels = []
    texts = []
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return labels, texts
        
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline() # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                labels.append(int(parts[0]))
                texts.append(parts[1])
    return labels, texts

def vectorize_text(texts):
    # Takes texts (list), converts to TF-IDF matrix, returns feature matrix (sparse) and vectorizer (object)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def split_data(X, y):
    # Takes feature matrix (sparse) and labels (array), splits into 80/20 sets, returns splits (tuple)
    return train_test_split(X, y, test_size=0.20, random_state=42)

def save_data_splits(X_train, X_test, y_train, y_test, vectorizer, output_dir):
    # Takes data splits and vectorizer, saves them to disk, returns nothing
    sparse.save_npz(os.path.join(output_dir, 'X_train.npz'), X_train)
    sparse.save_npz(os.path.join(output_dir, 'X_test.npz'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.joblib'))

def main():
    # Takes nothing, coordinates parsing, vectorizing, and splitting, returns nothing
    print("Building features and splitting data (80/20)...")
    labels, texts = parse_dataset(INPUT_FILE)
    
    if not labels or not texts:
        print("No valid data found to process. Exiting.")
        return

    X, vectorizer = vectorize_text(texts)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    save_data_splits(X_train, X_test, y_train, y_test, vectorizer, OUTPUT_DIR)
    print(f"Saved training/testing data to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()