import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from classifiers import get_all_models

PROCESSED_DIR = './data/processed'

def load_data_splits(input_dir):
    # Takes input directory (str), loads data splits from disk, returns splits (tuple)
    
    X_train = sparse.load_npz(os.path.join(input_dir, 'X_train.npz'))
    X_test = sparse.load_npz(os.path.join(input_dir, 'X_test.npz'))
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
    return X_train, X_test, y_train, y_test

def evaluate_performance(model, X_test, y_test):
    # Takes trained model and test data, calculates performance metrics, returns metrics (dict)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='macro', zero_division=0)
    
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4)
    }

def save_trained_model(model, name, output_dir):
    # Takes trained model, name (str), and directory (str), saves model to disk, returns nothing
    
    filename = name.replace(" ", "_").lower() + ".joblib"
    filepath = os.path.join(output_dir, filename)
    joblib.dump(model, filepath)

def main():
    # Takes nothing, orchestrates loading, training, and reporting, returns nothing
    
    X_train, X_test, y_train, y_test = load_data_splits(PROCESSED_DIR)
    models = get_all_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_performance(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        save_trained_model(model, name, PROCESSED_DIR)

    df = pd.DataFrame(results).set_index("Model")
    print(df.sort_values(by="F1-Score", ascending=False).to_string())
    df.to_csv(os.path.join(PROCESSED_DIR, "report.csv"))

if __name__ == "__main__":
    main()