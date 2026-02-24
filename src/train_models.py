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
    try:
        X_train = sparse.load_npz(os.path.join(input_dir, 'X_train.npz'))
        X_test = sparse.load_npz(os.path.join(input_dir, 'X_test.npz'))
        y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data splits: {e}")
        return None, None, None, None

def evaluate_model(model, X_test, y_test):
    # Takes trained model and test data, calculates performance metrics, returns metrics (dict)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='macro', zero_division=0)
    
    avg_confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        avg_confidence = np.mean(np.max(probabilities, axis=1))

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1_score, 4),
        "Mean Probability Confidence": round(avg_confidence, 4)
    }

def save_model_to_disk(model, name, output_dir):
    # Takes trained model, name (str), and directory (str), saves model to disk, returns nothing
    filename = name.replace(" ", "_").lower() + ".joblib"
    filepath = os.path.join(output_dir, filename)
    joblib.dump(model, filepath)

def train_and_evaluate_all(models, X_train, X_test, y_train, y_test, output_dir):
    # Takes models and data, trains models and calculates metrics, returns list of results (list)
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        
        save_model_to_disk(model, name, output_dir)
    return results

def generate_report(results, output_dir):
    # Takes results (list) and output directory (str), saves and prints comparison report, returns nothing
    df_results = pd.DataFrame(results).set_index("Model")
    report_path = os.path.join(output_dir, "model_comparison_report.csv")
    df_results.to_csv(report_path)
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON (Sorted by F1-Score)")
    print("=" * 80)
    print(df_results.sort_values(by="F1-Score", ascending=False).to_string())
    print("=" * 80)
    print(f"\nAll models saved to {output_dir}")
    print(f"Comparison report saved to {report_path}")

def main():
    # Takes nothing, orchestrates loading data, training models, and generating report, returns nothing
    print("\nLoading 80/20 data splits...")
    X_train, X_test, y_train, y_test = load_data_splits(PROCESSED_DIR)
    
    if X_train is None:
        print("Data splits not found. Please run build_features.py first.")
        return

    models = get_all_models()
    print("\nTraining and Evaluating Models:")
    print("-" * 60)
    
    results = train_and_evaluate_all(models, X_train, X_test, y_train, y_test, PROCESSED_DIR)
    generate_report(results, PROCESSED_DIR)

if __name__ == "__main__":
    main()