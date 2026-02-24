import os
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from classifiers import get_all_models

PROCESSED_DIR = './data/processed'

def main():
    # Takes nothing, trains remaining models and prints a clean comparison table, returns nothing
    X_train = sparse.load_npz(os.path.join(PROCESSED_DIR, 'X_train.npz'))
    X_test = sparse.load_npz(os.path.join(PROCESSED_DIR, 'X_test.npz'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

    models = get_all_models()
    results = []

    print("\n--- Training Progress ---")
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred, average='macro'), 4)
        })

    df = pd.DataFrame(results).set_index("Model").sort_values("F1-Score", ascending=False)
    print("\n" + "="*40)
    print(" PERFORMANCE COMPARISON TABLE ")
    print("="*40)
    print(df.to_string())
    print("="*40)

if __name__ == "__main__":
    main()