import pandas as pd, glob, os, re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

OUT   = "data/processed"
files = glob.glob(f"{OUT}/*-submission.csv")
rows  = []
for fp in files:
    name = re.sub(r"-submission\.csv$", "", os.path.basename(fp))
    df   = pd.read_csv(fp)
    rows.append({
        "model":     name,
        "accuracy":  round(accuracy_score(df["actual"], df["prediction"]), 4),
        "f1":        round(f1_score(df["actual"], df["prediction"]), 4),
        "precision": round(precision_score(df["actual"], df["prediction"]), 4),
        "recall":    round(recall_score(df["actual"], df["prediction"]), 4),
        "n_test":    len(df),
    })
report = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
report.index += 1
report.index.name = "rank"
report.to_csv(f"{OUT}/model_comparison_report.csv")
print("\n=== Model Comparison (by F1) ===")
print(report.to_string())
