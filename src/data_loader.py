import pandas as pd
import json
import os

# Resolve path relative to project root, not cwd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "data", "raw", "amazon_reviews.txt")

def load_reviews(path=None):
    if path is None:
        path = DEFAULT_PATH
    print(f"[data_loader] Loading from: {path}")
    try:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip", engine="python")
        df.columns = [c.strip().lower() for c in df.columns]
        text_col  = next((c for c in df.columns if c in
                          ["reviewtext","review_text","text","review","comment"]), df.columns[-1])
        label_col = next((c for c in df.columns if c in
                          ["overall","rating","label","score","stars","sentiment"]), None)
        if label_col is None:
            raise ValueError(f"No label column found. Columns: {list(df.columns)}")
        df = df[[text_col, label_col]].dropna()
        df.columns = ["text", "rating"]
        df["text"] = df["text"].astype(str)
        try:
            df["rating"] = pd.to_numeric(df["rating"])
            df["label"] = (df["rating"] >= 4).astype(int)
        except Exception:
            pos = {"positive","pos","1","good","great"}
            df["label"] = df["rating"].str.strip().str.lower().isin(pos).astype(int)
        print(f"[data_loader] Loaded {len(df)} rows | label dist: {df['label'].value_counts().to_dict()}")
        return df[["text","label"]]
    except Exception as e:
        print(f"[data_loader] TSV failed ({e}), trying JSON-lines...")
        records = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj   = json.loads(line)
                    text  = obj.get("reviewText") or obj.get("text") or ""
                    score = float(obj.get("overall") or obj.get("rating") or 0)
                    records.append({"text": str(text), "label": int(score >= 4)})
                except Exception:
                    continue
        if not records:
            raise RuntimeError(f"Could not parse {path} as TSV or JSON-lines")
        df = pd.DataFrame(records)
        print(f"[data_loader] Loaded {len(df)} rows via JSON-lines")
        return df
