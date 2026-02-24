import pandas as pd
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "data", "raw", "amazon_reviews.txt")

def load_reviews(path=None, sample_size=200000):
    if path is None:
        path = DEFAULT_PATH
    print(f"[data_loader] Loading from: {path}")

    # ── Sniff first line to detect format ───────────────────────────────
    with open(path, encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    # ── JSON-lines ───────────────────────────────────────────────────────
    if first_line.startswith("{"):
        records = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    obj   = json.loads(line)
                    text  = obj.get("reviewText") or obj.get("text") or ""
                    score = float(obj.get("overall") or obj.get("rating") or 0)
                    if text.strip():
                        records.append({"text": str(text), "label": int(score >= 4)})
                except Exception:
                    continue
        df = pd.DataFrame(records)

    # ── TSV / CSV ────────────────────────────────────────────────────────
    else:
        sep = "\t" if "\t" in first_line else ","
        df  = pd.read_csv(path, sep=sep, on_bad_lines="skip", engine="python")
        df.columns = [c.strip().lower() for c in df.columns]

        print(f"[data_loader] Columns detected: {list(df.columns)}")

        text_col  = next((c for c in df.columns if c in
                          ["reviewtext","review_text","text","review","comment"]), None)
        label_col = next((c for c in df.columns if c in
                          ["overall","rating","label","score","stars","sentiment"]), None)

        # fallback: first col = label, second = text (common competition format)
        if text_col is None or label_col is None:
            print(f"[data_loader] Falling back to col[0]=label, col[1]=text")
            df.columns = ["label", "text"] + list(df.columns[2:])
            text_col, label_col = "text", "label"

        df = df[[text_col, label_col]].dropna()
        df.columns = ["text", "rating"]
        df["text"] = df["text"].astype(str)

        # Convert to binary
        try:
            df["rating"] = pd.to_numeric(df["rating"])
            df["label"]  = (df["rating"] >= 4).astype(int)
        except Exception:
            pos = {"positive","pos","1","good","great","5","4"}
            df["label"] = df["rating"].astype(str).str.strip().str.lower().isin(pos).astype(int)

        df = df[["text","label"]]

    # ── Subsample for speed ──────────────────────────────────────────────
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[data_loader] Subsampled to {sample_size} rows for speed")

    print(f"[data_loader] Final: {len(df)} rows | labels: {df['label'].value_counts().to_dict()}")
    return df
