import pandas as pd, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "data", "raw", "amazon_reviews.txt")

def load_reviews(path=None, sample_size=200000):
    if path is None: path = DEFAULT_PATH
    print(f"[data_loader] Loading from: {path}")
    df = pd.read_csv(path, sep="	", on_bad_lines="skip", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[data_loader] Columns: {list(df.columns)}")
    text_col = next((c for c in df.columns if c in ["reviewtext","review_text","text","review","comment"]), None)
    label_col = next((c for c in df.columns if c in ["overall","rating","label","score","stars","sentiment"]), None)
    if text_col is None or label_col is None:
        df.columns = ["label","text"] + list(df.columns[2:])
        text_col, label_col = "text", "label"
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    print(f"[data_loader] Raw unique labels: {sorted(df[chr(108)+chr(97)+chr(98)+chr(101)+chr(108)].unique())}")
    if df["label"].max() <= 1:
        df["label"] = df["label"].astype(int)
    else:
        df["label"] = (df["label"] >= 4).astype(int)
    pos = df[df["label"]==1]
    neg = df[df["label"]==0]
    n = min(len(pos), len(neg), sample_size // 2)
    df = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[data_loader] Final: {len(df)} rows | labels: {df[chr(108)+chr(97)+chr(98)+chr(101)+chr(108)].value_counts().to_dict()}")
    return df