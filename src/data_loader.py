import pandas as pd
import json

def load_reviews(path="amazon_review.txt"):
    try:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip", engine="python")
        df.columns = [c.strip().lower() for c in df.columns]
        text_col  = next((c for c in df.columns if c in ["reviewtext","review_text","text","review","comment"]), df.columns[-1])
        label_col = next((c for c in df.columns if c in ["overall","rating","label","score","stars","sentiment"]), None)
        df = df[[text_col, label_col]].dropna()
        df.columns = ["text", "rating"]
        df["text"] = df["text"].astype(str)
        try:
            df["rating"] = pd.to_numeric(df["rating"])
            df["label"] = (df["rating"] >= 4).astype(int)
        except Exception:
            pos = {"positive","pos","1","good","great"}
            df["label"] = df["rating"].str.strip().str.lower().isin(pos).astype(int)
        return df[["text","label"]]
    except Exception:
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
        return pd.DataFrame(records)
