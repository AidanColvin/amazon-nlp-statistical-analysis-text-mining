import os, pandas as pd
from data_loader import load_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from lightgbm import LGBMClassifier

NAME = "xgboost-lgbm"
OUT  = "data/processed"
os.makedirs(OUT, exist_ok=True)

df = load_reviews()
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
vec  = TfidfVectorizer(max_features=10000, sublinear_tf=True)
Xtr  = vec.fit_transform(X_train)
Xte  = vec.transform(X_test)
model = LGBMClassifier(n_estimators=300, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1)
model.fit(Xtr, y_train)
preds = model.predict(Xte)

pd.DataFrame({"id": X_test.index, "prediction": preds, "actual": y_test.values}).to_csv(f"{OUT}/{NAME}-submission.csv", index=False)
pd.DataFrame(classification_report(y_test, preds, output_dict=True)).transpose().to_csv(f"{OUT}/{NAME}-classification-report.csv")
print(f"[{NAME}] Accuracy={accuracy_score(y_test,preds):.4f}  F1={f1_score(y_test,preds):.4f}")
