import os, pandas as pd
from data_loader import load_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

NAME = "gradient-boosting"
OUT  = "data/processed"
os.makedirs(OUT, exist_ok=True)

df = load_reviews()
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
vec  = TfidfVectorizer(max_features=5000, sublinear_tf=True)
Xtr  = vec.fit_transform(X_train).toarray()
Xte  = vec.transform(X_test).toarray()
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(Xtr, y_train)
preds = model.predict(Xte)

pd.DataFrame({"id": X_test.index, "prediction": preds, "actual": y_test.values}).to_csv(f"{OUT}/{NAME}-submission.csv", index=False)
pd.DataFrame(classification_report(y_test, preds, output_dict=True)).transpose().to_csv(f"{OUT}/{NAME}-classification-report.csv")
print(f"[{NAME}] Accuracy={accuracy_score(y_test,preds):.4f}  F1={f1_score(y_test,preds):.4f}")
