from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    # Takes model object and split data, fits model and calculates accuracy, returns accuracy score (float)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")
    return accuracy

def build_logistic_regression():
    # Takes nothing, initializes Logistic Regression model, returns model object
    return LogisticRegression(max_iter=1000, random_state=42)

def build_support_vector_machine():
    # Takes nothing, initializes SVM model, returns model object
    return SVC(kernel='linear', random_state=42)

def build_random_forest():
    # Takes nothing, initializes Random Forest model, returns model object
    return RandomForestClassifier(n_estimators=100, random_state=42)

def build_gradient_boosting():
    # Takes nothing, initializes Gradient Boosting model, returns model object
    return GradientBoostingClassifier(n_estimators=100, random_state=42)

def build_xgboost():
    # Takes nothing, initializes XGBoost model, returns model object
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

def build_naive_bayes():
    # Takes nothing, initializes Naive Bayes model, returns model object
    return MultinomialNB()