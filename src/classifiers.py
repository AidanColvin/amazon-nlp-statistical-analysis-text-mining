from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

def build_logistic_regression():
    # Takes nothing, initializes Logistic Regression model, returns model object
    return LogisticRegression(max_iter=1000, random_state=42)

def build_support_vector_machine():
    # Takes nothing, initializes SVM model, returns model object
    return SVC(kernel='linear', probability=True, random_state=42)

def build_random_forest():
    # Takes nothing, initializes Random Forest model, returns model object
    return RandomForestClassifier(n_estimators=100, random_state=42)

def build_gradient_boosting():
    # Takes nothing, initializes Gradient Boosting model, returns model object
    return GradientBoostingClassifier(n_estimators=100, random_state=42)

def build_naive_bayes():
    # Takes nothing, initializes Naive Bayes model, returns model object
    return MultinomialNB()

def get_all_models():
    # Takes nothing, maps model names to model objects, returns dict of models (dict)
    return {
        "Logistic Regression": build_logistic_regression(),
        "Support Vector Machine": build_support_vector_machine(),
        "Random Forest": build_random_forest(),
        "Gradient Boosting": build_gradient_boosting(),
        "Naive Bayes": build_naive_bayes()
    }