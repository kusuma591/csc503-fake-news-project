import os
import joblib
from sklearn.naive_bayes import MultinomialNB

MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)


def train(X_train, y_train, save=True):
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)
    
    if save:
        joblib.dump(clf, os.path.join(MODEL_DIR, "naive_bayes.pkl"))
        
    return clf


def predict(clf, X):
    return clf.predict(X)
