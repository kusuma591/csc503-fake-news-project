import os
import joblib
from sklearn.svm import LinearSVC

MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)


def train(X_train, y_train, save=True):
    clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    
    if save:
        joblib.dump(clf, os.path.join(MODEL_DIR, "svm.pkl"))
        
    return clf


def predict(clf, X):
    return clf.predict(X)
