import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "output", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)


def fit(train_texts, save=True):
    vec = TfidfVectorizer(
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
        ngram_range=(1, 2)
    )
    vec.fit(train_texts)
    
    if save:
        joblib.dump(vec, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    
    return vec


def transform(vec, texts):
    return vec.transform(texts)
