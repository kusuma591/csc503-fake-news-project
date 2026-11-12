import argparse
import random
import numpy as np
import pandas as pd
import sys
import os

random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.preprocess import partition, print_partition_summary
from features import tfidf
from models import naive_bayes as nb_module
from models import svm as svm_module
from evaluation import metrics as met

FAKE_WORDS = [
    "shocking", "unbelievable", "exposed", "conspiracy", "secret",
    "banned", "coverup", "whistleblower", "truth", "hidden",
    "government", "lies", "propaganda", "fake", "hoax",
]
REAL_WORDS = [
    "according", "reported", "confirmed", "announced", "official",
    "research", "study", "data", "evidence", "analysis",
    "published", "statement", "policy", "committee", "press",
]


def make_article(fake, topic, date):
    words = FAKE_WORDS if fake else REAL_WORDS
    extra = random.sample(["news", "article", "report", "claim", "source"], 3)
    body = " ".join(random.choices(words + extra, k=80))
    title = " ".join(random.choices(words, k=8))
    return {
        "title": title,
        "text": body,
        "text_combined": title + " " + body,
        "label": int(fake),
        "topic": topic,
        "date": date,
    }


def make_synthetic_data(n=500):
    rows = []

    for _ in range(n):
        date = pd.Timestamp("2017-01-01") + pd.Timedelta(days=random.randint(0, 364))
        rows.append(make_article(True, "politics", date))
        rows.append(make_article(False, "politics", date))

    for _ in range(n // 2):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=random.randint(0, 364))
        rows.append(make_article(True, "politics", date))
        rows.append(make_article(False, "politics", date))

    for _ in range(n // 2):
        date = pd.Timestamp("2017-06-01") + pd.Timedelta(days=random.randint(0, 180))
        rows.append(make_article(True, "world", date))
        rows.append(make_article(False, "world", date))

    for _ in range(n // 2):
        date = pd.Timestamp("2021-01-01") + pd.Timedelta(days=random.randint(0, 364))
        rows.append(make_article(True, "world", date))
        rows.append(make_article(False, "world", date))

    return pd.DataFrame(rows)


def run_pipeline(df):
    print("Partitioning data...")
    train_df, test_df, temporal, topic, both = partition(df)
    print_partition_summary(train_df, test_df, temporal, topic, both)

    print("\nFitting TF-IDF on training data only...")
    vec = tfidf.fit(train_df["text_combined"])
    print(f"  Vocabulary size: {len(vec.vocabulary_):,}")

    X_train = tfidf.transform(vec, train_df["text_combined"])
    splits = {
        "baseline": (tfidf.transform(vec, test_df["text_combined"]), test_df["label"]),
        "temporal": (tfidf.transform(vec, temporal["text_combined"]), temporal["label"]),
        "topic":    (tfidf.transform(vec, topic["text_combined"]), topic["label"]),
        "both":     (tfidf.transform(vec, both["text_combined"]), both["label"]),
    }

    classifiers = {
        "Naive Bayes": nb_module,
        "Linear SVM": svm_module,
    }

    results = {}
    deltas = {}

    for clf_name, clf_mod in classifiers.items():
        print(f"\n{clf_name}")
        clf = clf_mod.train(X_train, train_df["label"])
        results[clf_name] = {}
        deltas[clf_name] = {}

        for cond, (X, y) in splits.items():
            if len(y) == 0:
                results[clf_name][cond] = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
                continue
            m = met.evaluate(y, clf_mod.predict(clf, X))
            results[clf_name][cond] = m
            print(f"  {cond:12s}  accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")

        for cond in ["temporal", "topic", "both"]:
            deltas[clf_name][cond] = met.delta(
                results[clf_name][cond],
                results[clf_name]["baseline"]
            )

    df_results = met.build_results_table(results, deltas)
    met.save_results(df_results, "quickstart_results.csv")

    print("\nRESULTS TABLE")
    print(df_results.to_string(index=False))
    print("\nDone. Results saved to evaluation/results/quickstart_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Use real Kaggle data instead of synthetic")
    args = parser.parse_args()

    if args.real:
        print("Loading real datasets from data/raw/...")
        from data.load import load_and_merge
        df = load_and_merge()
    else:
        print("Running on synthetic demo data (no download needed)...")
        df = make_synthetic_data(n=500)

    run_pipeline(df)
