import random
import numpy as np
import sys
import os
import argparse
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(__file__))

from data.load import load_and_merge
from data.preprocess import partition, print_partition_summary
from features import tfidf
from models import naive_bayes as nb_module
from models import svm as svm_module
from evaluation import metrics as met
from evaluation import cross_validation as cv_module


def run(run_cv=False, output_cm=False, tuned=False):
    print("AMPETTS: Fake News Detection Under Topic and Temporal Shift")
    print("Team 6R: Olivia Eskes, Jay Robertson, Markus Sekhon, Kusuma Mattipa")
    print("CSC 503 Data Mining, University of Victoria")
    print()

    print("Step 1: Loading datasets...")
    df = load_and_merge()
    
    print("\n  Dataset topic and label distribution:")
    print("  "+
        df.groupby(["source_dataset", "topic", "label"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    
    print(f"  Total articles loaded: {len(df):,}")
    print(f"  Fake: {df['label'].sum():,}   Real: {(df['label'] == 0).sum():,}")
    print()

    print("Step 2: Partitioning into 4 conditions...")
    train_df, test_df, temporal, topic, both = partition(df)
    print_partition_summary(train_df, test_df, temporal, topic, both)
    print()
    
    if run_cv:
        print("Step 2.5: Running 5-fold cross-validation...")
        cv_results = cv_module.run_cross_validation(
            train_df,
            n_splits=5
        )    

    print("Step 3: Fitting TF-IDF on training data only...")
    vec = tfidf.fit(train_df["text_combined"])
    print(f"  Vocabulary size: {len(vec.vocabulary_):,} terms")
    print()

    X_train = tfidf.transform(vec, train_df["text_combined"])

    splits = {
        "baseline": (tfidf.transform(vec, test_df["text_combined"]), test_df["label"]),
        "temporal": (tfidf.transform(vec, temporal["text_combined"]), temporal["label"]),
        "topic":    (tfidf.transform(vec, topic["text_combined"]), topic["label"]),
        "both":     (tfidf.transform(vec, both["text_combined"]), both["label"]),
    }

    print("Step 4: Training classifiers and evaluating...")
    if tuned:
        classifiers = {
            "Naive Bayes": MultinomialNB(alpha=0.001),
            "Linear SVM": LinearSVC(
                C=5.0,
                penalty="l1",
                dual=False,
                max_iter=2000,
                random_state=42
            )
        }
    else:
        classifiers = {
            "Naive Bayes": nb_module,
            "Linear SVM": svm_module
        }

    results = {}
    deltas = {}

    for clf_name, clf_mod in classifiers.items():
        print(f"\n  {clf_name}")

        if tuned:
            clf = clf_mod
            clf.fit(X_train, train_df["label"])
        else:
            clf = clf_mod.train(X_train, train_df["label"])

        results[clf_name] = {}
        deltas[clf_name] = {}

        for cond, (X, y) in splits.items():
            if len(y) == 0:
                print(f"    {cond}: partition is empty, skipping")
                results[clf_name][cond] = {
                    "accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
                    "TP": 0, "TN": 0, "FP": 0, "FN": 0
                }
                continue

            if tuned:
                y_pred = clf.predict(X)
            else:
                y_pred = clf_mod.predict(clf, X)
            
            m = met.evaluate(y, y_pred)
            results[clf_name][cond] = m
            print(f"    {cond:12s}  accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")

        for cond in ["temporal", "topic", "both"]:
            deltas[clf_name][cond] = met.delta(
                results[clf_name][cond],
                results[clf_name]["baseline"]
            )

    print()
    print("Step 5: Saving results...")
    df_results = met.build_results_table(results, deltas)
    filename = "tuned_results.csv" if tuned else "default_results.csv"

    met.save_results(
        df_results,
        output_cm=output_cm,
        filename=filename
    )

    print()
    print("RESULTS TABLE")
    print(df_results.to_string(index=False))
    print()
    print("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AMPETTS: Fake News Detection Under Topic and Temporal Shift"
    )

    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Run 5-fold stratified cross-validation before main evaluation."
    )
    
    parser.add_argument(
        "--confusion-matrices",
        action="store_true",
        help="Save plots of the confusion matrices for each model and condition combination."
    )

    # Allow tuned hyperparameters
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="Use the best hyperparameters from RQ2."
    )

    args = parser.parse_args()

    run(
        run_cv=args.cross_validation, output_cm=args.confusion_matrices, tuned=args.tuned
    )