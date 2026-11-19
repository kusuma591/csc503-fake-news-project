import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from features import tfidf
from models import naive_bayes as nb_module
from models import svm as svm_module
from evaluation import metrics as met


def run_cross_validation(train_df, n_splits=5):
    X = train_df["text_combined"]
    y = train_df["label"]

    # Create folds once so both classifiers use identical splits
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=2026
    )

    folds = list(skf.split(X, y))

    classifiers = {
        "Naive Bayes": nb_module,
        "Linear SVM": svm_module
    }

    all_results = {}

    for clf_name, clf_mod in classifiers.items():
        print(f"\n{'=' * 65}")
        print(f"MODEL: {clf_name}")
        print(f"{'=' * 65}")

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            # Split original training data
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Fit TF-IDF using only training fold
            vec = tfidf.fit(X_train, save=False)

            X_train_tfidf = tfidf.transform(vec, X_train)
            X_test_tfidf = tfidf.transform(vec, X_test)

            # Train classifier
            clf = clf_mod.train(
                X_train_tfidf,
                y_train,
                save=False
            )

            # Predict and classify testing fold
            y_pred = clf_mod.predict(clf, X_test_tfidf)
            m = met.evaluate(y_test, y_pred)

            fold_results.append(m)

            # Display results
            print(f"\n{'-' * 50}")
            print(f"FOLD {fold}")
            print(f"{'-' * 50}")

            print(f"Training observations: {len(train_idx)}")
            print(f"Testing observations:  {len(test_idx)}")

            print(
                f"TP = {m['TP']}, FP = {m['FP']}"
            )
            print(
                f"FN = {m['FN']}, TN = {m['TN']}"
            )

            print(f"Accuracy  = {m['accuracy']:.4f}")
            print(f"Precision = {m['precision']:.4f}")
            print(f"Recall    = {m['recall']:.4f}")
            print(f"F1        = {m['f1']:.4f}")

        # Calculate mean performance across folds
        mean_metrics = {}

        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean_metrics[metric] = np.mean([
                result[metric] for result in fold_results
            ])

        all_results[clf_name] = {
            "folds": fold_results,
            "mean": mean_metrics
        }

        print(f"\n{'=' * 50}")
        print(f"{clf_name} CROSS-VALIDATION SUMMARY")
        print(f"{'=' * 50}")

        print(
            f"{'Fold':<12}"
            f"{'Accuracy':>12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1':>12}"
        )

        for i, m in enumerate(fold_results, start=1):
            print(
                f"{i:<12}"
                f"{m['accuracy']:>12.4f}"
                f"{m['precision']:>12.4f}"
                f"{m['recall']:>12.4f}"
                f"{m['f1']:>12.4f}"
            )

        print("-" * 60)

        print(
            f"{'Mean':<12}"
            f"{mean_metrics['accuracy']:>12.4f}"
            f"{mean_metrics['precision']:>12.4f}"
            f"{mean_metrics['recall']:>12.4f}"
            f"{mean_metrics['f1']:>12.4f}"
        )

    # Compare the two classifiers
    print("\n" + "=" * 65)
    print("CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 65)

    print(
        f"{'Classifier':<18}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    for clf_name, result in all_results.items():

        m = result["mean"]

        print(
            f"{clf_name:<18}"
            f"{m['accuracy']:>12.4f}"
            f"{m['precision']:>12.4f}"
            f"{m['recall']:>12.4f}"
            f"{m['f1']:>12.4f}"
        )

    # Save results
    rows = []

    for clf_name, result in all_results.items():

        for fold, m in enumerate(result["folds"], start=1):

            rows.append({
                "Classifier": clf_name,
                "Fold": fold,
                **m
            })

        rows.append({
            "Classifier": clf_name,
            "Fold": "Mean",
            **result["mean"]
        })

    results_df = pd.DataFrame(rows)

    os.makedirs(met.RESULTS_DIR, exist_ok=True)

    path = os.path.join(
        met.RESULTS_DIR,
        "cross_validation_results.csv"
    )

    results_df.to_csv(path, index=False)

    print(f"\nCross-validation results saved to {path}\n")

    return all_results
