
import os
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from data.load import load_and_merge
from data.preprocess import partition


# Load datasets and select training data
print("Loading datasets...", flush=True)

df = load_and_merge()
train_df, test_df, temporal, topic, both = partition(df)

X = train_df["text_combined"]
y = train_df["label"]

print(f"Training articles: {len(X)}", flush=True)


# Five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=2026
)


# Hyperparameters to test
models = {
    "SVM": (
        LinearSVC(
            max_iter=2000,
            random_state=42,
            dual=False
        ),
        {
            "model__C": [0.001, 0.1, 0.5, 1.0, 5.0],
            "model__penalty": ["l1", "l2"]
        }
    ),
    "Naive Bayes": (
        MultinomialNB(),
        {
            "model__alpha": [0.001, 0.1, 0.5, 1.0, 5.0]
        }
    )
}


results = []

for name, (classifier, parameters) in models.items():

    print(f"\nTesting {name}...", flush=True)

    # Same TF-IDF settings as the original project
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
            ngram_range=(1, 2)
        )),
        ("model", classifier)
    ])

    # Test all hyperparameter combinations
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=False,
        verbose=2,
        error_score="raise"
    )

    search.fit(X, y)

    print(f"\n{name} RESULTS")
    print("Best parameters:", search.best_params_)
    print("Best mean F1:", round(search.best_score_, 4))

    # Record all configurations
    for params, mean, std in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"]
    ):
        results.append({
            "Model": name,
            "C": params.get("model__C", ""),
            "Penalty": params.get("model__penalty", ""),
            "Alpha": params.get("model__alpha", ""),
            "Mean_F1": round(mean, 4),
            "Std_F1": round(std, 4)
        })


# Save results
results_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "evaluation",
    "results"
)

os.makedirs(results_dir, exist_ok=True)

results_df = pd.DataFrame(results)

output_path = os.path.join(
    results_dir,
    "hyperparameter_results.csv"
)

results_df.to_csv(output_path, index=False)


print("\nFINAL HYPERPARAMETER RESULTS")
print(results_df.to_string(index=False))

print(f"\nResults saved to: {os.path.abspath(output_path)}")
