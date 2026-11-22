# Fake News Detection Under Topic and Temporal Shift

**CSC 503 / SENG 474 - Data Mining | University of Victoria | Fall 2026**

Team: Olivia Eskes · Jay Robertson · Markus Sekhon · Kusuma Mattipa

---

## Overview

Machine learning models for fake news detection are typically trained on a fixed topic and time window. This project evaluates how much classification accuracy degrades when that distribution shifts — in topic, in time, or both.

We implement the **AMPETTS** framework (Algorithm for Model Performance Evaluation under Topic, Temporal, and combined Shift) using:

- **Naive Bayes** (Multinomial, α = 1; tuned α = 0.001)
- **Linear SVM** (C = 1, LinearSVC; tuned C = 5, L1 penalty)
- **TF-IDF** text representation (fit on training data only)

Each classifier is evaluated under four conditions:

| Condition | Topic | Time period |
| -------------- | ------------------------- | ----------- |
| Baseline | Politics (training topic) | Pre-2018 |
| Temporal shift | Politics | Post-2018 |
| Topic shift | World news | Pre-2018 |
| Both shifts | World news | Post-2018 |

In addition to these four evaluation conditions, we implement 5-fold stratified cross-validation to assess the consistency and generalization performance of both classifiers on the training data distribution.

---

## Illustrative Sample

```text
Training data: politics articles, pre-2018
   → TF-IDF vectorizer fit here

Evaluation:
  Baseline  (same topic, same time)  → expected high accuracy
  Temporal  (same topic, newer time) → vocabulary drift
  Topic     (new topic, same time)   → different terminology
  Both      (new topic, newer time)  → maximum degradation expected
```

---

## Prerequisites

- Python 3.9+
- pip

```bash
pip install -r requirements.txt
```

**requirements.txt** installs: `scikit-learn`, `pandas`, `numpy`, `joblib`

For generating plots, also install `matplotlib`:

```bash
pip install matplotlib
```

---

## Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/kusuma591/csc503-fake-news-project.git
cd csc503-fake-news-project/3_Solution

# 2. Install dependencies
pip install -r requirements.txt
pip install matplotlib
```

---

## Quick Start (no data download needed)

The quickstart runs on **synthetic demo data** so you can verify the pipeline works immediately:

```bash
python quickstart.py
```

Expected output: partition summary table → TF-IDF fit → results table → CSV saved to `evaluation/results/quickstart_results.csv`

---

## Running on Real Data


### Step 1 - Download datasets (only needed if cloning from GitHub)

**Training data**:
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Download and place `True.csv` and `Fake.csv` in `data/raw/`

**Testing data**:
https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset

Download and place `fake_news_dataset.csv` in `data/raw/`

### Step 2 - Run the full pipeline

```bash
python src/main.py
```

Or use the quickstart with real data:

```bash
python quickstart.py --real
```

The corrected dataset contains 64,898 articles, partitioned as follows:

| Partition | Articles |
|---|---:|
| Training | 19,313 |
| Baseline test | 4,829 |
| Temporal shift | 2,802 |
| Topic shift | 20,756 |
| Both shifts | 17,198 |

### Command line arguments

`--cross-validation` : Run 5-fold stratified cross-validation before main evaluation.

`--confusion-matrices` : Save plots of the confusion matrices for each model and condition combination.

`--tuned` : Run the classifiers using the best hyperparameters selected in the RQ2 study.

**Full example**:

```bash
python src/main.py --cross-validation --confusion-matrices
```

**Run tuned models**:

```bash
python src/main.py --tuned
```

### Hyperparameter Study (RQ2)

The hyperparameter study uses 5-fold stratified cross-validation to evaluate:

- **SVM:** C = 0.001, 0.1, 0.5, 1, 5, with L1 and L2 penalties.
- **Naive Bayes:** α = 0.001, 0.1, 0.5, 1, 5.

Run the study:

```bash
python src/hyperparameter_study.py
```

Results are saved to `evaluation/results/hyperparameter_results.csv`.

Generate the hyperparameter comparison graph:

```bash
python src/plot_hyperparameters.py
```

The graph is saved to `evaluation/results/rq2_hyperparameters_combined.pdf`.

The best configurations were SVM with C = 5 and L1 regularization (mean F1 = 0.9982) and Naive Bayes with α = 0.001 (mean F1 = 0.9573).

---

## Results

Output is written to `evaluation/results/default_results.csv` for the original configurations or `evaluation/results/tuned_results.csv` when using `--tuned`.

Additional results are saved to:

- `evaluation/results/cross_validation_results.csv` — 5-fold cross-validation results.
- `evaluation/results/hyperparameter_results.csv` — RQ2 hyperparameter results.
- `evaluation/results/rq2_hyperparameters_combined.pdf` — RQ2 comparison graph.

Example columns:

| Classifier | Condition | Accuracy | Precision | Recall | F1 | Δ Accuracy | Δ F1 |
| ----------- | --------- | -------- | --------- | ------ | ---- | ---------- | ----- |
| Naive Bayes | baseline | 0.93 | 0.94 | 0.92 | 0.93 | - | - |
| Naive Bayes | temporal | 0.78 | 0.80 | 0.76 | 0.78 | -0.15 | -0.15 |
| ... | ... | ... | ... | ... | ... | ... | ... |

*(Actual values depend on dataset version and random seed.)*

---

## Reproducibility

All random seeds are fixed at the top of `src/main.py` and `quickstart.py`:

```python
random.seed(42)
numpy.random.seed(42)
```

`train_test_split` and `LinearSVC` also use `random_state=42`. Given the same input data and these settings, results will be identical across runs.

The hyperparameter study uses 5-fold stratified cross-validation with `random_state=2026`, and fits TF-IDF separately within each training fold.

---

## Project Structure

```text
csc503-fake-news-project/
└── 3_Solution/
    ├── README.md
    ├── requirements.txt
    ├── quickstart.py                ← demo run, no downloads required
    ├── data/
    │   ├── raw/                     ← place downloaded CSV files here
    │   └── preprocessed/            ← generated partitions (auto-created)
    ├── output/
    │   └── models/                  ← saved vectorizer + model .pkl files
    ├── evaluation/
    │   └── results/                 ← output CSVs with metrics per condition
    └── src/
        ├── main.py                  ← full pipeline entry point
        ├── hyperparameter_study.py  ← RQ2 hyperparameter study
        ├── plot_hyperparameters.py  ← RQ2 comparison graph
        ├── data/
        │   ├── load.py              ← reads and merges both datasets
        │   └── preprocess.py        ← partitions into 4 conditions
        ├── features/
        │   └── tfidf.py            
