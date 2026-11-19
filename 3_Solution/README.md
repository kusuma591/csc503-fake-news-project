# Fake News Detection Under Topic and Temporal Shift

**CSC 503 / SENG 474 — Data Mining | University of Victoria | Fall 2026**

Team: Olivia Eskes · Jay Robertson · Markus Sekhon · Kusuma Mattipa

---

## Overview

Machine learning models for fake news detection are typically trained on a fixed topic and time window. This project evaluates how much classification accuracy degrades when that distribution shifts — in topic, in time, or both.

We implement the **AMPETTS** framework (Algorithm for Model Performance Evaluation under Topic, Temporal, and combined Shift) using:

- **Naive Bayes** (Multinomial, α = 1)
- **Linear SVM** (C = 1, LinearSVC)
- **TF-IDF** text representation (fit on training data only)

Each classifier is evaluated under four conditions:

| Condition      | Topic                     | Time period |
| -------------- | ------------------------- | ----------- |
| Baseline       | Politics (training topic) | Pre-2018    |
| Temporal shift | Politics                  | Post-2018   |
| Topic shift    | World news                | Pre-2018    |
| Both shifts    | World news                | Post-2018   |

---

## Illustrative Sample

```
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

---

## Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/kusuma591/csc503-fake-news-project.git
cd csc503-fake-news-project/3_Solution

# 2. Install dependencies
pip install -r requirements.txt
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


### Step 1 — Download datasets (only needed if cloning from GitHub)

**Training data**:
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

Download and place `True.csv` and `Fake.csv` in `data/raw/`

**Testing data**:
https://www.kaggle.com/datasets/mahdimashayekhi/fake-news-detection-dataset

Download and place `fake_news_dataset.csv` in `data/raw/`

### Step 2 — Run the full pipeline

```bash
python src/main.py
```

Or use the quickstart with real data:

```bash
python quickstart.py --real
```

---

## Results

Output is written to `evaluation/results/results.csv`.

Example columns:

| Classifier  | Condition | Accuracy | Precision | Recall | F1   | Δ Accuracy | Δ F1  |
| ----------- | --------- | -------- | --------- | ------ | ---- | ---------- | ----- |
| Naive Bayes | baseline  | 0.93     | 0.94      | 0.92   | 0.93 | -          | -     |
| Naive Bayes | temporal  | 0.78     | 0.80      | 0.76   | 0.78 | -0.15      | -0.15 |
| ...         | ...       | ...      | ...       | ...    | ...  | ...        | ...   |

_(Actual values depend on dataset version and random seed.)_

---

## Reproducibility

All random seeds are fixed at the top of `src/main.py` and `quickstart.py`:

```python
random.seed(42)
numpy.random.seed(42)
```

`train_test_split` and `LinearSVC` also use `random_state=42`. Given the same input data and these settings, results will be identical across runs.

---

## Project Structure

```
csc503-fake-news-project/
├── README.md
├── requirements.txt
├── quickstart.py          ← demo run, no downloads required
├── data/
│   ├── raw/               ← place downloaded CSV files here
│   └── preprocessed/      ← generated partitions (auto-created)
├── output/
│   └── models/            ← saved vectorizer + model .pkl files
├── evaluation/
│   └── results/           ← output CSVs with metrics per condition
└── src/
    ├── main.py            ← full pipeline entry point
    ├── data/
    │   ├── load.py        ← reads and merges both datasets
    │   └── preprocess.py  ← partitions into 4 conditions
    ├── features/
    │   └── tfidf.py       ← TF-IDF fit / transform
    ├── models/
    │   ├── naive_bayes.py ← Multinomial NB wrapper
    │   └── svm.py         ← Linear SVM wrapper
    └── evaluation/
        └── metrics.py     ← accuracy / precision / recall / F1 / delta
```

---

## Citation

If you build on this work, please cite:

```
Eskes O., Robertson J., Sekhon M., Mattipa K. (2026).
Fake News Detection Capabilities Over Topic and Time Shift.
CSC 503, University of Victoria.
```

---

## License

For academic use only (CSC 503 course project). Not for redistribution.
