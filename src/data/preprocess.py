import pandas as pd
from sklearn.model_selection import train_test_split

TRAINING_TOPIC = "politics"
CUTOFF_DATE = pd.Timestamp("2018-01-01")
TEST_SIZE = 0.20
RANDOM_STATE = 42


def partition(df):
    df = df.copy()

    baseline_pool = df[(df["topic"] == TRAINING_TOPIC) & (df["date"] < CUTOFF_DATE)]
    temporal = df[(df["topic"] == TRAINING_TOPIC) & (df["date"] >= CUTOFF_DATE)]
    topic = df[(df["topic"] != TRAINING_TOPIC) & (df["date"] < CUTOFF_DATE)]
    both = df[(df["topic"] != TRAINING_TOPIC) & (df["date"] >= CUTOFF_DATE)]

    train_df, test_df = train_test_split(
        baseline_pool,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=baseline_pool["label"]
    )

    return train_df, test_df, temporal, topic, both


def print_partition_summary(train_df, test_df, temporal, topic, both):
    partitions = [
        ("Train  (baseline train)", train_df),
        ("Test   (baseline test) ", test_df),
        ("Temporal shift         ", temporal),
        ("Topic shift            ", topic),
        ("Both shifts            ", both),
    ]
    for name, df in partitions:
        total = len(df)
        fake = df["label"].sum()
        real = total - fake
        print(f"  {name}  total={total:6d}  fake={fake:5d}  real={real:5d}")
