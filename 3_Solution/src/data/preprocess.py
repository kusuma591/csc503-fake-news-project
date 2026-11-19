import pandas as pd
from sklearn.model_selection import train_test_split

TRAINING_TOPIC = "politics"
TIME_THRESHOLD = pd.DateOffset(years=4)
TEST_SIZE = 0.20
RANDOM_STATE = 42


def partition(df):
    df = df.copy()

    baseline_pool = df[
        (df["topic"] == TRAINING_TOPIC) &
        (df["source_dataset"] == "dataset1")
    ]
    
    train_df, test_df = train_test_split(
        baseline_pool,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=baseline_pool["label"]
    )
    
    # Find the allowed dates for the testing sets
    earliest_train_date = train_df["date"].min()
    latest_train_date = train_df["date"].max()
    no_shift_start = latest_train_date - TIME_THRESHOLD
    no_shift_end = earliest_train_date + TIME_THRESHOLD

    future_shift_start = latest_train_date + TIME_THRESHOLD
    
    dataset1 = df[df["source_dataset"] == "dataset1"]
    dataset2 = df[df["source_dataset"] == "dataset2"]    
    
    temporal = dataset2[
    (dataset2["topic"] == TRAINING_TOPIC) &
    (dataset2["date"] > future_shift_start)
    ]
    
    topic = dataset1[
        (dataset1["topic"] != TRAINING_TOPIC) &
        (dataset1["date"] >= no_shift_start) &
        (dataset1["date"] <= no_shift_end)
    ]
    
    both = dataset2[
        (dataset2["topic"] != TRAINING_TOPIC) &
        (dataset2["date"] > future_shift_start)
    ]

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
