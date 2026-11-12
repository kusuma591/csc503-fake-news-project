import os
import pandas as pd

RAW_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))

DS1_REAL = os.path.join(RAW_DIR, "True.csv")
DS1_FAKE = os.path.join(RAW_DIR, "Fake.csv")
DS2_FILE = os.path.join(RAW_DIR, "fake_news_dataset.csv")


def _load_dataset1():
    real = pd.read_csv(DS1_REAL)
    fake = pd.read_csv(DS1_FAKE)

    real["label"] = 0
    fake["label"] = 1

    df = pd.concat([real, fake], ignore_index=True)
    df["topic"] = "politics"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Any unparseable dates become NaT — fill them with a safe pre-2018 date
    # so those articles don't vanish when preprocess.py filters by date < 2018
    df["date"] = df["date"].fillna(pd.Timestamp("2016-06-01"))

    # Keep 80% as pre-2018 (baseline pool), reassign 20% to post-2018 (temporal shift)
    # Split real and fake separately so both classes appear in both partitions
    real_part = df[df["label"] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
    fake_part = df[df["label"] == 1].sample(frac=1, random_state=42).reset_index(drop=True)

    real_cut = int(len(real_part) * 0.80)
    fake_cut = int(len(fake_part) * 0.80)

    real_part.loc[real_cut:, "date"] = pd.Timestamp("2020-06-01")
    fake_part.loc[fake_cut:, "date"] = pd.Timestamp("2020-06-01")

    df = pd.concat([real_part, fake_part], ignore_index=True)

    return df[["title", "text", "label", "date", "topic"]]


def _load_dataset2():
    df = pd.read_csv(DS2_FILE)
    df.columns = df.columns.str.strip().str.lower()

    df = df[["title", "text", "label"]].copy()

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    label_col = df["label"].astype(str).str.strip().str.lower()
    df["label"] = label_col.map({"fake": 1, "real": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["topic"] = "world"

    # Split into topic shift (pre-2018) and both shifts (post-2018)
    # Do this per class so both conditions have fake and real articles
    real_part = df[df["label"] == 0].reset_index(drop=True)
    fake_part = df[df["label"] == 1].reset_index(drop=True)

    real_half = len(real_part) // 2
    fake_half = len(fake_part) // 2

    real_part["date"] = pd.Timestamp("2017-06-01")
    fake_part["date"] = pd.Timestamp("2017-06-01")

    real_part.iloc[real_half:, real_part.columns.get_loc("date")] = pd.Timestamp("2020-06-01")
    fake_part.iloc[fake_half:, fake_part.columns.get_loc("date")] = pd.Timestamp("2020-06-01")

    df = pd.concat([real_part, fake_part], ignore_index=True)

    return df[["title", "text", "label", "date", "topic"]]


def load_and_merge():
    df1 = _load_dataset1()
    df2 = _load_dataset2()

    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["text_combined"] = df["title"].fillna("") + " " + df["text"].fillna("")

    return df.reset_index(drop=True)
