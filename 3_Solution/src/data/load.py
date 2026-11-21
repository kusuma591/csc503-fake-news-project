import os
import pandas as pd

RAW_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))

DS1_REAL = os.path.join(RAW_DIR, "True.csv")
DS1_FAKE = os.path.join(RAW_DIR, "Fake.csv")
DS2_FILE = os.path.join(RAW_DIR, "fake_news_dataset.csv")

def _normalize_topic(value):
    if pd.isna(value):
        return "world"
    topic = str(value).strip().lower()
    if any(kw in topic for kw in ("politic", "government", "left-news")):
        return "politics"
    return "world"


def _load_dataset1():
    real = pd.read_csv(DS1_REAL)
    fake = pd.read_csv(DS1_FAKE)

    real["label"] = 0
    fake["label"] = 1
    
    # Parse dates individually to handle mixed formats
    real["date"] = pd.to_datetime(
        real["date"].astype(str).str.strip(),
        format="mixed",
        errors="coerce"
    )

    fake["date"] = pd.to_datetime(
        fake["date"].astype(str).str.strip(),
        format="mixed",
        errors="coerce"
    )    

    df = pd.concat([real, fake], ignore_index=True) 
    df["topic"] = df["subject"].apply(_normalize_topic)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Fill in missing text with ""
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)    
    
    df["source_dataset"] = "dataset1"    
    
    return df[["title", "text", "label", "date", "topic", "source_dataset"]]


def _load_dataset2():
    df = pd.read_csv(DS2_FILE)
    df.columns = df.columns.str.strip().str.lower()
    
    # Normalize the labelling
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"fake": 1, "real": 0, "1": 1, "0": 0})
    )
    df["topic"] = df["category"].apply(_normalize_topic)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Fill in missing text with ""
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    
    df["source_dataset"] = "dataset2"

    return df[["title", "text", "label", "date", "topic", "source_dataset"]]


def load_and_merge():
    df1 = _load_dataset1()
    df2 = _load_dataset2()

    df = pd.concat([df1, df2], ignore_index=True)
    df["text_combined"] = (
        df["title"].fillna("") + " " + df["text"].fillna("")
    ).str.strip()

    return df.reset_index(drop=True)
