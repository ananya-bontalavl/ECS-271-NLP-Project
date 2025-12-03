import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Config
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

DATA_ROOT = "data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'\"()\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load Dataset From Kaggle
def load_bbc_dataset():
    print("Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("pariza/bbc-news-summary")

    articles_root = os.path.join(dataset_path, "BBC News Summary", "News Articles")
    summaries_root = os.path.join(dataset_path, "BBC News Summary", "Summaries")

    data_rows = []

    for category in os.listdir(articles_root):
        cat_article_dir = os.path.join(articles_root, category)
        cat_summary_dir = os.path.join(summaries_root, category)

        if not os.path.isdir(cat_article_dir):
            continue

        for filename in os.listdir(cat_article_dir):
            if filename.endswith(".txt"):
                article_path = os.path.join(cat_article_dir, filename)
                summary_path = os.path.join(cat_summary_dir, filename)

                if not os.path.exists(summary_path):
                    continue

                with open(article_path, "r", encoding="latin-1") as f:
                    article = f.read().strip()

                with open(summary_path, "r", encoding="latin-1") as f:
                    summary = f.read().strip()

                data_rows.append([category, article, summary])

    df = pd.DataFrame(data_rows, columns=["category", "text", "summary"])
    return df


# Sentiment setup (similar to your reference)
SENTIMENT_MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def add_sentiment(df):
    texts = df["text"].tolist()
    sentiments = sentiment_pipeline(texts, batch_size=16, truncation=True)

    sentiment_labels = [s["label"] for s in sentiments]
    sentiment_scores = [s["score"] for s in sentiments]

    df["sentiment_label"] = sentiment_labels
    df["sentiment_score"] = sentiment_scores

    return df


# Main Preprocessing Pipeline
def preprocess_and_save():
    df = load_bbc_dataset()

    df = df.dropna(subset=["text", "summary"])

    df["text"] = df["text"].apply(clean_text)
    df["summary"] = df["summary"].apply(clean_text)

    df = df[(df["text"].str.len() > 0) & (df["summary"].str.len() > 0)]

    #  NEW: add sentiment columns based on "text"
    print("Running sentiment analysis on full dataset...")
    df = add_sentiment(df)

    raw_csv_path = os.path.join(RAW_DIR, "bbc_full.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"Saved raw cleaned dataset with sentiment to {raw_csv_path}")

    # Split Train / Val / Test (same as before)
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["category"],
    )

    val_relative = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_relative,
        random_state=RANDOM_SEED,
        stratify=temp_df["category"],
    )

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print("Saved processed splits inside data/processed/")


# Dataset Inspection (Sanity Check)
def inspect_split(path, name):
    df = pd.read_csv(path)

    print(f"\n===== {name.upper()} ({path}) =====")
    print("Shape:", df.shape)
    print(df.head(3))

    print("\nNull values:")
    print(df.isna().sum())

    print("\nColumns:", list(df.columns))

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nText length stats:")
    print(df["text"].str.len().describe())

    print("\nSummary length stats:")
    print(df["summary"].str.len().describe())

    if "sentiment_label" in df.columns:
        print("\nSentiment label counts:")
        print(df["sentiment_label"].value_counts())


def inspect_all():
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = os.path.join(PROCESSED_DIR, "val.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test.csv")

    inspect_split(train_path, "train")
    inspect_split(val_path, "val")
    inspect_split(test_path, "test")


if __name__ == "__main__":
    preprocess_and_save()
    inspect_all()
