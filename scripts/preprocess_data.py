from pathlib import Path

import pandas as pd

from tweet_cleaner import clean_tweet_column


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "labeled_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_labeled_data.csv"


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df = clean_tweet_column(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")
    print(f"Saved cleaned data to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
