from pathlib import Path
import pickle

import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_labeled_data.csv"
TFIDF_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tfidf"


def main() -> None:
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{CLEANED_DATA_PATH} not found. Run preprocess_data.py first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    if "clean_tweet" not in df.columns:
        raise KeyError("Expected column 'clean_tweet' in cleaned dataset.")

    texts = df["clean_tweet"].fillna("").astype(str)
    labels = df["class"] if "class" in df.columns else None

    vectorizer = TfidfVectorizer(
        lowercase=False,  # text is already normalized during preprocessing
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    x_tfidf = vectorizer.fit_transform(texts)

    TFIDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(TFIDF_OUTPUT_DIR / "X_tfidf.npz", x_tfidf)

    with (TFIDF_OUTPUT_DIR / "tfidf_vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    if labels is not None:
        labels.to_csv(TFIDF_OUTPUT_DIR / "y_labels.csv", index=False, header=["class"])

    print(f"Loaded cleaned rows: {len(df)}")
    print(f"TF-IDF matrix shape: {x_tfidf.shape}")
    print(f"Saved matrix: {TFIDF_OUTPUT_DIR / 'X_tfidf.npz'}")
    print(f"Saved vectorizer: {TFIDF_OUTPUT_DIR / 'tfidf_vectorizer.pkl'}")
    if labels is not None:
        print(f"Saved labels: {TFIDF_OUTPUT_DIR / 'y_labels.csv'}")


if __name__ == "__main__":
    main()
