from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw_dataset.csv"
LEGACY_RAW_DATA_PATH = DATA_DIR / "raw" / "labeled_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "hate_speech_svc.joblib"

LABEL_MAP = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
RT_TAG_RE = re.compile(r"\bRT\b", flags=re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTISPACE_RE = re.compile(r"\s+")


def resolve_raw_data_path() -> Path:
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    if LEGACY_RAW_DATA_PATH.exists():
        return LEGACY_RAW_DATA_PATH
    raise FileNotFoundError(
        "Raw dataset not found. Expected either "
        f"{RAW_DATA_PATH} or {LEGACY_RAW_DATA_PATH}."
    )


def clean_text(text: str) -> str:
    cleaned = str(text)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = RT_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("#", "")
    cleaned = NON_ALPHA_RE.sub(" ", cleaned)
    cleaned = MULTISPACE_RE.sub(" ", cleaned).strip().lower()
    return cleaned


def load_raw_dataset() -> pd.DataFrame:
    df = pd.read_csv(resolve_raw_data_path())
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    required_columns = {"tweet", "class"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return df


def build_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["tweet"] = cleaned["tweet"].fillna("").astype(str)
    cleaned["class"] = cleaned["class"].astype(int)
    cleaned["clean_tweet"] = cleaned["tweet"].apply(clean_text)
    return cleaned


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )


def build_model() -> LinearSVC:
    return LinearSVC(
        C=0.5,
        class_weight="balanced",
        loss="hinge",
        max_iter=15000,
        random_state=42,
    )


def train_and_save_artifacts() -> tuple[TfidfVectorizer, LinearSVC]:
    df = build_clean_dataframe(load_raw_dataset())
    vectorizer = build_vectorizer()
    x = vectorizer.fit_transform(df["clean_tweet"])
    y = df["class"].to_numpy()

    model = build_model()
    model.fit(x, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    return vectorizer, model


def load_artifacts() -> tuple[TfidfVectorizer, LinearSVC]:
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        return train_and_save_artifacts()
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


def pseudo_probabilities(model: LinearSVC, text_vector):
    scores = model.decision_function(text_vector)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    row = scores[0]
    shifted = row - row.max()
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    return probs


def predict_text(model: LinearSVC, vectorizer: TfidfVectorizer, text: str) -> dict:
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred_class = int(model.predict(features)[0])
    probs = pseudo_probabilities(model, features)
    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "class_id": pred_class,
        "label": LABEL_MAP.get(pred_class, str(pred_class)),
        "scores": probs,
        "confidence": float(probs.max()),
    }
