from __future__ import annotations

import argparse
import pickle
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tweet_cleaner import clean_tweet


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORIZER_PATH = PROJECT_ROOT / "data" / "processed" / "tfidf" / "tfidf_vectorizer.pkl"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_labeled_data.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_metrics_summary.txt"
MODEL_DIR = PROJECT_ROOT / "models"

LABEL_MAP = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither",
}

MODEL_SPECS = {
    "logreg": {
        "display_name": "LogisticRegression (tfidf)",
        "cache_path": MODEL_DIR / "logreg_tfidf.joblib",
        "factory": lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    },
    "svc_tuned": {
        "display_name": "LinearSVC (optimized)",
        "cache_path": MODEL_DIR / "linear_svc_tuned.joblib",
        "factory": lambda: LinearSVC(
            C=0.5,
            class_weight="balanced",
            loss="hinge",
            max_iter=15000,
            random_state=42,
        ),
    },
    "mlp": {
        "display_name": "MLP (tfidf_optimized)",
        "cache_path": MODEL_DIR / "mlp_tfidf_optimized.joblib",
        "factory": lambda: Pipeline(
            steps=[
                ("svd", TruncatedSVD(n_components=200, random_state=42)),
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        random_state=42,
                    ),
                ),
            ]
        ),
    },
}


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def load_vectorizer():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Missing vectorizer at {VECTORIZER_PATH}. Run: python scripts/vectorize_tfidf.py"
        )
    with VECTORIZER_PATH.open("rb") as f:
        return pickle.load(f)


def load_training_data(vectorizer):
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing cleaned data at {TRAIN_DATA_PATH}. Run: python scripts/preprocess_data.py"
        )

    df = pd.read_csv(TRAIN_DATA_PATH)
    if "clean_tweet" not in df.columns or "class" not in df.columns:
        raise KeyError("Expected columns 'clean_tweet' and 'class' in cleaned dataset.")

    x = vectorizer.transform(df["clean_tweet"].fillna("").astype(str))
    y = df["class"].astype(int).to_numpy()
    return x, y


def load_or_train_models(vectorizer, retrain: bool):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    x, y = None, None
    loaded = {}

    for key, spec in MODEL_SPECS.items():
        cache_path = spec["cache_path"]
        if retrain and cache_path.exists():
            cache_path.unlink()

        if cache_path.exists():
            model = joblib.load(cache_path)
        else:
            if x is None or y is None:
                x, y = load_training_data(vectorizer)
            model = spec["factory"]()
            model.fit(x, y)
            joblib.dump(model, cache_path)

        loaded[key] = model

    return loaded


def parse_report_scores() -> pd.DataFrame | None:
    if not REPORT_PATH.exists():
        return None

    lines = REPORT_PATH.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Overall comparison"):
            start = i + 1
            break
    if start is None:
        return None

    table_lines = []
    for line in lines[start:]:
        if not line.strip():
            break
        table_lines.append(line)
    if not table_lines:
        return None

    try:
        df = pd.read_fwf(StringIO("\n".join(table_lines)))
        if "model" not in df.columns:
            return None
        return df
    except Exception:
        return None


def get_prediction_probs(model, x_vec: np.ndarray):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_vec)[0]
        return probs, float(np.max(probs))

    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_vec)
        scores = scores[0] if scores.ndim > 1 else np.array([scores[0]])
        if scores.shape[0] == 3:
            probs = softmax(scores)
            return probs, float(np.max(probs))

    probs = np.array([np.nan, np.nan, np.nan])
    return probs, float("nan")


def predict_all(models, vectorizer, text: str):
    cleaned = clean_tweet(text)
    x_vec = vectorizer.transform([cleaned])
    outputs = {}

    for key, model in models.items():
        pred_class = int(model.predict(x_vec)[0])
        probs, confidence = get_prediction_probs(model, x_vec)
        outputs[key] = {
            "pred_class": pred_class,
            "label": LABEL_MAP.get(pred_class, f"class_{pred_class}"),
            "confidence": confidence,
            "probs": probs,
        }
    return cleaned, outputs


def print_prediction_block(
    input_text: str,
    cleaned: str,
    outputs: dict,
    score_df: pd.DataFrame | None,
) -> None:
    print("\n=== Input ===")
    print(f"Raw: {input_text}")
    print(f"Cleaned: {cleaned}")

    for key, pred in outputs.items():
        display_name = MODEL_SPECS[key]["display_name"]
        print(f"\n--- {display_name} ---")
        print(f"Prediction: {pred['label']} (class {pred['pred_class']})")
        if not np.isnan(pred["confidence"]):
            probs = pred["probs"]
            print(f"Confidence: {pred['confidence']:.4f}")
            print(
                "Per-class scores: "
                f"0(hate)={probs[0]:.4f}, "
                f"1(offensive)={probs[1]:.4f}, "
                f"2(neither)={probs[2]:.4f}"
            )

        if score_df is not None:
            # Try exact match first; then fallback aliases for older report names.
            row = score_df[score_df["model"] == display_name]
            if row.empty:
                aliases = {
                    "LogisticRegression (tfidf)": ["LogisticRegression"],
                    "LinearSVC (optimized)": ["LinearSVC (tuned)"],
                    "MLP (tfidf_optimized)": ["MLP (best from notebook)"],
                }
                for alias in aliases.get(display_name, []):
                    row = score_df[score_df["model"] == alias]
                    if not row.empty:
                        break
            if not row.empty:
                r = row.iloc[0]
                print(
                    "Eval metrics: "
                    f"accuracy={r.get('accuracy', np.nan):.4f}, "
                    f"f1_macro={r.get('f1_macro', np.nan):.4f}, "
                    f"class0_f1={r.get('class0_f1', np.nan):.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI hate-speech prediction with current optimized classical models."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to classify. If omitted, interactive loop starts.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining and overwrite cached models.",
    )
    args = parser.parse_args()

    vectorizer = load_vectorizer()
    models = load_or_train_models(vectorizer, retrain=args.retrain)
    score_df = parse_report_scores()

    raw_text = " ".join(args.text).strip()

    def run_once(text: str) -> None:
        cleaned, outputs = predict_all(models, vectorizer, text)
        print_prediction_block(text, cleaned, outputs, score_df)

    if raw_text:
        run_once(raw_text)
        return

    print("Interactive mode started. Type 'exit' (or 'quit') to stop.")
    while True:
        user_text = input("\nEnter text: ").strip()
        if not user_text:
            print("Please enter some text, or type 'exit' to quit.")
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Exiting interactive mode.")
            break
        run_once(user_text)


if __name__ == "__main__":
    main()
