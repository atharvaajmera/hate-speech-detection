from __future__ import annotations

import argparse

from src.preprocess import LABEL_MAP, load_artifacts, predict_text, train_and_save_artifacts


def print_prediction(result: dict) -> None:
    print("\n=== Prediction ===")
    print(f"Input: {result['raw_text']}")
    print(f"Cleaned: {result['cleaned_text']}")
    print(f"Class ID: {result['class_id']}")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(
        "Per-class scores: "
        + ", ".join(
            f"{class_id}({LABEL_MAP[class_id]})={score:.4f}"
            for class_id, score in enumerate(result["scores"])
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict hate-speech class labels with the final submission SVC model."
    )
    parser.add_argument("text", nargs="*", help="Input text. Leave empty for interactive mode.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the TF-IDF vectorizer and LinearSVC before predicting.",
    )
    args = parser.parse_args()

    if args.retrain:
        vectorizer, model = train_and_save_artifacts()
    else:
        vectorizer, model = load_artifacts()

    text = " ".join(args.text).strip()
    if text:
        print_prediction(predict_text(model, vectorizer, text))
        return

    print("Interactive mode started. Type 'exit' or 'quit' to stop.")
    while True:
        user_text = input("\nEnter text: ").strip()
        if not user_text:
            print("Please enter some text, or type 'exit' to quit.")
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Exiting interactive mode.")
            break
        print_prediction(predict_text(model, vectorizer, user_text))


if __name__ == "__main__":
    main()
