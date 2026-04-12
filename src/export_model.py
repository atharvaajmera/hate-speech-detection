from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    # Allow running as `python src/export_model.py` from project root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import MODEL_PATH, VECTORIZER_PATH, train_and_save_artifacts


def main() -> None:
    train_and_save_artifacts()
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
