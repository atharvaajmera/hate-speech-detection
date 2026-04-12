# Hate Speech Project

This project builds a hate-speech classification pipeline for short social-media text. The final system uses text cleaning, TF-IDF vectorization, and a tuned `LinearSVC` classifier to predict one of three classes:

- `0`: `hate_speech`
- `1`: `offensive_language`
- `2`: `neither`

The repository is organized around one final submission notebook, one shared preprocessing module, and two runnable entrypoints:

- `predict.py` for command-line inference
- `app.py` for a Streamlit interface

## Project overview

The work in this repo follows a practical machine-learning workflow:

1. clean noisy tweet text by removing URLs, mentions, `RT`, punctuation, and extra whitespace
2. convert cleaned text into TF-IDF features
3. train a tuned `LinearSVC` model with `class_weight="balanced"`
4. evaluate with special attention to the minority `hate_speech` class
5. save the trained vectorizer and classifier as reusable `.pkl` artifacts

The final notebook, [`Hate_Speech_Analysis.ipynb`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\Hate_Speech_Analysis.ipynb), is the main submission path. Older experiments are preserved separately under `notebooks/experiments/`.

## Current folder structure

```text
hate-speech-detection/
|-- Hate_Speech_Analysis.ipynb
|-- app.py
|-- predict.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw_dataset.csv
|   |-- raw/
|   |   `-- labeled_data.csv
|   `-- processed/
|       |-- cleaned_labeled_data.csv
|       `-- tfidf/
|           |-- X_tfidf.npz
|           |-- tfidf_vectorizer.pkl
|           `-- y_labels.csv
|-- models/
|   |-- tfidf_vectorizer.pkl
|   `-- hate_speech_svc.pkl
|-- notebooks/
|   `-- experiments/
|       |-- 01_logreg_tfidf.ipynb
|       |-- 02_svm_optimized.ipynb
|       |-- 03_mlp_tfidf_optimized.ipynb
|       `-- 04_cnn_embeddings.ipynb
`-- src/
    |-- __init__.py
    |-- preprocess.py
    `-- export_model.py
```

## What each part does

- [`Hate_Speech_Analysis.ipynb`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\Hate_Speech_Analysis.ipynb): final EDA + preprocessing + training + conclusion notebook
- [`src/preprocess.py`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\src\preprocess.py): shared cleaning, dataset loading, training, artifact saving, and prediction helpers
- [`src/export_model.py`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\src\export_model.py): exports the final vectorizer and classifier into `models/`
- [`predict.py`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\predict.py): CLI entrypoint for inference
- [`app.py`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\app.py): Streamlit app for interactive inference
- [`notebooks/experiments`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\notebooks\experiments): archived model-development attempts

## Requirements

The single source of truth for dependencies is [`requirements.txt`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\requirements.txt).

The current environment uses:

- `numpy`
- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `streamlit`
- `tensorflow-cpu`

If the dependency list changes later, follow `requirements.txt` rather than this README summary.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run commands directly with the environment Python:

```powershell
.\.venv\Scripts\python.exe --version
```

## Run instructions

### 1. Open the final notebook

Use [`Hate_Speech_Analysis.ipynb`](c:\Users\Atharva\OneDrive\Desktop\WebD\hate-speech-detection\Hate_Speech_Analysis.ipynb) for the full project walkthrough:

- clean EDA
- preprocessing pipeline
- TF-IDF setup
- tuned `LinearSVC` training
- classification report with focus on Class 0 F1-score
- ablation summary versus deep-learning attempts

### 2. Export the final model artifacts

To regenerate the reusable model files:

```powershell
.\.venv\Scripts\python.exe src/export_model.py
```

This creates:

```text
models/tfidf_vectorizer.pkl
models/hate_speech_svc.pkl
```

### 3. Run CLI inference

Interactive mode:

```powershell
.\.venv\Scripts\python.exe predict.py
```

One-shot mode:

```powershell
.\.venv\Scripts\python.exe predict.py "your text here"
```

Retrain before inference:

```powershell
.\.venv\Scripts\python.exe predict.py --retrain
```

### 4. Run the Streamlit app

```powershell
streamlit run app.py
```

The app will load the saved `.pkl` artifacts if they exist. If not, it can retrain them from the raw dataset.

## Notes

- `data/raw_dataset.csv` is the primary raw dataset path used by the cleaned codebase.
- `data/raw/labeled_data.csv` is retained as the original source copy.
- `models/` may be empty until you run the notebook, `src/export_model.py`, or `predict.py --retrain`.
- `notebooks/experiments/` is intentionally preserved to show the model-development journey, but it is not part of the final submission path.
