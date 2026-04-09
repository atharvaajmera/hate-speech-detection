# Hate Speech Detection

Hate speech classification project using cleaned tweet text, TF-IDF features, and optimized classical models.

## Current scope

- Data preprocessing (`tweet` -> `clean_tweet`)
- TF-IDF feature generation
- Model notebooks for EDA, Logistic Regression, optimized SVM, and optimized MLP
- CLI prediction script that compares multiple trained models on input text

## Project structure

```text
hate-speech-detection/
|-- data/
|   |-- raw/
|   |   `-- labeled_data.csv
|   `-- processed/
|       |-- cleaned_labeled_data.csv
|       `-- tfidf/
|-- models/
|-- notebooks/
|   |-- cnn_embeddings.ipynb
|   |-- eda.ipynb
|   |-- logreg_tfidf.ipynb
|   |-- mlp_tfidf_optimized.ipynb
|   `-- svm_optimized.ipynb
|-- reports/
|   `-- model_metrics_summary.txt
|-- scripts/
|   |-- predict.py
|   |-- preprocess_data.py
|   |-- tweet_cleaner.py
|   `-- vectorize_tfidf.py
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run preprocessing

```powershell
python scripts/preprocess_data.py
```

This will create:

```text
data/processed/cleaned_labeled_data.csv
```

## Build TF-IDF features

```powershell
python scripts/vectorize_tfidf.py
```

This will create:

```text
data/processed/tfidf/X_tfidf.npz
data/processed/tfidf/tfidf_vectorizer.pkl
data/processed/tfidf/y_labels.csv
```

## Run CLI prediction (multi-model)

```powershell
python scripts/predict.py
```

One-shot mode:

```powershell
python scripts/predict.py "your text here"
```

Force retraining cached models:

```powershell
python scripts/predict.py --retrain
```

## Notebooks

- `notebooks/eda.ipynb`: detailed exploratory data analysis
- `notebooks/logreg_tfidf.ipynb`: logistic regression baseline on TF-IDF
- `notebooks/svm_optimized.ipynb`: optimized LinearSVC workflow
- `notebooks/mlp_tfidf_optimized.ipynb`: optimized MLP on TF-IDF+SVD
- `notebooks/cnn_embeddings.ipynb`: embedding-based 1D-CNN experiment
