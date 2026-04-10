# Hate Speech Detection

Hate speech classification project using cleaned tweet text, TF-IDF features, optimized classical models, and an embedding-based CNN experiment.

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
|   |-- detailed_eda_hate_speech.ipynb
|   |-- eda.ipynb
|   |-- logreg_tfidf.ipynb
|   |-- mlp_tfidf_optimized.ipynb
|   `-- svm_optimized.ipynb
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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you need to create the environment first:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run preprocessing

```powershell
.\.venv\Scripts\python.exe scripts/preprocess_data.py
```

This will create:

```text
data/processed/cleaned_labeled_data.csv
```

## Build TF-IDF features

```powershell
.\.venv\Scripts\python.exe scripts/vectorize_tfidf.py
```

This will create:

```text
data/processed/tfidf/X_tfidf.npz
data/processed/tfidf/tfidf_vectorizer.pkl
data/processed/tfidf/y_labels.csv
```

## Run CLI prediction (multi-model)

```powershell
.\.venv\Scripts\python.exe scripts/predict.py
```

One-shot mode:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py "your text here"
```

Force retraining cached models:

```powershell
.\.venv\Scripts\python.exe scripts/predict.py --retrain
```

## Notebooks

- `notebooks/eda.ipynb`: main exploratory data analysis notebook
- `notebooks/detailed_eda_hate_speech.ipynb`: extended EDA variant with extra analysis cells
- `notebooks/logreg_tfidf.ipynb`: logistic regression baseline on TF-IDF
- `notebooks/svm_optimized.ipynb`: optimized LinearSVC workflow
- `notebooks/mlp_tfidf_optimized.ipynb`: optimized MLP on TF-IDF+SVD
- `notebooks/cnn_embeddings.ipynb`: embedding-based 1D-CNN experiment using tokenized text and GloVe embeddings

## Notes

- `data/processed/` and `models/` are generated locally and ignored by Git.
- The CLI prints live model predictions from the cached artifacts in `models/`; it does not reuse notebook test-set metrics.
