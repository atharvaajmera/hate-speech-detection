# Hate Speech Project

Final submission layout for hate-speech detection using tweet preprocessing, TF-IDF features, and a tuned `LinearSVC` classifier.

## Final structure

```text
hate-speech-detection/
|-- Hate_Speech_Analysis.ipynb
|-- data/
|   |-- raw_dataset.csv
|   `-- raw/
|       `-- labeled_data.csv
|-- models/
|   |-- tfidf_vectorizer.pkl
|   `-- hate_speech_svc.pkl
|-- notebooks/
|   `-- experiments/
|       |-- 01_logreg_tfidf.ipynb
|       |-- 02_svm_optimized.ipynb
|       |-- 03_mlp_tfidf_optimized.ipynb
|       `-- 04_cnn_embeddings.ipynb
|-- src/
|   |-- __init__.py
|   `-- preprocess.py
|-- predict.py
|-- app.py
`-- requirements.txt
```

## What each part does

- `Hate_Speech_Analysis.ipynb`: TA-facing notebook with clean EDA, preprocessing, final model training, and conclusion.
- `notebooks/experiments/`: archived experiments showing the sequence of model attempts.
- `src/preprocess.py`: shared preprocessing, training, and model-loading utilities.
- `predict.py`: CLI prediction entrypoint.
- `app.py`: Streamlit interface for interactive predictions.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate final model artifacts

Run the root notebook `Hate_Speech_Analysis.ipynb`, or use:

```powershell
.\.venv\Scripts\python.exe predict.py --retrain
```

This creates:

```text
models/tfidf_vectorizer.pkl
models/hate_speech_svc.pkl
```

## CLI usage

Interactive mode:

```powershell
.\.venv\Scripts\python.exe predict.py
```

One-shot mode:

```powershell
.\.venv\Scripts\python.exe predict.py "your text here"
```

## Streamlit app

```powershell
streamlit run app.py
```
