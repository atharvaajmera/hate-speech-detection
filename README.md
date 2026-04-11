# Hate Speech Project

Final submission layout for hate-speech detection using tweet preprocessing, TF-IDF features, and an optimized `LinearSVC` classifier.

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
|-- src/
|   |-- __init__.py
|   `-- preprocess.py
|-- predict.py
|-- app.py
`-- requirements.txt
```

## What each file does

- `Hate_Speech_Analysis.ipynb`: EDA + model training notebook for submission.
- `src/preprocess.py`: shared preprocessing, training, and artifact-loading utilities.
- `predict.py`: CLI prediction script.
- `app.py`: Streamlit UI for interactive predictions.
- `models/`: saved TF-IDF vectorizer and trained LinearSVC model.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train and save model artifacts

Run the notebook `Hate_Speech_Analysis.ipynb`, or use:

```powershell
.\.venv\Scripts\python.exe predict.py --retrain
```

This will save:

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
