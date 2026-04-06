# Hate Speech Detection

Basic starter setup for a hate speech detection project using tweet data.

## Current scope

- Load the raw labeled tweet dataset
- Clean tweet text
- Save a processed dataset for later model training

## Project structure

```text
hate-speech-detection/
|-- data/
|   |-- raw/
|   |   `-- labeled_data.csv
|   `-- processed/
|-- preprocess_data.py
|-- requirements.txt
`-- tweet_cleaner.py
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
python preprocess_data.py
```

This will create:

```text
data/processed/cleaned_labeled_data.csv
```

## Next steps

- Add train/test split
- Train a baseline text classification model
- Evaluate model performance
- Build a small app or API for predictions

