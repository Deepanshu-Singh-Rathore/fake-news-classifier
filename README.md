# Fake News Classifier

Lightweight, explainable NLP system using TF‑IDF + Logistic Regression/LinearSVC. Minimal build focused on training and inference.

## Quick Start (Windows PowerShell)

Run all commands from the project root directory (e.g., `fakenews/`).

```powershell
# 1) Create and activate a local virtual environment (using .fenv)
python -m venv .fenv
.\.fenv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt -r .\dev-requirements.txt

# 3) (Optional) Use provided tiny merged dataset instead of sample
python .\fakenews\train.py --data .\data\merged.csv --model-out .\models\fake_news_model.joblib --ngrams 1,1 --max-features 2000 --class-weight balanced --cv 0 --min-df 1 --max-df 1.0 --calibrate

# 4) Run inference on a single text
python .\fakenews\infer.py --model .\models\fake_news_model.joblib --text "NASA releases high-res images from Europa flyby showing potential salt deposits." --proba

# (Optional) Adjust decision threshold
python .\fakenews\infer.py --model .\models\fake_news_model.joblib --text "Scientists confirm Atlantis found intact beneath Arctic ice shelf." --proba --threshold 0.55

# (Optional) Evaluate quickly using scikit-learn report printed during training; rerun training with different params as needed.
```

### Using Packaging / Console Scripts

After editable install (`python -m pip install -e .`) you can use console commands:

```powershell
fakenews-train --data .\data\merged.csv --model-out .\models\fake_news_model.joblib --ngrams 1,1 --max-features 2000 --class-weight balanced --cv 0 --min-df 1 --max-df 1.0
fakenews-infer --model .\models\fake_news_model.joblib --text "NASA releases high-res images from Europa flyby showing potential salt deposits." --proba --threshold 0.55
```

### Environment Setup

Create and activate a virtual environment and install dependencies as shown in Quick Start.

### Live News Detection

Removed to keep the project minimal. Use local CSVs with `text,label`.

### URL Classification from CLI

Use `fakenews-live` to fetch articles from NewsAPI and classify them. It can optionally extract full text (via trafilatura) and deduplicate by canonical URL and text hash.

### Tuning Decision Threshold

Compute an optimal REAL probability threshold using ROC / Youden's J:

```powershell
fakenews-tune-threshold --model .\models\fake_news_model.joblib --data .\data\merged.csv
```
Then pass `--threshold <value>` to `fakenews-infer`.

### Building Distribution Artifacts (Wheel / sdist)

Install build tooling and create distributable packages:

```powershell
python -m pip install build
python -m build  # generates dist/*.whl and dist/*.tar.gz
```

Install the wheel elsewhere:
```powershell
python -m pip install dist\fakenews-0.1.0-py3-none-any.whl
```

### Editable Install for Development
```powershell
python -m pip install -e .[dev]
```


> Note: A separate project `SummarAI/` exists in the repository for summarization; the instructions here apply only to the `fakenews/` fake news detection baseline.

### Ingest and Balance Larger Datasets

Removed helper scripts to keep minimal footprint. Prepare your CSVs externally with `text,label` columns.

## Better Features and Evaluation

- Use n‑grams (bigrams) to improve features and add k‑fold CV for more stable accuracy.

```powershell
python .\fakenews\train.py --data .\data\sample.csv --model-out .\models\fake_news_model.joblib --ngrams 1,2 --cv 5 --calibrate
```

## Data Format

 CSV with columns: `text,label`
 `label`: `0` for FAKE, `1` for REAL
 Example at `data/sample.csv`

## Project Layout

```
fakenews/
  preprocess.py       # text cleaning, dataset loading
  train.py            # training script (TF‑IDF + LogisticRegression)
  infer.py            # CLI for classifying a single text
  requirements.txt    # Python dependencies
  data/
    sample.csv        # tiny demo dataset
    merged.csv        # (optional) output from ingestion
  models/             # trained models saved here
  tests/
  scripts/
    ingest_dataset.py # dataset merge/balance/split utility
    run_tests.ps1     # runs `pytest -q tests/test_pipeline.py`
  scripts/            # removed for minimal setup
```

## CLI Demo

Classify a single text with probabilities and threshold:

```powershell
fakenews-infer --model .\models\fake_news_model.joblib --text "Example article text" --proba --threshold 0.55
```

## Developer: Tests

Install dev requirements and run tests:

```powershell
python -m pip install -r .\dev-requirements.txt
pytest -q tests\test_pipeline.py
```

## Troubleshooting

- "ModuleNotFoundError": ensure `.fenv` is activated and dependencies installed with `python -m pip install -r requirements.txt`.
- "FileNotFoundError" for model: train first, or pass the correct `--model` path.
- Tiny datasets: the script will skip the stratified split and train on all data; consider adding more samples per class.

## Recommended Parameters for Tiny Datasets
Use conservative settings to avoid sparse feature failures:

```powershell
python .\fakenews\train.py --data .\data\merged.csv --model-out .\models\fake_news_model.joblib --ngrams 1,1 --max-features 2000 --class-weight balanced --cv 0 --min-df 1 --max-df 1.0 --calibrate
```

Avoid high `min-df`, bigrams, and large `max-features` until you have >100 examples per class.

## Improving the Model
- Add more REAL/FAKE samples (balanced).
- Use ingestion script to merge sources and create `train.csv` / `test.csv`:
```powershell
python .\fakenews\scripts\ingest_dataset.py --data .\data\merged.csv --shuffle --balance none --test-size 0.25 --output .\data\merged.csv
python .\fakenews\train.py --data .\data\train.csv --model-out .\models\fake_news_model.joblib --ngrams 1,2 --max-features 5000 --class-weight balanced --cv 5 --min-df 2 --max-df 0.9 --calibrate
python .\fakenews\scripts\evaluate_model.py --model .\models\fake_news_model.joblib --data .\data\test.csv --confusion --show-mis --top 20
```

## Notes
- This is a baseline; for production, consider larger datasets, stronger models, threshold tuning, and a held-out test set.
## Overview
See `NLP_Overview.txt` for a concise description of the NLP pipeline, algorithms, and components in this project.
