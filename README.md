# Fake News Classifier

Lightweight, explainable NLP baseline using TF‑IDF + Logistic Regression/LinearSVC. Minimal build focused on training, inference, and a simple Streamlit UI.

## Quick Start (Windows PowerShell)

Run all commands from the project root directory.

```powershell
# 1) Create and activate a local virtual environment (using .fenv)
python -m venv .fenv
.\.fenv\Scripts\Activate.ps1

# 2) Install dependencies (runtime + dev)
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt -r .\dev-requirements.txt

# 3) Train on the provided tiny dataset (saves model to models/)
python .\fakenews\train.py --data .\data\merged.csv --model-out .\models\fake_news_model.joblib --ngrams 1,1 --max-features 2000 --class-weight balanced --cv 0 --min-df 1 --max-df 1.0 --auto-threshold

# 4) Run inference on a single text
python .\fakenews\infer.py --model .\models\fake_news_model.joblib --text "NASA releases high-res images from Europa flyby showing potential salt deposits." --proba

# (Optional) Adjust decision threshold manually
python .\fakenews\infer.py --model .\models\fake_news_model.joblib --text "Scientists confirm Atlantis found intact beneath Arctic ice shelf." --proba --threshold 0.55
```

### Streamlit UI

Launch an interactive UI to paste text and view predictions and probabilities.

```powershell
# From project root
streamlit run .\streamlit_app.py
```

If you saved threshold metadata during training (via `--auto-threshold`), the UI will load it from the `models/*.meta.json` file next to your model.

### Using Packaging / Console Scripts

After editable install (`python -m pip install -e .`), you can use console commands:

```powershell
fakenews-train --data .\data\merged.csv --model-out .\models\fake_news_model.joblib --ngrams 1,1 --max-features 2000 --class-weight balanced --cv 0 --min-df 1 --max-df 1.0 --auto-threshold
fakenews-infer --model .\models\fake_news_model.joblib --text "NASA releases high-res images from Europa flyby showing potential salt deposits." --proba --threshold 0.55
```

### Environment Setup

Create and activate a virtual environment and install dependencies as shown in Quick Start.

### Tuning Decision Threshold

Enable `--auto-threshold` during training to compute and store an optimal REAL probability threshold via ROC/Youden's J. You can override the threshold at inference time via `--threshold`.

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

## Data Format

- CSV with columns: `text,label`
- `label`: `0` for FAKE, `1` for REAL
- Examples at `data/sample.csv` and `data/merged.csv`

## Project Layout

```
README.md
pyproject.toml
requirements.txt
dev-requirements.txt
fakenews/
  preprocess.py       # text cleaning, dataset loading
  train.py            # training script (TF‑IDF + LogisticRegression/LinearSVC)
  infer.py            # CLI for classifying a single text
streamlit_app.py      # Streamlit UI for interactive classification
data/
  sample.csv          # tiny demo dataset
  merged.csv          # small combined dataset
models/               # trained models saved here
tests/
  test_pipeline.py    # sanity test on pipeline
```

## Developer: Tests

Install dev requirements and run tests:

```powershell
python -m pip install -r .\dev-requirements.txt
pytest -q tests\test_pipeline.py
```

## Troubleshooting

- Version mismatch warnings on model load: OK if you trained with a different scikit‑learn minor version; prefer re‑training under the current environment for long‑term stability.
- "ModuleNotFoundError": ensure `.fenv` is activated and dependencies installed with `python -m pip install -r requirements.txt`.
- "FileNotFoundError" for model: train first, or pass the correct `--model` path.
- Tiny datasets: the script may skip the stratified split and train on all data; consider adding more samples per class.

## Recommended Parameters for Tiny Datasets
Use conservative settings to avoid sparse feature failures:

```powershell
python .\fakenews\train.py --data .\data\merged.csv --model-out .\models\fake_news_model.joblib --ngrams 1,1 --max-features 2000 --class-weight balanced --cv 0 --min-df 1 --max-df 1.0 --auto-threshold
```

Avoid high `min-df`, bigrams, and large `max-features` until you have >100 examples per class.

## Improving the Model
- Add more REAL/FAKE samples (balanced).
- Try bigrams and cross‑validation when you have more data: `--ngrams 1,2 --cv 5`.
- Consider threshold tuning and a held‑out test set for evaluation.
