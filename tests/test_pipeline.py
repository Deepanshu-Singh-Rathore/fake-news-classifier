import os
import sys
from pathlib import Path
import joblib
import pandas as pd

# Ensure project root (folder containing preprocess.py) is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fakenews.preprocess import load_dataset, split_features_labels  # noqa: E402
from fakenews.train import build_pipeline  # noqa: E402


def test_training_and_inference(tmp_path):
    data_path = tmp_path / "mini.csv"
    df_small = pd.DataFrame({
        "text": ["Real news here, confirmed.", "Fake claim about aliens."],
        "label": [1, 0]
    })
    df_small.to_csv(data_path, index=False)

    df = load_dataset(str(data_path))
    X, y = split_features_labels(df)

    pipe = build_pipeline(max_features=100)
    pipe.fit(X, y)

    pred = pipe.predict(["confirmed real news"])[0]
    assert pred in (0, 1)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, str(model_path))
    assert os.path.exists(model_path)
