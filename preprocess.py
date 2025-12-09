import re
from typing import List

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALPHA_NUM_RE = re.compile(r"[^a-zA-Z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALPHA_NUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns: text, label (0=fake, 1=real)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns")
    df["clean"] = df["text"].map(clean_text)
    return df


def split_features_labels(df: pd.DataFrame):
    return df["clean"].tolist(), df["label"].astype(int).tolist()
