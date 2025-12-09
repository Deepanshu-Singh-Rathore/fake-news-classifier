import re
import pandas as pd

URL_PATTERN = re.compile(r'https?://\S+')
TAG_PATTERN = re.compile(r'<.*?>')
NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9\s]')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = URL_PATTERN.sub(' ', t)
    t = TAG_PATTERN.sub(' ', t)
    t = NON_ALPHANUM.sub(' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain text and label columns")
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].apply(clean_text)
    return df


def split_features_labels(df: pd.DataFrame):
    return df['text'].tolist(), df['label'].tolist()
