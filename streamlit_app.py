import os
import sys
import json
from pathlib import Path
import joblib
import streamlit as st

# Ensure repo root is on sys.path when running directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fakenews.preprocess import clean_text


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def load_threshold(meta_path: str):
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("threshold"))
    except Exception:
        return None


st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")

st.title("Fake News Classifier 📰")
st.caption("TF-IDF + Logistic Regression/LinearSVC baseline with optional thresholding.")

# Sidebar configuration
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model path", value=str(REPO_ROOT / "models" / "fake_news_model.joblib"))
use_auto_threshold = st.sidebar.checkbox("Auto threshold (from meta)", value=True)
manual_threshold = st.sidebar.slider("Manual threshold (REAL)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Load model lazily
model = None
load_error = None
if model_path:
    try:
        model = load_model(model_path)
    except Exception as e:
        load_error = str(e)

if load_error:
    st.error(f"Failed to load model: {load_error}")

# Text input
text = st.text_area("Article text", height=180, placeholder="Paste or type the news article text here...")

# Determine threshold
threshold = manual_threshold
if use_auto_threshold and model_path:
    meta_path = os.path.splitext(model_path)[0] + ".meta.json"
    t = load_threshold(meta_path)
    if t is not None:
        threshold = t
        st.sidebar.success(f"Auto threshold loaded: {threshold:.3f}")
    else:
        st.sidebar.info("No meta threshold found; using manual value.")

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Classify")
with col2:
    show_proba = st.checkbox("Show probabilities", value=True)

if run_btn:
    if not text.strip():
        st.warning("Please enter article text.")
    elif model is None:
        st.error("Model not loaded. Provide a valid model path.")
    else:
        cleaned = clean_text(text)
        # Predict
        label = None
        fake_p = None
        real_p = None
        classes = getattr(model, "classes_", [0, 1])
        try:
            if hasattr(model, "predict_proba") and show_proba:
                proba = model.predict_proba([cleaned])[0]
                try:
                    real_idx = list(classes).index(1)
                    fake_idx = list(classes).index(0)
                except ValueError:
                    fake_idx, real_idx = 0, 1
                real_p = float(proba[real_idx])
                fake_p = float(proba[fake_idx])
                label = "REAL" if real_p >= threshold else "FAKE"
            else:
                pred = int(model.predict([cleaned])[0])
                label = "REAL" if pred == 1 else "FAKE"
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            st.subheader(f"Prediction: {label}")
            if show_proba and real_p is not None and fake_p is not None:
                st.metric(label="REAL probability", value=f"{real_p:.3f}")
                st.metric(label="FAKE probability", value=f"{fake_p:.3f}")
                st.caption(f"Threshold: {threshold:.3f} | Classes order: {list(classes)}")
            with st.expander("Show cleaned text"):
                st.code(cleaned)

st.divider()
st.caption("Tip: Save threshold metadata during training with --auto-threshold and keep the .meta.json next to the model.")
