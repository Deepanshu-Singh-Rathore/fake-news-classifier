import streamlit as st
import joblib
from pathlib import Path
import sys
import requests

# allow importing preprocess when running from repo root
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from preprocess import clean_text  # noqa: E402

try:
    import trafilatura  # type: ignore
except Exception:
    trafilatura = None

def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

    st.title("📰 Fake News Detector")
    st.write("TF-IDF + Logistic Regression baseline")

    # Model selection
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    default_models = sorted([p for p in models_dir.glob("*.joblib")])
    model_choice = st.selectbox(
        "Select a trained model",
        options=["<upload a model>"] + [str(p) for p in default_models],
        index=1 if default_models else 0,
    )

    uploaded_model = st.file_uploader("...or upload a .joblib model", type=["joblib"])

    model = None
    if uploaded_model is not None:
        tmp_path = models_dir / "uploaded_uploaded_model.joblib"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_model.read())
        model = joblib.load(tmp_path)
    elif model_choice and model_choice != "<upload a model>":
        model = joblib.load(model_choice)

    threshold = st.slider("Decision threshold for REAL", min_value=0.30, max_value=0.80, value=0.50, step=0.01)

    st.subheader("Classify Text")
    text = st.text_area("Enter news text", height=160, placeholder="Type or paste the article text here...")
    if st.button("Predict from Text"):
        if model is None:
            st.error("Please select or upload a model first.")
        elif not text.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(text)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([cleaned])[0]
                classes = getattr(model, "classes_", [0,1])
                try:
                    real_idx = list(classes).index(1)
                    fake_idx = list(classes).index(0)
                except ValueError:
                    fake_idx, real_idx = 0, 1
                real_p = proba[real_idx]
                fake_p = proba[fake_idx]
                label = "REAL" if real_p >= threshold else "FAKE"
                st.metric(label=f"Prediction: {label}", value=f"REAL: {real_p:.3f}", delta=f"FAKE: {fake_p:.3f}")
            else:
                pred = model.predict([cleaned])[0]
                label = "REAL" if pred == 1 else "FAKE"
                st.success(f"Prediction: {label}")

    st.divider()
    st.subheader("Classify by URL")
    url = st.text_input("Article URL", placeholder="https://example.com/news/article")
    if st.button("Fetch & Predict from URL"):
        if model is None:
            st.error("Please select or upload a model first.")
        elif not url.strip():
            st.warning("Please enter a URL.")
        else:
            extracted = None
            try:
                if trafilatura is not None:
                    downloaded = trafilatura.fetch_url(url, no_ssl=True)
                    if downloaded:
                        extracted = trafilatura.extract(downloaded, include_comments=False, favor_recall=True)
            except Exception as e:
                st.info("Falling back to page content; extraction had an issue.")
            # fallback minimal body fetch if needed
            if not extracted:
                try:
                    r = requests.get(url, timeout=15)
                    if r.ok:
                        # very naive fallback: use text content directly
                        extracted = r.text[:15000]
                except Exception:
                    pass

            if not extracted:
                st.error("Could not extract content from the URL.")
            else:
                cleaned = clean_text(extracted)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba([cleaned])[0]
                    classes = getattr(model, "classes_", [0,1])
                    try:
                        real_idx = list(classes).index(1)
                        fake_idx = list(classes).index(0)
                    except ValueError:
                        fake_idx, real_idx = 0, 1
                    real_p = proba[real_idx]
                    fake_p = proba[fake_idx]
                    label = "REAL" if real_p >= threshold else "FAKE"
                    st.metric(label=f"Prediction: {label}", value=f"REAL: {real_p:.3f}", delta=f"FAKE: {fake_p:.3f}")
                else:
                    pred = model.predict([cleaned])[0]
                    label = "REAL" if pred == 1 else "FAKE"
                    st.success(f"Prediction: {label}")

    st.caption("Baseline demo. Consider larger datasets and stronger models for production.")

if __name__ == "__main__":
    main()
