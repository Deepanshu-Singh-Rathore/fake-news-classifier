import argparse
import json
import joblib
import os
from fakenews.preprocess import clean_text


def load_threshold(meta_path: str) -> float | None:
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("threshold"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Inference for Fake News Detection")
    parser.add_argument("--model", default="models/fake_news_model.joblib", help="Path to trained model")
    parser.add_argument("--text", required=True, help="News text to classify")
    parser.add_argument("--proba", action="store_true", help="Show prediction probabilities")
    parser.add_argument("--threshold", default="auto", help="Probability threshold for REAL (float or 'auto')")
    parser.add_argument("--debug", action="store_true", help="Show cleaned text and model class order")
    args = parser.parse_args()

    model = joblib.load(args.model)
    cleaned = clean_text(args.text)
    if args.debug:
        print(f"[DEBUG] Raw input: {args.text}")
        print(f"[DEBUG] Cleaned input: {cleaned}")
        if hasattr(model, "classes_"):
            print(f"[DEBUG] Model classes order: {list(model.classes_)} (index aligns with probabilities)")

    threshold = None
    if args.threshold == "auto":
        meta_path = os.path.splitext(args.model)[0] + ".meta.json"
        threshold = load_threshold(meta_path)
        if threshold is None:
            threshold = 0.5
            if args.debug:
                print("[DEBUG] No metadata threshold found; fallback to 0.5")
    else:
        try:
            threshold = float(args.threshold)
        except ValueError:
            threshold = 0.5
            print("Invalid threshold provided; using 0.5")

    if args.proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        classes = getattr(model, "classes_", [0, 1])
        try:
            real_idx = list(classes).index(1)
            fake_idx = list(classes).index(0)
        except ValueError:
            fake_idx, real_idx = 0, 1
        real_p = proba[real_idx]
        fake_p = proba[fake_idx]
        pred_label = "REAL" if real_p >= threshold else "FAKE"
        print(f"{pred_label} (FAKE={fake_p:.3f}, REAL={real_p:.3f}, threshold={threshold:.3f})")
    else:
        pred = model.predict([cleaned])[0]
        label = "REAL" if pred == 1 else "FAKE"
        print(label)


if __name__ == "__main__":
    main()
