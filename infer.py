import argparse
import joblib
from preprocess import clean_text


def main():
    parser = argparse.ArgumentParser(description="Inference for Fake News Detection")
    parser.add_argument("--model", default="models/fake_news_model.joblib", help="Path to trained model")
    parser.add_argument("--text", required=True, help="News text to classify")
    parser.add_argument("--proba", action="store_true", help="Show prediction probabilities")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for REAL (class=1)")
    parser.add_argument("--debug", action="store_true", help="Show cleaned text and model class order")
    args = parser.parse_args()

    model = joblib.load(args.model)
    cleaned = clean_text(args.text)
    if args.debug:
        print(f"[DEBUG] Raw input: {args.text}")
        print(f"[DEBUG] Cleaned input: {cleaned}")
        if hasattr(model, "classes_"):
            print(f"[DEBUG] Model classes order: {list(model.classes_)} (index aligns with probabilities)")

    if args.proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        # Determine index for class 1 (REAL) robustly
        classes = getattr(model, "classes_", [0, 1])
        try:
            real_idx = list(classes).index(1)
            fake_idx = list(classes).index(0)
        except ValueError:
            # Fallback assumption
            fake_idx, real_idx = 0, 1
        real_p = proba[real_idx]
        fake_p = proba[fake_idx]
        pred_label = "REAL" if real_p >= args.threshold else "FAKE"
        print(f"{pred_label} (FAKE={fake_p:.3f}, REAL={real_p:.3f}, threshold={args.threshold:.2f})")
    else:
        pred = model.predict([cleaned])[0]
        label = "REAL" if pred == 1 else "FAKE"
        print(label)


if __name__ == "__main__":
    main()
