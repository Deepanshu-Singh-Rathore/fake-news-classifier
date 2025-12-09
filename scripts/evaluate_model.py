import argparse
import joblib
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from preprocess import load_dataset, split_features_labels, clean_text  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Quick evaluation of a trained fake news model on a CSV")
    parser.add_argument("--model", required=True, help="Path to trained model .joblib")
    parser.add_argument("--data", required=True, help="CSV with text,label columns")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of rows for speed")
    parser.add_argument("--show-mis", action="store_true", help="Show misclassified examples")
    parser.add_argument("--top", type=int, default=15, help="Show top N informative features (if logistic regression)")
    parser.add_argument("--confusion", action="store_true", help="Show confusion matrix")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = load_dataset(args.data)
    if args.limit:
        df = df.head(args.limit)

    X, y = split_features_labels(df)
    preds = model.predict(X)

    correct = sum(int(p == t) for p, t in zip(preds, y))
    acc = correct / len(y) if y else 0.0
    print(f"Samples: {len(y)}  Accuracy: {acc:.4f}" )
    # Detailed per-class metrics
    try:
        print(classification_report(y, preds, digits=4))
    except Exception:
        pass
    if args.confusion:
        # Build confusion matrix counts
        labels = sorted(set(y + list(preds)))
        matrix = { (true,pred):0 for true in labels for pred in labels }
        for true,pred in zip(y,preds):
            matrix[(true,pred)] += 1
        print("Confusion Matrix (rows=true, cols=pred):")
        header = "      " + "  ".join(f"pred={l}" for l in labels)
        print(header)
        for t in labels:
            row_counts = "  ".join(f"{matrix[(t,p)]:3d}" for p in labels)
            print(f"true={t}  {row_counts}")

    if args.show_mis:
        print("--- Misclassified examples ---")
        for text, true, pred in zip(df['text'], y, preds):
            if pred != true:
                label_true = 'REAL' if true == 1 else 'FAKE'
                label_pred = 'REAL' if pred == 1 else 'FAKE'
                print(f"[TRUE={label_true}][PRED={label_pred}] {text[:200].strip()}" )

    # PR-AUC (FAKE class=0) if we have calibrated probabilities or decision scores
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        classes = getattr(model, 'classes_', [0,1])
        try:
            real_idx = list(classes).index(1)
        except ValueError:
            real_idx = 1
        # FAKE probability as 1 - REAL
        fake_probs = 1.0 - proba[:, real_idx]
        ap = average_precision_score(np.array(y) == 0, fake_probs)
        print(f"PR-AUC (FAKE): {ap:.4f}")

    # Feature inspection for logistic regression
    clf = getattr(model, 'named_steps', {}).get('clf')
    vec = getattr(model, 'named_steps', {}).get('tfidf')
    if clf is not None and vec is not None and hasattr(clf, 'coef_'):
        feature_names = vec.get_feature_names_out()
        # Assuming binary classification coef_[0] corresponds to class 0 vs class 1
        coefs = clf.coef_[0]
        top_fake_idx = coefs.argsort()[:args.top]
        top_real_idx = coefs.argsort()[::-1][:args.top]
        print(f"Top {args.top} REAL indicators:")
        print(", ".join(f"{feature_names[i]}({coefs[i]:.2f})" for i in top_real_idx))
        print(f"Top {args.top} FAKE indicators:")
        print(", ".join(f"{feature_names[i]}({coefs[i]:.2f})" for i in top_fake_idx))
    else:
        print("Model does not support coefficient feature inspection.")


if __name__ == "__main__":
    main()
