import argparse
import joblib
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from preprocess import load_dataset, split_features_labels  # noqa: E402

def compute_best_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return thresholds[best_idx], fpr[best_idx], tpr[best_idx], auc(fpr, tpr)

def main():
    parser = argparse.ArgumentParser(description="Suggest optimal REAL probability threshold via ROC (Youden's J)")
    parser.add_argument("--model", required=True, help="Path to trained model (.joblib)")
    parser.add_argument("--data", required=True, help="CSV with text,label columns")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of rows")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = load_dataset(args.data)
    if args.limit:
        df = df.head(args.limit)
    X, y = split_features_labels(df)

    if not hasattr(model, "predict_proba"):
        print("Model does not support probabilities; cannot tune threshold.")
        sys.exit(1)

    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", [0,1])
    try:
        real_idx = list(classes).index(1)
    except ValueError:
        real_idx = 1
    real_probs = proba[:, real_idx]

    threshold, fpr, tpr, roc_auc = compute_best_threshold(y, real_probs)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Best threshold (Youden's J): {threshold:.4f} (TPR={tpr:.3f}, FPR={fpr:.3f})")
    print("Use: infer --threshold", f"{threshold:.4f}")

if __name__ == "__main__":
    main()