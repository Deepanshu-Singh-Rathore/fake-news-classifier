import argparse
import os
import json
import joblib
from collections import Counter
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_curve

from fakenews.preprocess import load_dataset, split_features_labels


def build_pipeline(model_type: str = "logistic", max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 1),
                   class_weight: str | None = None, min_df: int = 1, max_df: float = 1.0) -> Pipeline:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    cw = (None if class_weight in (None, 'none') else 'balanced')
    if model_type == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight=cw)
    elif model_type == "svc":
        base = LinearSVC(class_weight=cw)
        # Calibrated to get probabilities
        clf = CalibratedClassifierCV(base_estimator=base, cv=3)
    elif model_type == "transformer":
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise RuntimeError("transformers/torch not installed. Install extras: pip install -e .[transformer]")
        # Lightweight wrapper using DistilBERT (binary)
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        class TransformerWrapper:
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model
                self.classes_ = [0, 1]

            def fit(self, X, y):
                # Placeholder: no fine-tuning for speed; in real usage implement training loop.
                return self

            def predict(self, X):
                probs = self.predict_proba(X)
                return [int(p[1] >= 0.5) for p in probs]

            def predict_proba(self, X):
                outputs = []
                for text in X:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
                    with torch.no_grad():
                        logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    outputs.append(probs)
                return outputs

        clf = TransformerWrapper(tokenizer, base_model)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([
        ("tfidf", vec),
        ("clf", clf)
    ])


def compute_best_threshold(y_true, real_probs):
    fpr, tpr, thresholds = roc_curve(y_true, real_probs)
    youden = tpr - fpr
    best_idx = int(youden.argmax())
    return float(thresholds[best_idx]), float(fpr[best_idx]), float(tpr[best_idx])


def main():
    parser = argparse.ArgumentParser(description="Train Fake News Detection model")
    parser.add_argument("--data", required=True, help="Path to CSV with columns: text,label")
    parser.add_argument("--model-out", default=os.path.join("models", "fake_news_model.joblib"),
                        help="Output path for trained model")
    parser.add_argument("--model-type", choices=["logistic", "svc", "transformer"], default="logistic",
                        help="Classifier backend")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max features")
    parser.add_argument("--ngrams", type=str, default="1,1", help="TF-IDF ngram range as start,end (e.g., 1,2)")
    parser.add_argument("--cv", type=int, default=0, help="Optional k-fold cross-validation folds (>=2)")
    parser.add_argument("--class-weight", type=str, default="none", choices=["none", "balanced"], help="Use class weighting for imbalance")
    parser.add_argument("--min-df", type=int, default=1, help="Ignore terms with document freq < min-df")
    parser.add_argument("--max-df", type=float, default=1.0, help="Ignore terms in > max-df fraction of docs (0<max-df<=1)")
    parser.add_argument("--auto-threshold", action="store_true", help="Compute and store optimal REAL probability threshold metadata")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X, y = split_features_labels(df)
    cls_counts = Counter(y)
    can_stratify = all(count >= 2 for count in cls_counts.values()) and len(set(y)) >= 2 and len(y) >= 5

    try:
        n_start, n_end = map(int, args.ngrams.split(","))
    except Exception:
        n_start, n_end = 1, 1

    pipe = build_pipeline(
        model_type=args.model_type,
        max_features=args.max_features,
        ngram_range=(n_start, n_end),
        class_weight=args.class_weight,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, digits=4))
        if args.cv and args.cv >= 2 and args.model_type != "transformer":
            skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
            print(f"CV (n={args.cv}) accuracy: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
        eval_X, eval_y = X_test, y_test
    else:
        pipe.fit(X, y)
        print("Dataset too small for stratified split; trained on full data.")
        if args.cv and args.cv >= 2 and len(set(y)) >= 2 and args.model_type != "transformer":
            skf = StratifiedKFold(n_splits=min(args.cv, max(2, min(cls_counts.values()))), shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
            print(f"CV (n={skf.get_n_splits()}) accuracy: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
        eval_X, eval_y = X, y

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f"Model saved to {args.model_out}")

    if args.auto_threshold and hasattr(pipe, 'predict_proba'):
        try:
            proba = pipe.predict_proba(eval_X)
            # Determine REAL index
            classes = getattr(pipe, "classes_", [0, 1])
            real_idx = list(classes).index(1) if 1 in classes else 1
            real_probs = [p[real_idx] for p in proba]
            threshold, fpr, tpr = compute_best_threshold(eval_y, real_probs)
            meta = {
                "model_type": args.model_type,
                "threshold": threshold,
                "threshold_metrics": {"fpr": fpr, "tpr": tpr},
                "classes": list(classes),
                "vectorizer": {
                    "max_features": args.max_features,
                    "ngram_range": [n_start, n_end],
                    "min_df": args.min_df,
                    "max_df": args.max_df
                }
            }
            meta_path = os.path.splitext(args.model_out)[0] + ".meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"Stored threshold metadata at {meta_path} (threshold={threshold:.4f})")
        except Exception as e:
            print(f"Failed to compute threshold metadata: {e}")
    elif args.auto_threshold:
        print("auto-threshold requested but model lacks predict_proba; skipped.")


if __name__ == "__main__":
    main()
