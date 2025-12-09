import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

from preprocess import load_dataset, split_features_labels
from collections import Counter


def build_pipeline(max_features: int = 5000, ngram_range=(1, 1), class_weight: str | None = None,
                   min_df: int = 1, max_df: float = 1.0, calibrate: bool = False,
                   model_type: str = "lr", C: float = 1.0, sublinear_tf: bool = False) -> Pipeline:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, max_df=max_df, sublinear_tf=sublinear_tf)
    if model_type == "linear_svc":
        base_clf = LinearSVC(C=C, class_weight=(None if class_weight in (None, 'none') else 'balanced'))
        # LinearSVC does not support predict_proba; calibration with decision_function would be a larger change.
        clf = base_clf
    else:
        base_clf = LogisticRegression(max_iter=1000, class_weight=(None if class_weight in (None, 'none') else 'balanced'), C=C)
        # scikit-learn >=1.3 uses 'estimator' instead of 'base_estimator'
        clf = CalibratedClassifierCV(estimator=base_clf, cv=3, method='sigmoid') if calibrate else base_clf
    return Pipeline([
        ("tfidf", vec),
        ("clf", clf)
    ])


def main():
    parser = argparse.ArgumentParser(description="Train Fake News Detection model")
    parser.add_argument("--data", required=True, help="Path to CSV with columns: text,label")
    parser.add_argument("--model-out", default=os.path.join("models", "fake_news_model.joblib"),
                        help="Output path for trained model")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max features")
    parser.add_argument("--ngrams", type=str, default="1,1", help="TF-IDF ngram range as start,end (e.g., 1,2)")
    parser.add_argument("--cv", type=int, default=0, help="Optional k-fold cross-validation folds (>=2)")
    parser.add_argument("--class-weight", type=str, default="none", choices=["none", "balanced"], help="Use class weighting for imbalance")
    parser.add_argument("--min-df", type=int, default=1, help="Ignore terms with document freq < min-df")
    parser.add_argument("--max-df", type=float, default=1.0, help="Ignore terms in > max-df fraction of docs (0<max-df<=1)")
    parser.add_argument("--model-type", type=str, default="lr", choices=["lr", "linear_svc"], help="Classifier type: lr or linear_svc")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength for LR/LinearSVC")
    parser.add_argument("--calibrate", action="store_true", help="Wrap classifier with probability calibration (sigmoid)")
    parser.add_argument("--tune-hyperparams", action="store_true", help="Run GridSearchCV over TF-IDF + classifier hyperparameters")
    parser.add_argument("--tune-scoring", type=str, default="f1_macro", choices=["f1", "f1_macro"], help="Scoring metric for hyperparam tuning")
    args = parser.parse_args()

    df = load_dataset(args.data)
    X, y = split_features_labels(df)

    cls_counts = Counter(y)
    can_stratify = all(count >= 2 for count in cls_counts.values()) and len(set(y)) >= 2 and len(y) >= 5

    # parse ngram range
    try:
        n_start, n_end = map(int, args.ngrams.split(","))
    except Exception:
        n_start, n_end = 1, 1

    pipe = build_pipeline(
        args.max_features,
        ngram_range=(n_start, n_end),
        class_weight=args.class_weight,
        min_df=args.min_df,
        max_df=args.max_df,
        calibrate=args.calibrate,
        model_type=args.model_type,
        C=args.C,
    )

    # Optional hyperparameter tuning
    if args.tune_hyperparams:
        if not can_stratify:
            print("Dataset too small for robust tuning; proceeding without tuning.")
        else:
            param_grid = {
                "tfidf__max_features": [5000, 10000, 20000, 50000],
                "tfidf__ngram_range": [(1,1), (1,2)],
                "tfidf__min_df": [1, 2, 5],
                "tfidf__max_df": [0.85, 0.9, 0.95],
                "tfidf__sublinear_tf": [True],
            }
            if args.model_type == "lr":
                # When calibration is enabled, the LogisticRegression is inside CalibratedClassifierCV under 'estimator'
                param_grid["clf__estimator__C" if isinstance(pipe.named_steps["clf"], CalibratedClassifierCV) else "clf__C"] = [0.1, 1, 3, 10]
            elif args.model_type == "linear_svc":
                param_grid["clf__C"] = [0.1, 1, 3, 10]

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            search = GridSearchCV(pipe, param_grid=param_grid, cv=skf, scoring=args.tune_scoring, n_jobs=-1)
            print("Running hyperparameter search...")
            search.fit(X, y)
            print(f"Best score ({args.tune_scoring}): {search.best_score_:.4f}")
            print("Best params:")
            for k, v in search.best_params_.items():
                print(f" - {k}: {v}")
            pipe = search.best_estimator_

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, digits=4))
        # optional CV on training split if requested
        if args.cv and args.cv >= 2:
            skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
            print(f"CV (n={args.cv}) accuracy: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
    else:
        # Small dataset fallback: train on all data and skip eval split
        pipe.fit(X, y)
        print("Dataset too small for stratified split; trained on full data.")
        if args.cv and args.cv >= 2 and len(set(y)) >= 2:
            skf = StratifiedKFold(n_splits=min(args.cv, max(2, min(cls_counts.values()))), shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
            print(f"CV (n={skf.get_n_splits()}) accuracy: mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
