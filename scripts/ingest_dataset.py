import argparse
import os
from typing import List

import pandas as pd

from pathlib import Path

REQUIRED_COLUMNS = {"text", "label"}


def read_many(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing columns: {missing}")
        frames.append(df[["text", "label"]])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["text", "label"])  # basic sanity
    return out


def balance(df: pd.DataFrame, method: str, per_class: int | None = None) -> pd.DataFrame:
    if method == "none":
        return df

    counts = df["label"].value_counts()
    if len(counts) < 2:
        # nothing to balance if only one class
        return df

    if per_class is not None:
        target = per_class
    else:
        target = counts.min() if method == "undersample" else counts.max()

    parts = []
    for label, group in df.groupby("label"):
        if method == "undersample":
            g = group.sample(n=min(target, len(group)), random_state=42)
        elif method == "oversample":
            if len(group) == 0:
                continue
            reps = int((target + len(group) - 1) // len(group))
            g = pd.concat([group] * reps, ignore_index=True).sample(n=target, random_state=42)
        else:
            g = group
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Ingest and (optionally) balance fake news datasets")
    parser.add_argument("--data", nargs='+', required=True, help="One or more CSV files (text,label)")
    parser.add_argument("--output", default=str(Path("data") / "merged.csv"), help="Output CSV path")
    parser.add_argument("--balance", choices=["none", "undersample", "oversample"], default="none",
                        help="Balance strategy across classes")
    parser.add_argument("--per-class", type=int, default=None, help="Limit samples per class after balancing")
    parser.add_argument("--test-size", type=float, default=None, help="Optional test split fraction, e.g., 0.2")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows before saving")

    args = parser.parse_args()

    df = read_many(args.data)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df = balance(df, method=args.balance, per_class=args.per_class)

    os.makedirs(Path(args.output).parent, exist_ok=True)

    if args.test_size and 0.0 < args.test_size < 1.0:
        test_n = max(1, int(len(df) * args.test_size))
        test = df.sample(n=test_n, random_state=42)
        train = df.drop(test.index).reset_index(drop=True)
        test = test.reset_index(drop=True)

        out_train = str(Path(args.output).with_name("train.csv"))
        out_test = str(Path(args.output).with_name("test.csv"))
        train.to_csv(out_train, index=False)
        test.to_csv(out_test, index=False)
        print(f"Wrote {len(train)} rows to {out_train}")
        print(f"Wrote {len(test)} rows to {out_test}")
    else:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
