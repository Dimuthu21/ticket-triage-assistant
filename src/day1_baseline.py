"""
Day 1: data audit + stratified split + TF-IDF baseline for support-ticket QUEUE classification.

Run (downloads the dataset from Hugging Face):
    python src/day1_baseline.py

Run with a local CSV instead:
    python src/day1_baseline.py --csv path/to/tickets.csv

Everything is deterministic (fixed seed), so re-running on Colab gives the same splits.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

HF_DATASET = "Tobi-Bueck/customer-support-tickets"


def load_data(csv_path, hf_name):
    if csv_path:
        return pd.read_csv(csv_path)
    from datasets import load_dataset
    ds = load_dataset(hf_name)
    return pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)


def prepare(df, target, lang):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"\nColumns found: {list(df.columns)}")
    for needed in ("subject", "body", target):
        if needed not in df.columns:
            raise SystemExit(f"Column '{needed}' not found. Adjust --target or check the columns printed above.")
    if lang and "language" in df.columns:
        before = len(df)
        df = df[df["language"].astype(str).str.lower() == lang.lower()]
        print(f"Language filter '{lang}': {before} -> {len(df)} rows")
    # Input text = subject + body ONLY.
    # We do NOT use 'answer', 'type', 'priority' or tags as features: 'answer' does not exist
    # when a new ticket arrives (leakage), and the others would make the task unrealistically easy.
    df["text"] = (df["subject"].fillna("").astype(str) + ". " + df["body"].fillna("").astype(str)).str.strip()
    df = df[(df["text"].str.len() > 5) & df[target].notna()]
    before = len(df)
    df = df.drop_duplicates(subset="text")  # exact duplicate tickets must not leak across splits
    print(f"Dropped {before - len(df)} exact-duplicate tickets -> {len(df)} rows")
    return df[["text", target]].rename(columns={target: "label"}).reset_index(drop=True)


def audit(df, out):
    counts = df["label"].value_counts()
    print("\n=== Class distribution ===")
    print(counts.to_string())
    print(f"\nImbalance ratio (largest/smallest class): {counts.max() / counts.min():.1f}")
    words = df["text"].str.split().str.len()
    print(f"Words per ticket: mean={words.mean():.0f}, median={words.median():.0f}, 95th pct={words.quantile(.95):.0f}")
    counts.to_csv(out / "class_distribution.csv")
    ax = counts.sort_values().plot.barh(figsize=(8, 5))
    ax.set_title("Tickets per queue")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=150)
    plt.close()


def split(df, seed, data_dir):
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(data_dir / f"{name}.csv", index=False)
    print(f"\nSplit sizes -> train {len(train)}, val {len(val)}, test {len(test)} (saved to {data_dir}/)")
    return train, val, test


def scores(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted"), 4),
    }


def make_model(clf):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000, lowercase=True)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--hf-name", default=HF_DATASET)
    ap.add_argument("--target", default="queue")
    ap.add_argument("--lang", default="en", help="use '' to keep all languages")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-dir", default="data/processed")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = prepare(load_data(a.csv, a.hf_name), a.target, a.lang or None)
    audit(df, out)
    train, val, test = split(df, a.seed, Path(a.data_dir))

    # 1) Floor: what does always guessing the most common class score?
    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    results = {"majority_class": {"val": scores(val["label"], dummy.predict(val["text"]))}}

    # 2) Two simple baselines; choose by VALIDATION macro-F1 (the test set is touched only once)
    candidates = {
        "tfidf_logreg": make_model(LogisticRegression(max_iter=2000, class_weight="balanced")),
        "tfidf_linearsvc": make_model(LinearSVC(class_weight="balanced")),
    }
    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(train["text"], train["label"])
        s = scores(val["label"], model.predict(val["text"]))
        results[name] = {"val": s}
        print(f"\n{name}: validation {s}")
        if s["macro_f1"] > best_f1:
            best_name, best_model, best_f1 = name, model, s["macro_f1"]

    # 3) Final, one-time test evaluation of the chosen baseline
    pred = best_model.predict(test["text"])
    results["chosen_baseline"] = best_name
    results[best_name]["test"] = scores(test["label"], pred)
    results["n_classes"] = int(df["label"].nunique())
    results["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    print(f"\n=== CHOSEN BASELINE: {best_name} ===\nTEST: {results[best_name]['test']}")
    print("\n" + classification_report(test["label"], pred, zero_division=0))

    (out / "day1_metrics.json").write_text(json.dumps(results, indent=2))
    (out / "day1_classification_report.json").write_text(
        json.dumps(classification_report(test["label"], pred, output_dict=True, zero_division=0), indent=2))

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(test["label"], pred, xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title(f"Baseline confusion matrix ({best_name}, test set)")
    plt.tight_layout()
    plt.savefig(out / "day1_confusion_matrix.png", dpi=150)
    plt.close()

    # Wrong predictions, saved for tomorrow's error analysis
    wrong = test.assign(predicted=pred)
    wrong = wrong[wrong["label"] != wrong["predicted"]]
    wrong.head(100).to_csv(out / "day1_misclassified_sample.csv", index=False)
    print(f"Saved metrics, confusion matrix and {min(len(wrong), 100)} misclassified examples to {out}/")


if __name__ == "__main__":
    main()
