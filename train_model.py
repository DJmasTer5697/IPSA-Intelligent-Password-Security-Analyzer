"""
Train/evaluate the IPSA Random Forest model.

Expected CSV columns:
    password,strength
Strength may be 0/1/2 or weak/medium/strong.

IMPORTANT:
Do not report metrics from a dataset unless the model is actually run.
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.model_selection import train_test_split

from features import extract_features, FEATURE_COLUMNS

LABEL_MAP = {
    "0": 0, "1": 1, "2": 2,
    "weak": 0, "easy": 0,
    "medium": 1, "moderate": 1,
    "strong": 2
}

def load_data(path):
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "password" not in df.columns:
        raise ValueError("CSV must contain a 'password' column.")
    strength_col = "strength" if "strength" in df.columns else "strength_category"
    if strength_col not in df.columns:
        raise ValueError("CSV must contain 'strength' or 'strength_category'.")
    df = df[["password", strength_col]].dropna()
    df["password"] = df["password"].astype(str)
    df["label"] = df[strength_col].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

def make_features(passwords):
    return pd.DataFrame([extract_features(p) for p in passwords], columns=FEATURE_COLUMNS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="passwords.csv")
    ap.add_argument("--out", default="model")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_data(args.data)

    # Remove exact duplicate passwords to reduce leakage.
    df = df.drop_duplicates(subset=["password"]).reset_index(drop=True)

    X = make_features(df["password"].tolist())
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "dataset_rows_after_deduplication": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
    }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        y_test, pred, target_names=["Weak","Moderate","Strong"], zero_division=0
    )
    with open(os.path.join(args.out, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_test, pred, labels=[0,1,2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Weak","Moderate","Strong"])
    disp.plot()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=220)
    plt.close()

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    importance.to_csv(os.path.join(args.out, "feature_importance.csv"), header=["importance"])

    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS},
        os.path.join(args.out, "ipsa_random_forest.joblib")
    )

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", report)
    print("\nTop features:\n", importance.head(10))

if __name__ == "__main__":
    main()
