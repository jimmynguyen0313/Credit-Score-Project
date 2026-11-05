"""Train and evaluate the credit score models.

    python train.py                # expects data/train.csv
    python train.py --data PATH

Evaluation splits by *customer*: the dataset has 8 monthly rows per
customer, so a random row split would put the same person in both train
and test and overstate accuracy. Test metrics come from customers the
model has never seen. The saved model is then refit on all the data.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_model.features import CLASSES, FEATURE_NAMES, load
from credit_model.model import OrdinalCreditModel

MODEL_PATH = Path("models/credit_model.joblib")
METRICS_PATH = Path("models/metrics.json")


def candidates():
    return {
        "Baseline (always 'Standard')": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)),
        "Gradient boosting (unconstrained)": HistGradientBoostingClassifier(random_state=0),
        "Ordinal gradient boosting (monotonic) ★": OrdinalCreditModel(random_state=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    args = parser.parse_args()

    df = load(args.data)
    X, y, groups = df[FEATURE_NAMES], df["Credit_Score"], df["Customer_ID"]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    print(f"{len(df):,} rows · {groups.nunique():,} customers · "
          f"train {len(X_train):,} / test {len(X_test):,} (split by customer)\n")

    results = []
    for name, model in candidates().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")
        results.append({"model": name, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        print(f"{name:42s} accuracy {acc:.3f}   macro F1 {f1:.3f}")
        if isinstance(model, OrdinalCreditModel):
            chosen_cm = confusion_matrix(y_test, pred, labels=CLASSES).tolist()

    # Refit the chosen model on all data for the app.
    final = OrdinalCreditModel(random_state=0).fit(X, y)
    bundle = {
        "model": final,
        "medians": X.median().to_dict(),
        "sklearn_version": sklearn.__version__,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    metrics = {
        "rows": len(df),
        "customers": int(groups.nunique()),
        "test_rows": len(X_test),
        "class_share": y.value_counts(normalize=True).reindex(CLASSES).round(3).to_dict(),
        "results": results,
        "confusion_matrix": {"labels": CLASSES, "matrix": chosen_cm},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB) and {METRICS_PATH}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
