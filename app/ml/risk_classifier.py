"""
Random Forest risk classifier trained on PaySim transaction data.
Provides a 'fraud probability' that the rule-based RiskEngine can blend
into its composite score.

Train:    python -m app.ml.risk_classifier
Predict:  RiskClassifier.load().predict_proba(features_dict)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "amount_log",
    "hour_of_day",
    "is_off_hours",
    "is_transfer",
    "balance_drained",
    "amount_to_balance",
    "dest_new",
]

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "paysim_features.csv"
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_PATH = MODEL_DIR / "risk_rf.joblib"


class RiskClassifier:
    """Wrap a trained RF model with a clean predict_proba interface."""

    def __init__(self, model: RandomForestClassifier | None = None) -> None:
        self.model = model

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "RiskClassifier":
        if not path.exists():
            raise FileNotFoundError(f"Trained model not found at {path}. Run training first.")
        model = joblib.load(path)
        return cls(model=model)

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return P(fraud) in [0, 1] for a single transaction's features."""
        if self.model is None:
            raise RuntimeError("No model loaded.")
        row = np.array([[features[c] for c in FEATURE_COLS]])
        return float(self.model.predict_proba(row)[0, 1])

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("No model loaded.")
        return dict(zip(FEATURE_COLS, self.model.feature_importances_.tolist()))


def train(
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int | None = 12,
    random_state: int = 42,
    sample_size: int | None = 500_000,  # subsample for speed; None = use all 6.3M
) -> dict[str, Any]:
    print(f"Loading features from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  rows: {len(df):,}  fraud rate: {df['is_fraud'].mean():.4%}")

    if sample_size and sample_size < len(df):
        frauds = df[df.is_fraud == 1]
        non_frauds = df[df.is_fraud == 0].sample(n=sample_size - len(frauds), random_state=random_state)
        df = pd.concat([frauds, non_frauds]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        print(f"  subsampled to {len(df):,} rows (kept all fraud cases)")

    X = df[FEATURE_COLS].values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  train: {len(X_train):,}  test: {len(X_test):,}")

    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",  # critical: 0.13% positive class
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("\nEvaluating on held-out test set...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  ROC AUC: {auc_roc:.4f}")
    print(f"  PR  AUC: {auc_pr:.4f}  (more informative for class-imbalanced data)")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    TN={cm[0, 0]:>8}  FP={cm[0, 1]:>6}")
    print(f"    FN={cm[1, 0]:>8}  TP={cm[1, 1]:>6}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"], digits=4))

    print("Feature importances:")
    importances = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda kv: kv[1], reverse=True)
    for name, score in importances:
        print(f"  {name:<22} {score:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": cm.tolist(),
        "feature_importance": dict(importances),
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    train()