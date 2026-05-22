"""
PaySim dataset loader and feature engineering for the risk classifier.

PaySim is a synthetic mobile money simulator dataset (Kaggle: ealaxi/paysim1).
We map its transaction features to our 6 risk signals, then output a clean
labeled dataset for training the Random Forest classifier.

Columns in raw PaySim:
  step          int    Hour of simulation (1..744 = ~31 days)
  type          str    CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
  amount        float  Transaction amount
  nameOrig      str    Originating account
  oldbalanceOrg float  Originator balance before
  newbalanceOrig float Originator balance after
  nameDest      str    Destination account
  oldbalanceDest float Destination balance before
  newbalanceDest float Destination balance after
  isFraud       int    1 if fraudulent (our label)
  isFlaggedFraud int   1 if rule-flagged (>200k transfer)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
RAW_PATH = DATA_ROOT / "raw" / "paysim.csv"
PROCESSED_DIR = DATA_ROOT / "processed"


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    """Load raw PaySim CSV. Pass nrows for quick iteration."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"PaySim CSV not found at {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, nrows=nrows)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw PaySim rows to the 6 risk-signal feature space the engine uses.
    Output columns:
      amount_log         log10(amount+1)
      hour_of_day        step % 24
      is_off_hours       1 if hour in 0..5 else 0
      is_transfer        1 if type in {TRANSFER, CASH_OUT} else 0
      balance_drained    1 if newbalanceOrig == 0 and oldbalanceOrg > 0
      amount_to_balance  amount / (oldbalanceOrg + 1)
      dest_new           1 if oldbalanceDest == 0 (new/unseen destination)
      is_fraud           label (0/1)
    """
    out = pd.DataFrame()
    import numpy as np

    out["amount_log"] = np.log10(df["amount"].clip(lower=0) + 1)
    out["hour_of_day"] = df["step"] % 24
    out["is_off_hours"] = ((out["hour_of_day"] >= 0) & (out["hour_of_day"] < 6)).astype(int)
    out["is_transfer"] = df["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    out["balance_drained"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
    out["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    out["dest_new"] = (df["oldbalanceDest"] == 0).astype(int)
    out["is_fraud"] = df["isFraud"].astype(int)
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Quick stats — class balance, fraud rate, feature ranges."""
    return {
        "rows": len(df),
        "fraud_count": int(df["is_fraud"].sum()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "feature_means": df.drop(columns=["is_fraud"]).mean().to_dict(),
    }


def save_processed(df: pd.DataFrame, name: str = "paysim_features.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / name
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    print(f"Loading PaySim from {RAW_PATH}...")
    raw = load_raw()
    print(f"  raw rows: {len(raw):,}")
    print(f"  columns: {list(raw.columns)}")
    print(f"  fraud rate (raw): {raw['isFraud'].mean():.4%}")

    print("\nEngineering features...")
    features = engineer_features(raw)
    print(f"  output columns: {list(features.columns)}")

    stats = summarize(features)
    print(f"\nProcessed dataset:")
    print(f"  rows: {stats['rows']:,}")
    print(f"  fraud count: {stats['fraud_count']:,}")
    print(f"  fraud rate: {stats['fraud_rate']:.4%}")

    out_path = save_processed(features)
    print(f"\nSaved to: {out_path}")