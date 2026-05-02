"""
Kaza Credit — CatBoost Training Pipeline
==========================================
Trains model on Ethiopian borrower data, evaluates it, saves artifacts.

Run:   python prediction/ml/train.py
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "ethiopian_synthetic.csv"
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL_FEATURES = ["merchant_category", "region", "business_segment"]
TARGET = "repaid_on_time"
FEATURE_COLUMNS = [
    "monthly_income_etb", "telebirr_tx_count_30d", "telebirr_avg_tx_value_etb",
    "iqub_participation", "iqub_months_active", "merchant_category", "region",
    "utility_bill_paid_on_time", "mobile_topup_count_30d",
    "income_stability_score", "business_segment",
]


def load_data():
    if not DATA_PATH.exists():
        print("ERROR: Training data not found.")
        print("Run: python data/generate_ethiopian_data.py")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records | Repayment rate: {df[TARGET].mean():.1%}")
    return df


def train(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_indices = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=30,
    )

    print("\nTraining CatBoost...")
    model.fit(
        Pool(X_train, y_train, cat_features=cat_indices),
        eval_set=Pool(X_test, y_test, cat_features=cat_indices),
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "data_note": "Synthetic Ethiopian mobile money data (proxy phase)",
    }

    print("\n======= KAZA CREDIT MODEL EVALUATION =======")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(classification_report(y_test, y_pred, target_names=["Default", "Repaid"]))

    return model, metrics


def save_artifacts(model, metrics):
    model.save_model(str(MODEL_DIR / "catboost_kaza.cbm"))
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"\nSaved model + metrics to {MODEL_DIR}")


if __name__ == "__main__":
    df = load_data()
    model, metrics = train(df)
    save_artifacts(model, metrics)
    print("\nDone. Start server: python manage.py runserver")