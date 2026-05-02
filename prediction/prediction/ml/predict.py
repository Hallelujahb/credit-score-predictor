"""
Kaza Credit — Inference Engine with SHAP Explainability
=========================================================
Loads trained model, produces credit scores + factor explanations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier

MODEL_DIR = Path(__file__).parent / "saved_models"

SCORE_BANDS = [
    (750, 850, "low_risk",      "auto_approve"),
    (650, 749, "moderate_risk", "approve_with_conditions"),
    (550, 649, "elevated_risk", "manual_review"),
    (0,   549, "high_risk",     "decline"),
]

_model = None
_feature_columns = None


def _load_model():
    global _model, _feature_columns
    if _model is None:
        model_path = MODEL_DIR / "catboost_kaza.cbm"
        if not model_path.exists():
            raise FileNotFoundError(
                "Model file not found. Run: python prediction/ml/train.py"
            )
        _model = CatBoostClassifier()
        _model.load_model(str(model_path))
        with open(MODEL_DIR / "feature_columns.json") as f:
            _feature_columns = json.load(f)
    return _model, _feature_columns


def probability_to_score(prob: float) -> int:
    """Map repayment probability [0,1] → credit score [300,850]."""
    return int(300 + prob * 550)


def get_risk_tier(score: int):
    for lo, hi, tier, rec in SCORE_BANDS:
        if lo <= score <= hi:
            return tier, rec
    return "high_risk", "decline"


def get_shap_factors(model, input_df, feature_columns):
    """Return top 3 SHAP-driven factors with direction and weight."""
    try:
        shap_vals = model.get_feature_importance(
            data=input_df, type="ShapValues"
        )
        row = shap_vals[0][:-1]  # drop bias term
        top_idx = np.argsort(np.abs(row))[::-1][:3]
        return [
            {
                "feature": feature_columns[i],
                "impact": "positive" if row[i] > 0 else "negative",
                "weight": round(float(abs(row[i])), 4),
            }
            for i in top_idx
        ]
    except Exception:
        # Fallback: use global feature importance
        imp = model.get_feature_importance()
        top_idx = np.argsort(imp)[::-1][:3]
        return [
            {"feature": feature_columns[i], "impact": "positive",
             "weight": round(float(imp[i]), 4)}
            for i in top_idx
        ]


def score_borrower(data: dict) -> dict:
    """
    Main function. Takes borrower dict, returns full score result.

    Args:
        data: dict with borrower features (see API docs)

    Returns:
        dict: credit_score, risk_tier, recommendation, factors, probability
    """
    model, feature_columns = _load_model()

    row = {
        "monthly_income_etb":       data.get("monthly_income_etb", 0),
        "telebirr_tx_count_30d":    data.get("telebirr_tx_count_30d", 0),
        "telebirr_avg_tx_value_etb":data.get("telebirr_avg_tx_value_etb", 0),
        "iqub_participation":       int(data.get("iqub_participation", False)),
        "iqub_months_active":       data.get("iqub_months_active", 0),
        "merchant_category":        data.get("merchant_category", "retail"),
        "region":                   data.get("region", "addis_ababa_other"),
        "utility_bill_paid_on_time":int(data.get("utility_bill_paid_on_time", False)),
        "mobile_topup_count_30d":   data.get("mobile_topup_count_30d", 0),
        "income_stability_score":   data.get("income_stability_score", 0.5),
        "business_segment":         data.get("business_segment", "shop"),
    }

    df = pd.DataFrame([row])[feature_columns]
    prob = float(model.predict_proba(df)[0][1])
    score = probability_to_score(prob)
    tier, recommendation = get_risk_tier(score)
    factors = get_shap_factors(model, df, feature_columns)

    return {
        "credit_score": score,
        "risk_tier": tier,
        "approval_recommendation": recommendation,
        "repayment_probability": round(prob, 4),
        "confidence": round(max(prob, 1 - prob), 4),
        "top_factors": factors,
    }