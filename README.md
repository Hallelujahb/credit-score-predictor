# Kaza Credit — Ethiopia's Alternative Credit Scoring Engine

> Turning mobile money footprints into formal credit identities for Ethiopian entrepreneurs.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-4.2-green.svg)](https://djangoproject.com)
[![Model](https://img.shields.io/badge/Model-CatBoost-yellow.svg)](https://catboost.ai)
[![Stage](https://img.shields.io/badge/Stage-MVP-orange.svg)]()

---

## The Problem

Over 60% of Ethiopian small business owners have never accessed formal credit —
not because they are poor risks, but because banks require collateral and borrowing
history that informal entrepreneurs don't have.

Meanwhile these same entrepreneurs use **Telebirr, M-Birr, and CBE Birr** every day.
They generate rich financial signals. No bank currently reads them.

**Kaza Credit reads them.**

---

## What It Does

Kaza Credit is a machine-learning scoring engine that converts alternative financial
data — mobile money transactions, merchant payment patterns, and *iqub* savings
participation — into a structured credit score that lenders can act on instantly.

```
Borrower's mobile money behavior
        ↓
  Kaza Credit API
        ↓
Credit Score (300–850) + Risk Tier + SHAP Explanation
        ↓
  Lender decides in seconds, not weeks
```

---

## Model Performance

| Metric | CatBoost (Kaza) |
|---|---|
| Accuracy | **75.5%** |
| F1 Score | **72.6%** |
| AUC-ROC  | **0.8039** |

> Trained on synthetic Ethiopian mobile money data (Phase 1). Pilot data collection
> begins Q1 2025 in Addis Ababa.

---

## API

```bash
POST /api/v1/score/
```

```json
{
  "monthly_income_etb": 4500,
  "telebirr_tx_count_30d": 23,
  "iqub_participation": true,
  "merchant_category": "retail",
  "region": "addis_ababa_merkato"
}
```

Response:
```json
{
  "credit_score": 672,
  "risk_tier": "moderate_risk",
  "approval_recommendation": "approve_with_conditions",
  "repayment_probability": 0.6763,
  "top_factors": [
    {"feature": "telebirr_tx_count_30d", "impact": "positive", "weight": 0.31},
    {"feature": "iqub_participation",    "impact": "positive", "weight": 0.16}
  ]
}
```

---

## Quickstart

```bash
git clone https://github.com/Hallelujahb/credit-score-predictor.git
cd credit-score-predictor
python -m venv venv && source venv/bin/activate
python -m pip install -r requirements.txt
python data/generate_ethiopian_data.py
cd prediction
python prediction/ml/train.py
python manage.py migrate
python manage.py runserver
```

Test the API:
```bash
curl -X POST http://localhost:8000/api/v1/score/ \
  -H "Content-Type: application/json" \
  -d '{"monthly_income_etb": 4500, "telebirr_tx_count_30d": 23, "iqub_participation": true, "merchant_category": "retail"}'
```

---

## Project Structure

```
credit-score-predictor/
├── prediction/ml/train.py      ← CatBoost training pipeline
├── prediction/ml/predict.py    ← Inference + SHAP explainability
├── api/views.py                ← REST API endpoints
├── data/generate_ethiopian_data.py  ← Synthetic data generator
└── docs/                       ← Model card, data strategy, pilot plan
```

---

## Ethiopian Market Context

| Signal | Traditional Bank | Kaza Credit |
|---|---|---|
| Formal payslip | Required | Not needed |
| Telebirr history | Ignored | Core feature |
| *Iqub* participation | Ignored | Core feature |
| Mobile top-up frequency | Ignored | Feature |

Ethiopia's mobile money ecosystem crossed **40 million registered users** (2023).
This is the data layer Kaza Credit is built on.

---

## Built By

**Hallelujah** — CS Student, Addis Ababa
EAII AI Incubator Program — 2025
