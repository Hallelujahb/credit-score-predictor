# Kaza Credit — Copilot Context

This is Kaza Credit, Ethiopia's alternative credit scoring engine. It uses
machine learning to convert mobile money transaction data (Telebirr, M-Birr,
CBE Birr) into formal credit scores for Ethiopian micro-entrepreneurs.

## Stack
- Django 4.2 backend
- CatBoost gradient boosting model
- Django REST Framework for API
- django-q2 for scheduled retraining
- Ethiopian financial features (ETB currency, iqub savings groups, Telebirr)

## Key Rules
- All currency is in Ethiopian Birr (ETB)
- Feature names must reference Ethiopian mobile money context
- Credit scores are on a 300-850 scale
- The model predicts repayment probability for Ethiopian borrowers
- Training data is synthetic until pilot phase (Q1 2025)
- SHAP values must be returned with every score for explainability

## Feature Columns (always use these exact names)
monthly_income_etb, telebirr_tx_count_30d, telebirr_avg_tx_value_etb,
iqub_participation, iqub_months_active, merchant_category, region,
utility_bill_paid_on_time, mobile_topup_count_30d, income_stability_score,
business_segment