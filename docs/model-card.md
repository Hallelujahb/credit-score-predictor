# Kaza Credit — Model Card

## Model Overview
- **Type:** CatBoost Gradient Boosting Classifier
- **Task:** Binary classification — predict on-time loan repayment
- **Output:** Credit score (300–850) + risk tier + SHAP explanation
- **Stage:** MVP

## Intended Use
Assist Ethiopian microfinance institutions in assessing credit risk for
small business owners with no formal borrowing history.

**Out of scope:** Automated rejection without human review.

## Training Data
| Property | Value |
|---|---|
| Source | Synthetic Ethiopian mobile money simulation |
| Records | 2,000 (MVP) |
| Real data | Planned Q1 2025 (Addis Ababa pilot) |
| Repayment rate | ~72% (calibrated to Ethiopian MFI averages) |

## Features & Importance
| Feature | Importance |
|---|---|
| telebirr_tx_count_30d | High |
| income_stability_score | High |
| iqub_participation | Medium |
| monthly_income_etb | Medium |
| utility_bill_paid_on_time | Medium |
| mobile_topup_count_30d | Low |

## Evaluation Metrics
| Metric | Value |
|---|---|
| Accuracy | 75.5% |
| Precision | 74.7% |
| Recall | 70.7% |
| F1 Score | 72.6% |
| AUC-ROC | 0.8039 |
> On synthetic data. Real metrics measured during pilot.

## Risk Tiers
| Score | Tier | Action |
|---|---|---|
| 750–850 | Low Risk | Auto-approve |
| 650–749 | Moderate | Approve with conditions |
| 550–649 | Elevated | Manual review |
| <550 | High Risk | Decline + financial coaching |

## Ethical Considerations
- No gender, ethnicity, or religion features used
- Every score includes a SHAP explanation (borrower has right to know why)
- High-risk borrowers referred to financial literacy programs, not just rejected
- Disparate impact by region will be monitored during pilot

## Limitations
1. Trained on synthetic data — real performance unknown until pilot
2. 2,000 training samples is small
3. Does not account for Ethiopian fasting season income seasonality
4. Mobile money data is self-reported at MVP stage

## Roadmap
- Q1 2025: 20–50 borrower pilot, collect real labeled data
- Q2 2025: Retrain + bias audit
- Q3 2025: Telebirr API integration
- Q4 2025: NBE regulatory review
