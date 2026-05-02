# Kaza Credit — Pilot Plan

## Objective
Collect real repayment labels from a small Ethiopian pilot so that the
synthetic model can be validated and retrained with actual borrower data.

## Pilot Design
- Partner with 1–2 Addis Ababa MFIs or microfinance cooperatives
- Recruit 20–50 small business owners in Merkato and Bole
- Collect consented Telebirr transaction summaries, utility bill payment status, and iqub participation
- Offer parallel manual lending decisions while the model scores borrowers

## Data Collection
- Record borrower features in the same schema as the synthetic dataset
- Track loan repayment for at least 3 months after disbursement
- Label outcome as on-time repayment or delinquent

## Success Criteria
- At least 70% data completeness across feature fields
- Repayment labels collected for at least 80% of pilot borrowers
- Model AUC-ROC improves vs. synthetic baseline after retraining

## Next Steps
1. Use pilot data to retrain the CatBoost model
2. Audit for regional bias and fairness
3. Integrate pilot API flow with Telebirr data ingestion
4. Scale from pilot to a broader Addis/Amhara rollout
