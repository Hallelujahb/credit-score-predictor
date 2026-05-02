# Kaza Credit — Data Strategy

## The Ethiopian Data Problem

Getting labeled repayment data from Ethiopian banks and MFIs is genuinely
difficult. There is no central credit bureau. Data sharing agreements take
months. Institutions are protective of their portfolios.

This is not unique to Ethiopia — M-Shwari in Kenya faced the same challenge
in 2012. They solved it by starting with synthetic validation, then building
from pilot partnerships.

## Our Three-Phase Data Strategy

### Phase 1 — Synthetic Proxy (Current — MVP)
We generate synthetic data using `data/generate_ethiopian_data.py`, calibrated
against published statistics:
- EthioTelecom Telebirr usage reports (2023)
- NBE Financial Inclusion Report (2022)
- IFC Ethiopia MSME Finance Gap Study

This establishes model architecture and validates our feature engineering
before a single real borrower is scored.

### Phase 2 — Pilot Collection (Q1 2025)
20–50 volunteer small business owners in Merkato and Bole, Addis Ababa.
- Borrowers share Telebirr transaction history (consented)
- Loans made by partner MFI using manual process in parallel
- Repayment outcomes tracked for 3 months
- This data is ground truth for V2 model

### Phase 3 — API Integration (Q3 2025)
Direct integration with Telebirr API (pending NBE approval) for automatic,
real-time feature extraction. Eliminates manual data entry entirely.

## Why This Approach Is Credible

M-Shwari (Kenya) — started with synthetic scoring, reached 1M users in 18 months.
Branch (Nigeria) — used airtime top-up data as proxy for income before bank integration.
Kaza Credit applies the same validated pattern to the Ethiopian market.
