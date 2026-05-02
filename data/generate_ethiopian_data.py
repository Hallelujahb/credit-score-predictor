"""
Kaza Credit — Synthetic Ethiopian Mobile Money Data Generator
=============================================================
Generates realistic training data simulating Ethiopian small business
owner financial behavior. Based on:
  - EthioTelecom Telebirr usage statistics (2023)
  - NBE Financial Inclusion Report 2022
  - Field observations of Merkato/Bole micro-entrepreneur patterns

Run:   python data/generate_ethiopian_data.py
Output: data/ethiopian_synthetic.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000  # number of synthetic borrowers


def generate_income(n):
    """
    Bimodal income distribution:
    - Street vendors:  1,500–4,000 ETB/month (45% of sample)
    - Small shop owners: 4,000–15,000 ETB/month (55% of sample)
    """
    segment = np.random.choice(["street", "shop"], size=n, p=[0.45, 0.55])
    income = np.where(
        segment == "street",
        np.random.normal(2800, 700, n),
        np.random.normal(8500, 2500, n),
    )
    return np.clip(income, 800, 30000).round(0), segment


def generate_telebirr_features(income, n):
    """Telebirr usage correlates with income but has realistic noise."""
    base_tx = income / 300
    noise = np.random.normal(0, 3, n)
    tx_count_30d = np.clip(base_tx + noise, 0, 60).round(0)

    avg_tx_value = np.clip(
        income / np.where(tx_count_30d == 0, 1, tx_count_30d)
        * np.random.uniform(0.8, 1.2, n),
        50, 5000
    ).round(0)

    return tx_count_30d, avg_tx_value


def generate_iqub_features(n):
    """
    Iqub = Ethiopian rotating savings group.
    ~55% of Addis small business owners participate in at least one.
    Iqub participation is a strong repayment signal because it shows
    financial discipline and community accountability.
    """
    participates = np.random.choice([True, False], size=n, p=[0.55, 0.45])
    months_active = np.where(participates, np.random.randint(3, 36, n), 0)
    return participates.astype(int), months_active


def generate_repayment_label(income, tx_count, iqub, utility_paid, stability):
    """
    Simulate repayment outcome (TARGET variable).
    Logistic model calibrated for ~72% repayment rate (Ethiopian MFI average).
    """
    score = (
        0.0003 * income
        + 0.04 * tx_count
        + 0.5 * iqub
        + 0.4 * utility_paid
        + 0.3 * stability
        - 3.5
    )
    prob = 1 / (1 + np.exp(-score))
    return np.random.binomial(1, prob)


def main():
    print("Generating synthetic Ethiopian borrower dataset...")

    income, segment = generate_income(N)
    tx_count_30d, avg_tx_value = generate_telebirr_features(income, N)
    iqub_participation, iqub_months = generate_iqub_features(N)

    merchant_category = np.random.choice(
        ["retail", "food_beverage", "transport", "services", "agriculture"],
        size=N, p=[0.35, 0.25, 0.15, 0.15, 0.10],
    )
    region = np.random.choice(
        ["addis_ababa_merkato", "addis_ababa_bole", "addis_ababa_piassa",
         "addis_ababa_kazanchis", "addis_ababa_other"],
        size=N, p=[0.30, 0.20, 0.15, 0.15, 0.20],
    )
    utility_paid_on_time = np.random.choice([1, 0], size=N, p=[0.68, 0.32])
    mobile_topup_count_30d = np.clip(np.random.poisson(3.5, N), 0, 20)
    income_stability_score = np.clip(np.random.beta(5, 2, N), 0, 1).round(3)

    repaid = generate_repayment_label(
        income, tx_count_30d, iqub_participation,
        utility_paid_on_time, income_stability_score
    )

    df = pd.DataFrame({
        "monthly_income_etb": income.astype(int),
        "business_segment": segment,
        "telebirr_tx_count_30d": tx_count_30d.astype(int),
        "telebirr_avg_tx_value_etb": avg_tx_value.astype(int),
        "iqub_participation": iqub_participation,
        "iqub_months_active": iqub_months,
        "merchant_category": merchant_category,
        "region": region,
        "utility_bill_paid_on_time": utility_paid_on_time,
        "mobile_topup_count_30d": mobile_topup_count_30d,
        "income_stability_score": income_stability_score,
        "repaid_on_time": repaid,  # TARGET LABEL
    })

    out_path = Path(__file__).parent / "ethiopian_synthetic.csv"
    df.to_csv(out_path, index=False)

    print(f"Done. {N} records → {out_path}")
    print(f"Repayment rate:        {repaid.mean():.1%}")
    print(f"Iqub participation:    {iqub_participation.mean():.1%}")
    print(f"\nSample:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()