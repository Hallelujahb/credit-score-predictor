import numpy as np
import pandas as pd
from pathlib import Path


def generate_credit_dataset(n_rows=1000, output_file=None):
    np.random.seed(42)

    monthly_income = np.random.normal(5000, 1500, n_rows).clip(1000, 15000)
    avg_utility_bill = np.random.normal(150, 50, n_rows).clip(50, 500)
    mobile_spend = np.random.normal(120, 60, n_rows).clip(20, 500)
    savings_ratio = np.random.uniform(0.05, 0.40, n_rows)
    savings_amount = (monthly_income * savings_ratio).round(2)
    years_at_job = np.random.randint(0, 31, n_rows)
    utilization_rate = np.random.uniform(0.05, 0.90, n_rows).round(4)
    days_since_overdraft = np.random.randint(0, 365, n_rows)
    is_entrepreneur = np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2])
    past_on_time_payments = np.random.randint(0, 61, n_rows)

    base_score = (
        (monthly_income / 15000) * 20 +
        (avg_utility_bill / 500) * 10 +
        (mobile_spend / 500) * 5 +
        (savings_amount / 6000) * 20 +
        (years_at_job / 30) * 10 +
        (1 - utilization_rate) * 15 +
        (1 - days_since_overdraft / 365) * 10 +
        is_entrepreneur * 5 +
        (past_on_time_payments / 60) * 5 +
        np.random.normal(0, 5, n_rows)
    )

    credit_score = np.clip(base_score, 0, 100).round(2)

    df = pd.DataFrame({
        'monthly_income': monthly_income.round(2),
        'avg_utility_bill': avg_utility_bill.round(2),
        'mobile_spend': mobile_spend.round(2),
        'savings_amount': savings_amount,
        'years_at_job': years_at_job,
        'utilization_rate': utilization_rate,
        'days_since_overdraft': days_since_overdraft,
        'is_entrepreneur': is_entrepreneur,
        'past_on_time_payments': past_on_time_payments,
        'credit_score': credit_score,
    })

    if output_file is None:
        output_file = Path(__file__).resolve().parent / 'credit_data_1000.csv'
    else:
        output_file = Path(output_file)

    df.to_csv(output_file, index=False)
    print(f"Dataset created: {output_file}")
    return df


if __name__ == '__main__':
    generate_credit_dataset()
