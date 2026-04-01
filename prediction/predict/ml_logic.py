from pathlib import Path

import numpy as np
import pandas as pd
from django.utils import timezone
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .models import CreditApplicant, SystemStatus

APP_DIR = Path(__file__).resolve().parent
LINEAR_MODEL_PATH = APP_DIR / 'linear_predictor.joblib'
CATBOOST_MODEL_PATH = APP_DIR / 'catboost_predictor.cbm'

REQUIRED_COLUMNS = [
    'monthly_income',
    'avg_utility_bill',
    'mobile_spend',
    'savings_amount',
    'years_at_job',
    'utilization_rate',
    'days_since_overdraft',
    'is_entrepreneur',
    'past_on_time_payments',
    'credit_score',
]


def generate_synthetic_training_data(num_rows=300):
    np.random.seed(42)
    monthly_income = np.random.normal(5000, 1500, num_rows).clip(1000, 15000)
    avg_utility_bill = np.random.normal(150, 50, num_rows).clip(50, 500)
    mobile_spend = np.random.normal(120, 60, num_rows).clip(20, 500)
    savings_ratio = np.random.uniform(0.05, 0.40, num_rows)
    savings_amount = (monthly_income * savings_ratio).round(2)
    years_at_job = np.random.randint(0, 31, num_rows)
    utilization_rate = np.random.uniform(0.05, 0.90, num_rows).round(4)
    days_since_overdraft = np.random.randint(0, 365, num_rows)
    is_entrepreneur = np.random.choice([0, 1], size=num_rows, p=[0.8, 0.2])
    past_on_time_payments = np.random.randint(0, 61, num_rows)

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
        np.random.normal(0, 5, num_rows)
    )

    credit_score = np.clip(base_score, 0, 100).round(2)
    return pd.DataFrame({
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


def load_training_dataframe():
    queryset = CreditApplicant.objects.all()
    dataframe = pd.DataFrame.from_records(
        queryset.values(
            'monthly_income',
            'avg_utility_bill',
            'mobile_spend',
            'savings_amount',
            'years_at_job',
            'utilization_rate',
            'days_since_overdraft',
            'is_entrepreneur',
            'past_on_time_payments',
            'credit_score',
        )
    )

    if dataframe.empty:
        dataframe = generate_synthetic_training_data()

    dataframe = dataframe.astype({
        'monthly_income': float,
        'avg_utility_bill': float,
        'mobile_spend': float,
        'savings_amount': float,
        'years_at_job': int,
        'utilization_rate': float,
        'days_since_overdraft': int,
        'is_entrepreneur': int,
        'past_on_time_payments': int,
        'credit_score': float,
    })
    return dataframe


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'r2': float(r2_score(y_test, y_pred)),
        'mae': float(mean_absolute_error(y_test, y_pred)),
    }


def train_model_task(catboost_iterations=5):
    dataframe = load_training_dataframe()
    X = dataframe[
        [
            'monthly_income',
            'avg_utility_bill',
            'mobile_spend',
            'savings_amount',
            'years_at_job',
            'utilization_rate',
            'days_since_overdraft',
            'is_entrepreneur',
            'past_on_time_payments',
        ]
    ].astype(float)
    y = dataframe['credit_score'].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    dump(linear, LINEAR_MODEL_PATH)
    linear_metrics = evaluate_model(linear, X_test, y_test)

    from catboost import CatBoostRegressor

    catboost = CatBoostRegressor(
        verbose=0,
        random_state=42,
        iterations=catboost_iterations,
        learning_rate=0.1,
        thread_count=-1,
    )
    catboost.fit(X_train, y_train)
    catboost.save_model(str(CATBOOST_MODEL_PATH))
    catboost_metrics = evaluate_model(catboost, X_test, y_test)

    status, _ = SystemStatus.objects.get_or_create(pk=1)
    status.last_retrained = timezone.now()
    status.linear_r2 = linear_metrics['r2']
    status.catboost_r2 = catboost_metrics['r2']
    status.save()

    return {
        'linear_model_path': str(LINEAR_MODEL_PATH),
        'catboost_model_path': str(CATBOOST_MODEL_PATH),
        'linear_metrics': linear_metrics,
        'catboost_metrics': catboost_metrics,
        'last_retrained': status.last_retrained.isoformat(),
    }
