import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.files import File
from django.utils import timezone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump

from .models import TrainingDataset, TrainingRun

logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'ml_model.joblib'
SCALER_PATH = APP_DIR / 'scaler.joblib'
DATA_COLUMNS = [
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
DATA_FEED_DIR = Path(settings.BASE_DIR) / 'data_feed'
RETRAIN_INTERVAL_SECONDS = int(os.environ.get('RETRAIN_INTERVAL_SECONDS', 3600))


def ensure_feed_dir():
    DATA_FEED_DIR.mkdir(parents=True, exist_ok=True)


def read_dataset_file(file_path):
    df = pd.read_csv(file_path)
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset file {file_path} is missing required columns: {missing}")

    df = df[DATA_COLUMNS].dropna()
    if df.empty:
        raise ValueError(f"Dataset file {file_path} contains no usable rows")

    df = df.astype({
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
    return df


def generate_synthetic_data():
    np.random.seed(42)
    monthly_income = np.random.normal(5000, 1500, 1000).clip(1000, 15000)
    avg_utility_bill = np.random.normal(150, 50, 1000).clip(50, 500)
    mobile_spend = np.random.normal(120, 60, 1000).clip(20, 500)
    savings_ratio = np.random.uniform(0.05, 0.40, 1000)
    savings_amount = (monthly_income * savings_ratio).round(2)
    years_at_job = np.random.randint(0, 31, 1000)
    utilization_rate = np.random.uniform(0.05, 0.90, 1000).round(4)
    days_since_overdraft = np.random.randint(0, 365, 1000)
    is_entrepreneur = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    past_on_time_payments = np.random.randint(0, 61, 1000)

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
        np.random.normal(0, 5, 1000)
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


def load_training_data():
    datasets = TrainingDataset.objects.all()
    if not datasets.exists():
        return generate_synthetic_data(), 0, 1000

    frames = []
    row_count = 0
    for dataset in datasets:
        dataset_path = dataset.file.path
        if not Path(dataset_path).exists():
            logger.warning('Skipping missing dataset file: %s', dataset_path)
            continue
        frame = read_dataset_file(dataset_path)
        row_count += len(frame)
        frames.append(frame)

    if not frames:
        return generate_synthetic_data(), 0, 1000

    combined = pd.concat(frames, ignore_index=True)
    return combined, datasets.count(), row_count


def ingest_new_datasets_from_feed():
    ensure_feed_dir()
    created = 0
    for csv_path in sorted(DATA_FEED_DIR.glob('*.csv')):
        if TrainingDataset.objects.filter(name=csv_path.name).exists():
            continue

        with csv_path.open('rb') as opened_file:
            dataset = TrainingDataset()
            dataset.file.save(csv_path.name, File(opened_file), save=False)
            try:
                dataset.rows = sum(1 for _ in open(dataset.file.path)) - 1
            except Exception:
                dataset.rows = 0
            dataset.save()
            created += 1
            logger.info('Ingested dataset %s from feed', csv_path.name)
    return created


def train_model(data_frame):
    X = data_frame[
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
    y = data_frame['credit_score'].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    dump(scaler, SCALER_PATH)
    dump(model, MODEL_PATH)


def retrain_model():
    run = TrainingRun.objects.create()
    try:
        ingest_new_datasets_from_feed()
        data, dataset_count, row_count = load_training_data()

        train_model(data)

        TrainingDataset.objects.filter(status=TrainingDataset.STATUS_PENDING).update(
            status=TrainingDataset.STATUS_PROCESSED,
            processed_at=timezone.now(),
        )

        run.status = TrainingRun.STATUS_SUCCESS
        run.dataset_count = dataset_count
        run.row_count = row_count
        run.message = f"Trained on {dataset_count} dataset(s), {row_count} row(s)."
    except Exception as exc:
        logger.exception('Retraining failed')
        run.status = TrainingRun.STATUS_FAILED
        run.message = str(exc)
    finally:
        run.completed_at = timezone.now()
        run.save()
    return run


def start_retraining_scheduler():
    logger.info('Starting retraining scheduler every %s seconds', RETRAIN_INTERVAL_SECONDS)
    while True:
        try:
            retrain_model()
        except Exception:
            logger.exception('Scheduled retraining cycle failed')
        time.sleep(RETRAIN_INTERVAL_SECONDS)
