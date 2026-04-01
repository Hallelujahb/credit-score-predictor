from pathlib import Path
import numpy as np
from joblib import load

APP_DIR = Path(__file__).resolve().parent
LINEAR_MODEL_PATH = APP_DIR / 'linear_predictor.joblib'


class CreditPredictor:
    def __init__(self):
        if not LINEAR_MODEL_PATH.exists():
            raise FileNotFoundError('Linear regression model not found. Train the model first.')
        self.model = load(LINEAR_MODEL_PATH)

    def predict(
        self,
        monthly_income,
        avg_utility_bill,
        mobile_spend,
        savings_amount,
        years_at_job,
        utilization_rate,
        days_since_overdraft,
        is_entrepreneur,
        past_on_time_payments,
    ):
        input_data = np.array([
            [
                monthly_income,
                avg_utility_bill,
                mobile_spend,
                savings_amount,
                years_at_job,
                utilization_rate,
                days_since_overdraft,
                is_entrepreneur,
                past_on_time_payments,
            ]
        ], dtype=float)
        prediction = self.model.predict(input_data)[0]
        return round(float(prediction), 2)
