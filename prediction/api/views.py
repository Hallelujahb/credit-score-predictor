"""
Kaza Credit REST API
=====================
POST /api/v1/score/    Score a borrower
GET  /api/v1/health/   Health check
GET  /api/v1/model/    Model metrics
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from prediction.ml.predict import score_borrower

MODEL_DIR = Path(__file__).resolve().parents[1] / "prediction" / "ml" / "saved_models"


@method_decorator(csrf_exempt, name="dispatch")
class ScoreView(View):

    REQUIRED_FIELDS = ["monthly_income_etb", "telebirr_tx_count_30d"]

    def post(self, request):
        try:
            body = json.loads(request.body)
        except (json.JSONDecodeError, ValueError):
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        missing = [f for f in self.REQUIRED_FIELDS if f not in body]
        if missing:
            return JsonResponse(
                {"error": f"Missing required fields: {missing}"}, status=400
            )

        try:
            result = score_borrower(body)
        except FileNotFoundError as e:
            return JsonResponse({"error": str(e), "hint": "Run: python prediction/ml/train.py"}, status=503)
        except Exception as e:
            return JsonResponse({"error": f"Scoring failed: {str(e)}"}, status=500)

        result["scored_at"] = datetime.now(timezone.utc).isoformat()
        result["api_version"] = "1.0"
        return JsonResponse(result)

    def get(self, request):
        return JsonResponse({
            "endpoint": "POST /api/v1/score/",
            "required_fields": self.REQUIRED_FIELDS,
            "example": {
                "monthly_income_etb": 4500,
                "telebirr_tx_count_30d": 23,
                "telebirr_avg_tx_value_etb": 320,
                "merchant_category": "retail",
                "iqub_participation": True,
                "iqub_months_active": 8,
                "mobile_topup_count_30d": 4,
                "utility_bill_paid_on_time": True,
                "income_stability_score": 0.75,
                "region": "addis_ababa_merkato",
                "business_segment": "shop"
            }
        })


class HealthView(View):
    def get(self, request):
        model_ready = (MODEL_DIR / "catboost_kaza.cbm").exists()
        return JsonResponse({
            "status": "ok" if model_ready else "model_not_trained",
            "model_ready": model_ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


class ModelInfoView(View):
    def get(self, request):
        metrics_path = MODEL_DIR / "metrics.json"
        if not metrics_path.exists():
            return JsonResponse({"error": "Model not trained yet."}, status=404)
        with open(metrics_path) as f:
            metrics = json.load(f)
        return JsonResponse({
            "model": "CatBoostClassifier",
            "version": "1.0",
            "metrics": metrics,
        })