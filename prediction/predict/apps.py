import sys
from django.apps import AppConfig


class PredictConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predict'

    def ready(self):
        if len(sys.argv) >= 2 and sys.argv[1] == 'runserver':
            try:
                from .q_tasks import register_train_model_schedule

                register_train_model_schedule()
            except Exception:
                pass
