from datetime import timedelta
from django.utils import timezone
from django_q.models import Schedule


def register_train_model_schedule():
    Schedule.objects.update_or_create(
        name='Retrain credit models every Sunday at midnight',
        defaults={
            'func': 'predict.ml_logic.train_model_task',
            'schedule_type': Schedule.WEEKLY,
            'minutes': 60 * 24 * 7,
            'next_run': _next_sunday_midnight(),
            'repeats': -1,
        },
    )


def _next_sunday_midnight():
    now = timezone.now()
    days_ahead = 6 - now.weekday()
    if days_ahead < 0:
        days_ahead += 7
    next_sunday = now + timedelta(days=days_ahead)
    return next_sunday.replace(hour=0, minute=0, second=0, microsecond=0)
