from django.utils import timezone
from django_q.models import Schedule


def register_train_model_schedule():
    Schedule.objects.update_or_create(
        name='Retrain credit models every hour',
        defaults={
            'func': 'predict.ml_logic.train_model_task',
            'schedule_type': Schedule.MINUTES,
            'minutes': 60,
            'next_run': timezone.now(),
            'repeats': -1,
        },
    )
