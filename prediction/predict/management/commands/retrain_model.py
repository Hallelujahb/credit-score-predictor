from django.core.management.base import BaseCommand
from predict.ml_logic import train_model_task


class Command(BaseCommand):
    help = 'Retrain the credit applicant models from current training data.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--iterations',
            type=int,
            default=5,
            help='Number of CatBoost iterations to train (lower is faster).',
        )
        parser.add_argument(
            '--fast',
            action='store_true',
            help='Use the fastest retrain setting (iteration 5).',
        )

    def handle(self, *args, **options):
        iterations = 5 if options['fast'] else options['iterations']
        self.stdout.write(f"Running retrain_model with {iterations} CatBoost iterations...")
        result = train_model_task(catboost_iterations=iterations)
        self.stdout.write(self.style.SUCCESS(
            f"Retrained successfully. Linear R²: {result['linear_metrics']['r2']:.4f}, CatBoost R²: {result['catboost_metrics']['r2']:.4f}"
        ))
        if result.get('last_retrained'):
            self.stdout.write(f"Last retrained: {result['last_retrained']}")
