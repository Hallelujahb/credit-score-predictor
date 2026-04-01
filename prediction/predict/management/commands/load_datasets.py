from django.core.management.base import BaseCommand
from predict.trainer import ingest_new_datasets_from_feed


class Command(BaseCommand):
    help = 'Load new CSV dataset files from the data_feed directory into the Django dataset store.'

    def handle(self, *args, **options):
        created = ingest_new_datasets_from_feed()
        self.stdout.write(self.style.SUCCESS(
            f"Ingested {created} new dataset file(s) from data_feed/."
        ))
