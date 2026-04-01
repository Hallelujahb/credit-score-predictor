from django.core.management.base import BaseCommand

from predict.generate_credit_data import generate_credit_dataset
from predict.models import CreditApplicant


class Command(BaseCommand):
    help = 'Generate synthetic credit applicant training data and optionally insert it into the database.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--rows',
            type=int,
            default=300,
            help='Number of synthetic rows to generate.',
        )
        parser.add_argument(
            '--save-csv',
            type=str,
            default=None,
            help='Optional path to save the generated CSV file.',
        )
        parser.add_argument(
            '--insert-db',
            action='store_true',
            help='Insert generated rows directly into the CreditApplicant table.',
        )

    def handle(self, *args, **options):
        rows = options['rows']
        save_csv = options['save_csv']
        insert_db = options['insert_db']

        output_file = save_csv if save_csv else None
        df = generate_credit_dataset(rows, output_file=output_file)

        self.stdout.write(self.style.SUCCESS(f'Generated {rows} synthetic credit rows.'))
        if output_file:
            self.stdout.write(self.style.SUCCESS(f'Saved CSV to: {output_file}'))

        if insert_db:
            applicants = []
            for row in df.to_dict(orient='records'):
                applicants.append(CreditApplicant(
                    monthly_income=row['monthly_income'],
                    avg_utility_bill=row['avg_utility_bill'],
                    mobile_spend=row['mobile_spend'],
                    savings_amount=row['savings_amount'],
                    years_at_job=int(row['years_at_job']),
                    utilization_rate=float(row['utilization_rate']),
                    days_since_overdraft=int(row['days_since_overdraft']),
                    is_entrepreneur=bool(int(row['is_entrepreneur'])),
                    past_on_time_payments=int(row['past_on_time_payments']),
                    credit_score=float(row['credit_score']),
                ))
            CreditApplicant.objects.bulk_create(applicants)
            self.stdout.write(self.style.SUCCESS(f'Inserted {len(applicants)} rows into CreditApplicant.'))
