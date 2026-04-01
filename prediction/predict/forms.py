from django import forms
from .models import CreditApplicant, TrainingDataset

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


class DatasetUploadForm(forms.Form):
    csv_file = forms.FileField(label='Credit Applicant CSV file')

    def clean_csv_file(self):
        import pandas as pd

        uploaded_file = self.cleaned_data['csv_file']
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            raise forms.ValidationError('Upload a valid CSV file.') from exc

        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise forms.ValidationError(f"CSV must include columns: {', '.join(REQUIRED_COLUMNS)}")

        uploaded_file.seek(0)
        return uploaded_file


class CreditApplicantForm(forms.ModelForm):
    class Meta:
        model = CreditApplicant
        fields = [
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
