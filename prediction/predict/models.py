from django.db import models
from django.utils import timezone


class TrainingDataset(models.Model):
    STATUS_PENDING = 'pending'
    STATUS_PROCESSED = 'processed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSED, 'Processed'),
    ]

    file = models.FileField(upload_to='datasets/')
    name = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    rows = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)

    def save(self, *args, **kwargs):
        if not self.name:
            self.name = self.file.name
        super().save(*args, **kwargs)

    def mark_processed(self):
        self.status = self.STATUS_PROCESSED
        self.processed_at = timezone.now()
        self.save(update_fields=['status', 'processed_at'])

    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    STATUS_STARTED = 'started'
    STATUS_SUCCESS = 'success'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_STARTED, 'Started'),
        (STATUS_SUCCESS, 'Success'),
        (STATUS_FAILED, 'Failed'),
    ]

    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_STARTED)
    dataset_count = models.PositiveIntegerField(default=0)
    row_count = models.PositiveIntegerField(default=0)
    message = models.TextField(blank=True)

    def __str__(self):
        return f"{self.started_at:%Y-%m-%d %H:%M:%S} {self.status}"


class CreditApplicant(models.Model):
    monthly_income = models.DecimalField(max_digits=12, decimal_places=2)
    avg_utility_bill = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True, default=0)
    mobile_spend = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True, default=0)
    savings_amount = models.DecimalField(max_digits=12, decimal_places=2)
    years_at_job = models.IntegerField()
    utilization_rate = models.FloatField()
    days_since_overdraft = models.IntegerField(null=True, blank=True, default=0)
    is_entrepreneur = models.BooleanField(default=False)
    past_on_time_payments = models.IntegerField()
    credit_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Income: {self.monthly_income}, Score: {self.credit_score}"


class CreditApplication(models.Model):
    monthly_income_etb = models.FloatField()
    telebirr_tx_count_30d = models.IntegerField()
    merchant_category = models.CharField(max_length=50)
    region = models.CharField(max_length=100)
    iqub_participation = models.BooleanField(default=False)

    credit_score = models.IntegerField(null=True, blank=True)
    risk_tier = models.CharField(max_length=50, null=True, blank=True)
    approval_recommendation = models.CharField(max_length=100, null=True, blank=True)
    repayment_probability = models.FloatField(null=True, blank=True)

    scored_at = models.DateTimeField(auto_now_add=True)
    api_version = models.CharField(max_length=10, default="1.0")

    class Meta:
        ordering = ['-scored_at']

    def __str__(self):
        return f"Application {self.id} | Score: {self.credit_score} | {self.scored_at.date()}"


class SystemStatus(models.Model):
    last_retrained = models.DateTimeField(null=True, blank=True)
    linear_r2 = models.FloatField(null=True, blank=True)
    catboost_r2 = models.FloatField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Last retrained: {self.last_retrained or 'never'}"
