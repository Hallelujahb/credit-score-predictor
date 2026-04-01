from django.contrib import admin
from .models import TrainingDataset, TrainingRun


@admin.register(TrainingDataset)
class TrainingDatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'uploaded_at', 'status', 'processed_at', 'rows')
    list_filter = ('status', 'uploaded_at')
    readonly_fields = ('uploaded_at', 'processed_at')


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ('started_at', 'completed_at', 'status', 'dataset_count', 'row_count')
    readonly_fields = ('started_at', 'completed_at', 'status', 'dataset_count', 'row_count', 'message')
    ordering = ('-started_at',)
