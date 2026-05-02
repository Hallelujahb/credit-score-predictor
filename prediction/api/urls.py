from django.urls import path
from . import views

urlpatterns = [
    path("v1/score/",   views.ScoreView.as_view(),     name="api-score"),
    path("v1/health/",  views.HealthView.as_view(),    name="api-health"),
    path("v1/model/",   views.ModelInfoView.as_view(), name="api-model"),
]