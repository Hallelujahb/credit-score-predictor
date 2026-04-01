from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_price, name='predict_price'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('add-data/', views.add_data_point, name='add_data_point'),
    path('import-csv/', views.import_csv, name='import_csv'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('retrain-now/', views.retrain_now, name='retrain_now'),
    path('predict-json/', views.predict_api, name='predict_api'),
]