from django.urls import path
from . import views

app_name = 'ml_app'  # Ensure the app namespace is set

urlpatterns = [
    path('results/', views.ml_results, name='ml_results'),
    path('process/<int:folder_id>/', views.process_ml, name='process_ml'),    
]