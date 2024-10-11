from django.urls import path
from . import views

app_name = 'ml_app'  # Ensure the app namespace is set

urlpatterns = [
    path('process/<int:folder_id>/', views.process_ml, name='process_ml'),    
    path('view-yolo-results/<int:folder_id>/', views.view_yolo_results, name='view_yolo_results'),

]