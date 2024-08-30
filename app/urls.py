from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('create-folder/', views.create_folder, name='create_folder'),
    path('delete-folder/<int:folder_id>/', views.delete_folder, name='delete_folder'),
    path('upload-images/<int:folder_id>/', views.upload_images, name='upload_images'),
    path('view-folder/<int:folder_id>/', views.view_folder, name='view_folder'),
    path('delete-image/<int:image_id>/', views.delete_image, name='delete_image'),
    path('start-processing/', views.start_processing, name='start_processing'),
    path('view-results/<int:folder_id>/', views.view_results, name='view_results'),
    path('processed-folders/', views.processed_folders, name='processed_folders'),
    path('folder/<int:folder_id>/image/<int:image_id>/edit/', views.edit_image_stages, name='edit_image_stages'),
    path('save-edited-image/<str:image_id>/', views.save_edited_image, name='save_edited_image'),







]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
