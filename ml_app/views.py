import logging
from django.shortcuts import render, get_object_or_404
from app.models import Folder
from django.http import JsonResponse
from .ml_pipeline import run_pipeline
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
from .models import MLResult

logger = logging.getLogger(__name__)

def process_ml(request, folder_id):
    folder = get_object_or_404(Folder, id=folder_id)
    folder_path = os.path.join(settings.MEDIA_ROOT, 'folders', folder.name)
    yolo_folder_path = os.path.join(folder_path, 'yolo')
    os.makedirs(yolo_folder_path, exist_ok=True)

    results = []

    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, image_file)

            try:
                detections, best_match_info, result_image_path = run_pipeline(image_path)

                # Get the relative path for the result image
                relative_result_path = os.path.relpath(result_image_path, settings.MEDIA_ROOT)
                result_url = os.path.join(settings.MEDIA_URL, relative_result_path)

                results.append({
                    'image_name': image_file,
                    'result_image': result_url,
                    'num_detections': len(detections),
                    'best_match': best_match_info
                })

            except Exception as e:
                print(f"Error processing image {image_file}: {str(e)}")

    context = {
        'folder': folder,
        'results': results,
    }
    
    return render(request, 'ml_app/yolo_results.html', context)

def ml_results(request):
    results = MLResult.objects.all().order_by('-created_at')
    return render(request, 'ml_app/results.html', {'results': results})