import os
import json
import shutil
import logging
from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from urllib.parse import urljoin
from .models import Folder, MLResult
from .ml_pipeline import run_pipeline

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

                # Ensure the result image is saved in the yolo folder
                result_image_filename = os.path.basename(result_image_path)
                new_result_image_path = os.path.join(yolo_folder_path, result_image_filename)
                shutil.move(result_image_path, new_result_image_path)

                # Get the URL for the result image
                relative_result_path = os.path.relpath(new_result_image_path, settings.MEDIA_ROOT)
                result_url = urljoin(settings.MEDIA_URL, relative_result_path.replace('\\', '/'))

                # Get the URL for the standard image
                standard_image = best_match_info.get('standard_image')
                if standard_image:
                    standard_image_url = urljoin(settings.MEDIA_URL, f'standard_images/{standard_image}')
                else:
                    standard_image_url = None

                result_info = {
                    'image_name': image_file,
                    'result_image': result_url,
                    'num_detections': len(detections),
                    'best_match': {
                        'standard_image': standard_image,
                        'standard_image_url': standard_image_url,
                        'similarity_score': best_match_info.get('similarity_score', 'N/A'),
                        'grade': best_match_info.get('grade', 'N/A')
                    }
                }

                # Save or update the MLResult
                ml_result, created = MLResult.objects.update_or_create(
                    folder=folder,
                    image_name=image_file,
                    defaults={
                        'num_detections': len(detections),
                        'best_match_standard_image': standard_image,
                        'best_match_similarity_score': best_match_info.get('similarity_score', 0.0),
                        'best_match_grade': best_match_info.get('grade', 'N/A'),
                        'result_image_path': relative_result_path,
                    }
                )

                results.append(result_info)

                # Save result info as JSON
                json_filename = f"{os.path.splitext(image_file)[0]}.json"
                json_path = os.path.join(yolo_folder_path, json_filename)
                with open(json_path, 'w') as json_file:
                    json.dump(result_info, json_file)

            except Exception as e:
                logger.error(f"Error processing image {image_file}: {str(e)}")

    context = {
        'folder': folder,
        'results': results,
    }
    
    return render(request, 'ml_app/yolo_results.html', context)

def view_yolo_results(request, folder_id):
    folder = get_object_or_404(Folder, id=folder_id)
    
    yolo_results_path = os.path.join(settings.MEDIA_ROOT, 'folders', folder.name, 'yolo')
    
    results = []
    if os.path.exists(yolo_results_path):
        for filename in os.listdir(yolo_results_path):
            if filename.lower().endswith('.json'):
                json_path = os.path.join(yolo_results_path, filename)
                
                with open(json_path, 'r') as json_file:
                    result_info = json.load(json_file)
                
                # Ensure the result image path is correct
                if result_info['result_image'].startswith('/media/'):
                    result_info['result_image'] = result_info['result_image'].replace('\\', '/')
                else:
                    result_image_path = os.path.join('folders', folder.name, 'yolo', os.path.basename(result_info['result_image']))
                    result_info['result_image'] = urljoin(settings.MEDIA_URL, result_image_path.replace('\\', '/'))

                # Ensure the standard image path is correct
                if result_info['best_match']['standard_image_url'].startswith('/media/'):
                    result_info['best_match']['standard_image_url'] = result_info['best_match']['standard_image_url'].replace('\\', '/')
                else:
                    standard_image_path = os.path.join('standard_images', result_info['best_match']['standard_image'])
                    result_info['best_match']['standard_image_url'] = urljoin(settings.MEDIA_URL, standard_image_path.replace('\\', '/'))
                
                results.append(result_info)
    
    context = {
        'folder': folder,
        'results': results,
    }
    
    return render(request, 'ml_app/yolo_results.html', context)

