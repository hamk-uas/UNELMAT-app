import numpy as np
import json  

from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
import os
import cv2
import base64
import shutil
from django.db import IntegrityError
from .models import Folder, Image
from .image_processing import process_images_in_folder
from .detection import detect_and_grade_blisters
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt
import logging
from .classification import classify_and_compare, STANDARD_IMAGES_PATH

def home(request):
    folders = Folder.objects.all()
    return render(request, 'app/home.html', {'folders': folders})

def create_folder(request):
    if request.method == 'POST':
        folder_name = request.POST.get('folder_name')
        try:
            folder, created = Folder.objects.get_or_create(name=folder_name)
            if created:
                # Ensure the folder is created in the filesystem
                folder_path = os.path.join(settings.MEDIA_ROOT, 'folders', folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                return redirect('home')
            else:
                # Folder already exists
                error_message = 'A folder with this name already exists. Please choose a different name.'
                folders = Folder.objects.all()  # Get the folders again to display them
                return render(request, 'app/home.html', {
                    'folders': folders,
                    'error_message': error_message,
                })
        except IntegrityError:
            error_message = 'A folder with this name already exists. Please choose a different name.'
            folders = Folder.objects.all()  # Get the folders again to display them
            return render(request, 'app/home.html', {
                'folders': folders,
                'error_message': error_message,
            })

    return redirect('home')


def delete_folder(request, folder_id):
    folder = get_object_or_404(Folder, id=folder_id)
    folder_path = os.path.join(settings.MEDIA_ROOT, 'folders', folder.name)

    # Delete the folder from the filesystem if it exists
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)  # Removes the folder and all its contents
        except Exception as e:
            print(f"Error deleting folder {folder_path}: {e}")

    # Delete the folder object from the database
    folder.delete()

    return redirect('home')

def upload_images(request, folder_id):
    folder = get_object_or_404(Folder, id=folder_id)
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        for image in images:
            Image.objects.create(folder=folder, image=image)
        return redirect('view_folder', folder_id=folder.id)
    return render(request, 'app/upload_images.html', {'folder': folder})

# def upload_images(request, folder_id):
#     folder = get_object_or_404(Folder, id=folder_id)
#     if request.method == 'POST':
#         images = request.FILES.getlist('images')
#         for image in images:
#             # Convert the uploaded image to a numpy array
#             np_img = np.frombuffer(image.read(), np.uint8)
#             img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#             # Resize the image
#             scale_factor = 0.5  # Scale down to 50% of the original size
#             width = int(img.shape[1] * scale_factor)
#             height = int(img.shape[0] * scale_factor)
#             resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

#             # Encode the image back to the format for saving
#             _, buffer = cv2.imencode('.jpg', resized_img)
#             image_content = ContentFile(buffer.tobytes())

#             # Save the resized image
#             resized_image = InMemoryUploadedFile(
#                 file=image_content,
#                 field_name=None,
#                 name=image.name,
#                 content_type='image/jpeg',
#                 size=image_content.tell(),
#                 charset=None
#             )
#             # Save the image object in the database
#             Image.objects.create(folder=folder, image=resized_image)

#         return redirect('view_folder', folder_id=folder.id)
#     return render(request, 'app/upload_images.html', {'folder': folder})

def view_folder(request, folder_id):
    folder = get_object_or_404(Folder, id=folder_id)
    images = folder.images.all()
    # Add filename attribute to each image object
    for image in images:
        image.filename = os.path.splitext(os.path.basename(image.image.name))[0]
    return render(request, 'app/view_folder.html', {'folder': folder, 'images': images})

def delete_image(request, image_id):
    # Fetch the image object from the database
    image = get_object_or_404(Image, id=image_id)

    # Get the path of the image file in the filesystem
    image_path = image.image.path

    # Delete the image file from the filesystem if it exists
    if os.path.exists(image_path):
        try:
            os.remove(image_path)  # Removes the image file
            print(f"Image file {image_path} has been deleted.")
        except Exception as e:
            print(f"Error deleting image file {image_path}: {e}")

    # Delete the image object from the database
    image.delete()

    # Redirect to the folder view 
    return redirect('view_folder', folder_id=image.folder.id)

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def start_processing(request):
    if request.method == 'POST':
        folder_id = request.POST.get('folder_id')
        folder = get_object_or_404(Folder, id=folder_id)

        folder_path = os.path.join(settings.MEDIA_ROOT, 'folders', folder.name)
        processed_folder_path = os.path.join(folder_path, 'processed')
        removal_path = os.path.join(processed_folder_path, 'removals')
        detection_path = os.path.join(folder_path, 'detection')

        ensure_directory_exists(processed_folder_path)
        ensure_directory_exists(detection_path)

        print("Starting image processing...")
        print("path", folder_path)
        mask_areas = process_images_in_folder(folder_path)
        print(f"Mask areas returned: {mask_areas}")

        if not mask_areas:
            print("Error: No mask areas were returned from process_images_in_folder")
            return redirect('home')

        file_names = [f for f in os.listdir(removal_path) if f.lower().endswith('.jpg')]
        print(f"Files found in removals folder: {file_names}")

        for file_name in file_names:
            image_name = os.path.splitext(file_name)[0].replace('_removal', '')
            
            if image_name in mask_areas:
                removal_image_path = os.path.join(removal_path, file_name)
                mask_area = mask_areas[image_name]
                print(f"Processing {image_name} with mask area {mask_area}")
                detect_and_grade_blisters(removal_image_path, detection_path, image_name, mask_area)
            else:
                print(f"Warning: No mask area found for {image_name}. Using default value.")
                # Use a default mask area or skip this image
                default_mask_area = 1000000  # Adjust this value as needed
                removal_image_path = os.path.join(removal_path, file_name)
                detect_and_grade_blisters(removal_image_path, detection_path, image_name, default_mask_area)

        return redirect('view_results', folder_id=folder.id)

    return redirect('home')


def processed_folders(request):
    processed_folders = Folder.objects.filter(images__isnull=False).distinct()
    
    return render(request, 'app/processed_folders.html', {'processed_folders': processed_folders})


def view_results(request, folder_id):
    folder = get_object_or_404(Folder, id=folder_id)
    images_info = []

    # Loop through all images associated with the folder
    for image_obj in folder.images.all():
        original_image_name = os.path.splitext(os.path.basename(image_obj.image.name))[0]

        # Paths for the combined blister image and grading file
        combined_blister_path = os.path.join(settings.MEDIA_ROOT, 'folders', str(folder.name), 'detection', 'combined_blisters', f'{original_image_name}_blister.jpg')
        grading_file_path = os.path.join(settings.MEDIA_ROOT, 'folders', str(folder.name), 'detection', 'grading', f'{original_image_name}_grade.txt')

        # Check if the combined blister image exists
        combined_blister_url = None
        if os.path.exists(combined_blister_path):
            combined_blister_url = os.path.join(settings.MEDIA_URL, 'folders', str(folder.name), 'detection', 'combined_blisters', f'{original_image_name}_blister.jpg').replace('\\', '/')

        # Load grading information if the file exists
        grading_info = None
        if os.path.exists(grading_file_path):
            with open(grading_file_path, "r") as file:
                grade = file.readline().split(": ")[1].strip()
                area = file.readline().split(": ")[1].strip()
                frequency = file.readline().split(": ")[1].strip()
                grading_info = {
                    'grade': grade,
                    'area': float(area),
                    'frequency': frequency,
                }

        # Append image data to the list
        images_info.append({
            'id': image_obj.id,  # Pass the image ID for linking to the edit page
            'original_image_name': original_image_name,
            'combined_blister_url': combined_blister_url,
            'grading_info': grading_info,
        })

    context = {
        'folder': folder,
        'images_info': images_info,
    }
    return render(request, 'app/view_results.html', context)

logger = logging.getLogger(__name__)







@csrf_exempt
def save_edited_image(request, image_id):
    if request.method == 'POST':
        try:
            # Fetch the image object
            image_obj = Image.objects.get(id=image_id)

            # Parsing JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))
            image_data = data.get('image_data')
            stage = data.get('stage')

            if not image_data or not stage:
                return JsonResponse({'error': 'Invalid data'}, status=400)

            # Decoding the image data from base64
            try:
                format, imgstr = image_data.split(';base64,')
                img_data = base64.b64decode(imgstr)
            except Exception as decode_error:
                logger.error(f'Error decoding image data: {decode_error}')
                return JsonResponse({'error': f'Error decoding image data: {decode_error}'}, status=400)

            # Determine the directory based on the stage
            folder_name = os.path.basename(image_obj.image.path).split('_')[0]
            base_dir = os.path.join(settings.MEDIA_ROOT, 'folders', folder_name, 'processed')

            if stage == 'Masking':
                save_path = os.path.join(base_dir, 'rectangles', f"{image_id}_rectangles.jpg")
            elif stage == 'Background Removal':
                save_path = os.path.join(base_dir, 'contours', f"{image_id}_contour.jpg")
            else:
                return JsonResponse({'error': 'Invalid stage'}, status=400)

            # Ensure the directory exists
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            except OSError as os_error:
                logger.error(f'Error creating directories: {os_error}')
                return JsonResponse({'error': f'Error creating directories: {os_error}'}, status=500)

            # Save the edited image
            try:
                with open(save_path, 'wb') as f:
                    f.write(img_data)
            except Exception as file_error:
                logger.error(f'Error saving image file: {file_error}')
                return JsonResponse({'error': f'Error saving image file: {file_error}'}, status=500)

            return JsonResponse({'success': True, 'message': 'Image saved successfully!'})

        except Image.DoesNotExist:
            logger.error('Image not found')
            return JsonResponse({'error': 'Image not found'}, status=404)
        except json.JSONDecodeError as json_error:
            logger.error(f'Invalid JSON data: {json_error}')
            return JsonResponse({'error': f'Invalid JSON data: {json_error}'}, status=400)
        except Exception as e:
            logger.error(f'Unexpected error: {str(e)}')
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    

from django.conf import settings

def edit_image_stages(request, folder_id, image_id):
    folder = get_object_or_404(Folder, id=folder_id)
    image = get_object_or_404(Image, id=image_id)
    active_stage = request.GET.get('stage', 'Background Removal')
    original_image_name = os.path.splitext(os.path.basename(image.image.name))[0]

    # Helper function to retrieve image paths
    def get_image_path(subfolder, suffix):
        relative_path = os.path.join('folders', folder.name, subfolder, f'{original_image_name}{suffix}')
        full_path = os.path.join(settings.MEDIA_ROOT, relative_path)
        if os.path.exists(full_path):
            return os.path.join(settings.MEDIA_URL, relative_path.replace('\\', '/'))
        return None

    stages = {
        "Background Removal": {
            "Original": image.image.url, 
            "Contours": get_image_path('processed/contours', '_contour.jpg'),
            "Region": get_image_path('processed/regions', '_region.jpg'),
        },
        "Masking": {
            "Region": get_image_path('processed/regions', '_region.jpg'),
            "Rectangles": get_image_path('processed/rectangles', '_rec.jpg'),
            "Removals": get_image_path('processed/removals', '_removal.jpg'),
        },
        "Detection": {
            "Combined_Blisters": get_image_path('detection/combined_blisters', '_blister.jpg'),
            "Large_Blisters": get_image_path('detection/large_blisters', '_lblister.jpg'),
            "Small_Blisters": get_image_path('detection/small_blisters', '_sblister.jpg'),
        },
        "Classification": {
            "Combined_Blisters": get_image_path('detection/combined_blisters', '_blister.jpg'),
        }
    }

    classification_result = None
    classification_error = None
    if active_stage == "Classification":
        combined_blister_path = os.path.join(settings.MEDIA_ROOT, 'folders', str(folder.name), 'detection', 'combined_blisters', f'{original_image_name}_blister.jpg')
        try:
            classification_result, classification_error = classify_and_compare(combined_blister_path)
            if classification_result and 'standard_image' in classification_result:
                standard_image_path = os.path.join(STANDARD_IMAGES_PATH, classification_result['standard_image'])
                if os.path.exists(standard_image_path):
                    classification_result['standard_image_url'] = os.path.join(settings.MEDIA_URL, 'standard_images', classification_result['standard_image'])
                else:
                    classification_error = f"Standard image not found: {classification_result['standard_image']}"
                    classification_result['standard_image'] = None
        except Exception as e:
            classification_error = f"An error occurred during classification: {str(e)}"

    context = {
        'folder': folder,
        'image': image,
        'stages': stages,
        'active_stage': active_stage,
        'original_image_name': original_image_name,
        'classification_result': classification_result,
        'classification_error': classification_error,
    }
    return render(request, 'app/edit_image.html', context)