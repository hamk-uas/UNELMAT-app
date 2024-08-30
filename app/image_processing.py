import cv2
import numpy as np
import os
import datetime
from django.conf import settings
from .models import Image, ProcessedImage


def create_directories_if_not_exist(*paths):
    """Ensure that all specified directories exist, creating them if necessary."""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def process_image_with_contours(image_obj, contour_dir, region_dir):
    """Process an image to detect and exclude edge defects, and extract ROI."""
    try:
        image_path = image_obj.image.path

        print(f"Loading image from path: {image_path}")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Failed to load image '{image_path}'.")
            return None

        gray = img

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # Apply morphological operations to close gaps in the edges
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)

        # Invert the edge image to create a mask
        mask = cv2.bitwise_not(edges_closed)

        # Use the mask to focus on the center of the image, excluding edges
        h, w = img.shape[:2]
        mask_center = np.zeros_like(gray)
        cv2.rectangle(mask_center, (int(w * 0.1), int(h * 0.1)), (int(w * 0.9), int(h * 0.9)), 255, -1)
        mask_combined = cv2.bitwise_and(mask, mask_center)

        # Find contours from the combined mask
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No contours found in image {image_path}")
            return None

        # Find the bounding box of the remaining contours
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # Clamp the bounding box coordinates to ensure they are within image bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)

        # Draw the bounding box on the original image (for visualization)
        img_contour = img.copy()
        cv2.drawContours(img_contour, contours, -1, (255), 2)
        cv2.rectangle(img_contour, (x, y), (x2, y2), (255), 2)

        # Extract the region of interest (ROI)
        img_region = img[y:y2, x:x2]

        # Save images
        original_filename = os.path.splitext(os.path.basename(image_obj.image.name))[0]

        contour_image_path = os.path.join(contour_dir, f'{original_filename}_contour.jpg')
        region_image_path = os.path.join(region_dir, f'{original_filename}_region.jpg')
        cv2.imwrite(contour_image_path, img_contour)
        cv2.imwrite(region_image_path, img_region)

        # Create and save ProcessedImage instance
        processed_image = ProcessedImage(
            original_image=image_obj,
            contour_image=contour_image_path,
            region_image=region_image_path,
            processed_at=datetime.datetime.now()
        )
        processed_image.save()

        return region_image_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def template_matching(image, templates_X, templates_L, region_dir, rec_path, removal_path):
    """
    Mask the X-marker and the L-marker, return the total area of the mask.
    """
    region_img_path = os.path.join(region_dir, f'{image}_region.jpg')
    
    if not os.path.exists(region_img_path):
        print(f"Error: Region image file does not exist: {region_img_path}")
        return None, 0
    
    img = cv2.imread(region_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to load region image '{region_img_path}'.")
        return None, 0

    img_rec = img.copy()
    img_removal = img.copy()

    maxX = 0
    maxL = 0

    locX = (0, 0)
    locL = (0, 0)

    Xh, Xw = 0, 0
    Lh, Lw = 0, 0

    mask = []

    # Template matching for X markers
    for template_path in templates_X:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result == np.max(result))

        if maxX < np.max(result):
            maxX = np.max(result)
            locX = (loc[1][0], loc[0][0])
            Xh = template.shape[1]
            Xw = template.shape[0]

    if maxX > 0.1:
        cv2.rectangle(img_rec, locX, (locX[0] + Xh, locX[1] + Xw), (255), 2)
        # Apply masking: fill the area within the detected X-marker with the mean gray value
        img_removal[locX[1]:locX[1] + Xw, locX[0]:locX[0] + Xh] = np.mean(img_removal)
        mask.append(Xh * Xw)

    # Template matching for L markers
    for template_path in templates_L:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result == np.max(result))

        if maxL < np.max(result):
            maxL = np.max(result)
            locL = (loc[1][0], loc[0][0])
            Lh = template.shape[1]
            Lw = template.shape[0]

    if maxL > 0.1:
        cv2.rectangle(img_rec, locL, (locL[0] + Lh, locL[1] + Lw), (255), 2)
        # Apply masking: fill the area within the detected L-marker with the mean gray value
        img_removal[locL[1]:locL[1] + Lw, locL[0]:locL[0] + Lh] = np.mean(img_removal)
        mask.append(Lh * Lw)

    # Save the images
    rec_image_path = os.path.join(rec_path, f'{image}_rec.jpg')
    removal_image_path = os.path.join(removal_path, f'{image}_removal.jpg')
    cv2.imwrite(rec_image_path, img_rec)
    cv2.imwrite(removal_image_path, img_removal)

    return removal_image_path, sum(mask)


def process_images_in_folder(folder_path):
    """Process all images in the specified folder for contours and template matching."""
    if not os.path.exists(folder_path):
        print(f"Input folder does not exist: {folder_path}")
        return {}

    file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]

    # Define output directories
    output_dir = os.path.join(folder_path, 'processed')
    contour_dir = os.path.join(output_dir, 'contours')
    region_dir = os.path.join(output_dir, 'regions')
    rec_dir = os.path.join(output_dir, 'rectangles')
    removal_dir = os.path.join(output_dir, 'removals')

    create_directories_if_not_exist(output_dir, contour_dir, region_dir, rec_dir, removal_dir)

    templates_path = os.path.join(settings.MEDIA_ROOT, 'xl_templates')
    templates_X = [os.path.join(templates_path, f) for f in os.listdir(templates_path) if 'X' in f and (f.endswith('.jpg') or f.endswith('.png'))]
    templates_L = [os.path.join(templates_path, f) for f in os.listdir(templates_path) if 'L' in f and (f.endswith('.jpg') or f.endswith('.png'))]

    areas_of_mask = {}

    # Process each image
    for file_name in file_names:
        image_name, _ = os.path.splitext(file_name)
        image_obj = Image.objects.filter(image__iendswith=file_name.lower()).first()
        if not image_obj:
            print(f"Image object not found for file {file_name}")
            continue

        # Process the image and obtain the region image through contour detection
        region_image_path = process_image_with_contours(image_obj, contour_dir, region_dir)

        if region_image_path:
            # Perform template matching using the region image
            removal_image_name, mask_sum = template_matching(image_name, templates_X, templates_L, region_dir, rec_dir, removal_dir)
            if removal_image_name and mask_sum > 0:
                areas_of_mask[image_name] = mask_sum

    print('Processing complete.')
    print('Mask areas:', areas_of_mask)

    return areas_of_mask
