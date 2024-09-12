import cv2
import numpy as np
import os
import time
from sklearn.cluster import DBSCAN

def draw_blister_contours(img, contours, color, thickness=2):
    marked_img = img.copy()
    cv2.drawContours(marked_img, contours, -1, color, thickness)
    return marked_img

def detect_and_grade_blisters(removal_image_path, detection_path, original_image_name, mask_area):
    start_time = time.time()
    print(f"Processing image: {removal_image_path}")
    print(f"Mask area: {mask_area}")

    # Define directories
    large_blister_dir = os.path.join(detection_path, 'large_blisters')
    small_blister_dir = os.path.join(detection_path, 'small_blisters')
    combined_blister_dir = os.path.join(detection_path, 'combined_blisters')
    grading_dir = os.path.join(detection_path, 'grading')
    defect_dir = os.path.join(detection_path, 'defects')

    # Ensure directories exist
    for directory in [large_blister_dir, small_blister_dir, combined_blister_dir, grading_dir, defect_dir]:
        os.makedirs(directory, exist_ok=True)

    # Read the image
    img = cv2.imread(removal_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to load image '{removal_image_path}'.")
        return None

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Use global thresholding to better detect whitish blisters
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Additional dilation to grow the detected edges outward
    kernel_dilate = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)

    # Morphological operations to clean up the result
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours with the approximation set to capture all edge points
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter and categorize contours
    small_blister_contours = []
    large_blister_contours = []
    defect_contours = []
    
    min_small_blister_area = 200
    max_small_blister_area = 5000
    max_large_blister_area = 7000

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_small_blister_area:  # Ignore very small regions (likely noise)
            continue
        elif min_small_blister_area <= area <= max_small_blister_area:
            small_blister_contours.append(cnt)
        elif max_small_blister_area < area <= max_large_blister_area:
            large_blister_contours.append(cnt)
        elif area > max_large_blister_area:  # Treat as a defect
            defect_contours.append(cnt)

    # Draw contours on separate images
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_small = draw_blister_contours(img_color, small_blister_contours, (0, 255, 0))  # Green for small blisters
    cv2.imwrite(os.path.join(small_blister_dir, f'{original_image_name}_sblister.jpg'), img_small)

    img_large = draw_blister_contours(img_color, large_blister_contours, (0, 255, 255))  # Yellow for large blisters
    cv2.imwrite(os.path.join(large_blister_dir, f'{original_image_name}_lblister.jpg'), img_large)

    img_defect = draw_blister_contours(img_color, defect_contours, (0, 0, 255))  # Red for defects
    cv2.imwrite(os.path.join(defect_dir, f'{original_image_name}_defect.jpg'), img_defect)

    # Combined image with all contours
    img_combined = img_color.copy()
    img_combined = draw_blister_contours(img_combined, small_blister_contours, (0, 255, 0))  # Green
    img_combined = draw_blister_contours(img_combined, large_blister_contours, (0, 255, 255))  # Yellow
    img_combined = draw_blister_contours(img_combined, defect_contours, (0, 0, 255))  # Red
    cv2.imwrite(os.path.join(combined_blister_dir, f'{original_image_name}_blister.jpg'), img_combined)

    # Calculate metrics for grading (excluding defects)
    all_blisters = small_blister_contours + large_blister_contours
    total_blister_area = sum(cv2.contourArea(contour) for contour in all_blisters)
    sarea = total_blister_area / mask_area if mask_area > 0 else 0

    # Calculate centroids for DBSCAN clustering
    centroids = np.array([(cv2.moments(c)['m10']/cv2.moments(c)['m00'], 
                           cv2.moments(c)['m01']/cv2.moments(c)['m00']) 
                          for c in all_blisters 
                          if cv2.moments(c)['m00'] != 0])

    if centroids.size > 0:
        db = DBSCAN(eps=100, min_samples=1).fit(centroids)
        labels = db.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    else:
        num_clusters = 0

    # Grading logic
    if num_clusters > 5 or len(all_blisters) > 10:
        grade = 'Grade 5'
    elif sarea > 0.03:
        grade = 'Grade 4'
    elif sarea > 0.02:
        grade = 'Grade 3'
    elif sarea > 0.01:
        grade = 'Grade 2'
    elif sarea > 0:
        grade = 'Grade 1'
    else:
        grade = 'Grade 0'

    # Save grading information
    grading_file_path = os.path.join(grading_dir, f"{original_image_name}_grade.txt")
    with open(grading_file_path, "w") as file:
        file.write(f'Grade: {grade}\n')
        file.write(f'Area of blisters: {sarea}\n')
        file.write(f'Number of blisters: {len(all_blisters)}\n')
        file.write(f'Number of blister clusters: {num_clusters}\n')

    return grade
