import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
import time
from concurrent.futures import ThreadPoolExecutor

def filter_significant_contours(contours, min_area=1000):
    """Filter out contours that are too small to be considered significant blisters."""
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

def fill_holes_in_contours(contours):
    """Fill the holes between contours during blister detection."""
    if len(contours) <= 1:
        return contours

    merged_contours = []
    rms = set()

    for i in range(len(contours)):
        if i in rms:
            continue

        for j in range(i + 1, len(contours)):
            if j in rms:
                continue

            # Ensure the contour has points and convert to tuple of integers
            if len(contours[i]) > 0 and len(contours[j]) > 0:
                point1 = tuple(map(int, contours[i][0][0]))
                point2 = tuple(map(int, contours[j][0][0]))

                # Ensure the points are valid tuples
                if isinstance(point1, tuple) and isinstance(point2, tuple):
                    dist1 = cv2.pointPolygonTest(contours[j], point1, False)
                    dist2 = cv2.pointPolygonTest(contours[i], point2, False)

                    if dist1 >= 0 or dist2 >= 0:
                        merged_contours.append(cv2.convexHull(np.vstack([contours[i], contours[j]])))
                        rms.add(i)
                        rms.add(j)
                        break

    # Append remaining contours that weren't merged
    for i in range(len(contours)):
        if i not in rms:
            merged_contours.append(contours[i])

    return merged_contours

def process_blister_detection(contours, gray_shape):
    small_blister_contours = []
    large_blister_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.001 * gray_shape[0] * gray_shape[1]:
            small_blister_contours.append(contour)
        else:
            large_blister_contours.append(contour)

    return small_blister_contours, large_blister_contours

def ctrd(contour):
    """Calculate the centroid of a contour."""
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    return (0, 0)

def detect_and_grade_blisters(removal_image_path, detection_path, original_image_name, mask_area):
    start_time = time.time()
    print(f"Starting detection and grading for {original_image_name}...")

    img = cv2.imread(removal_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to load image '{removal_image_path}'.")
        return None

    large_blister_dir = os.path.join(detection_path, 'large_blisters')
    small_blister_dir = os.path.join(detection_path, 'small_blisters')
    combined_blister_dir = os.path.join(detection_path, 'combined_blisters')
    grading_dir = os.path.join(detection_path, 'grading')

    for directory in [large_blister_dir, small_blister_dir, combined_blister_dir, grading_dir]:
        os.makedirs(directory, exist_ok=True)

    median = cv2.medianBlur(img, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    tophat = cv2.morphologyEx(median, cv2.MORPH_TOPHAT, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bottomhat = cv2.morphologyEx(median, cv2.MORPH_BLACKHAT, kernel)
    gray = cv2.add(median, bottomhat)
    gray = cv2.add(gray, -tophat)

    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No contours found for image: {original_image_name}")
        return None

    significant_contours = filter_significant_contours(contours)

    # Parallelize the processing of small and large blisters
    with ThreadPoolExecutor() as executor:
        future_small_large_blisters = executor.submit(process_blister_detection, significant_contours, gray.shape)

        small_blister_contours, large_blister_contours = future_small_large_blisters.result()

    img_small = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    img_large = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_small, small_blister_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(img_large, large_blister_contours, -1, (0, 255, 0), 1)

    small_blister_image_path = os.path.join(small_blister_dir, f'{original_image_name}_sblister.jpg')
    large_blister_image_path = os.path.join(large_blister_dir, f'{original_image_name}_lblister.jpg')

    cv2.imwrite(small_blister_image_path, img_small)
    cv2.imwrite(large_blister_image_path, img_large)

    combined_start_time = time.time()

    # Merge small and large blister contours
    contours = fill_holes_in_contours(small_blister_contours + large_blister_contours)

    img_final = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_final, contours, -1, (0, 255, 0), 1)
    full_blister_image_path = os.path.join(combined_blister_dir, f'{original_image_name}_blister.jpg')
    cv2.imwrite(full_blister_image_path, img_final)

    print(f"Time taken for combining blisters: {time.time() - combined_start_time} seconds")

    grading_start_time = time.time()

    total_blister_area = sum(cv2.contourArea(contour) for contour in contours)
    sarea = total_blister_area / mask_area

    # Calculate centroids and ensure correct shape
    centroids = np.array([ctrd(c) for c in contours])

    if centroids.size == 0:
        print(f"No centroids found for image: {original_image_name}")
        return None

    if centroids.ndim == 1:
        centroids = centroids.reshape(-1, 2)  # Reshape to ensure it's 2D

    db = DBSCAN(eps=100, min_samples=1).fit(centroids)
    labels = np.array(db.labels_)

    print(f"Time taken for grading: {time.time() - grading_start_time} seconds")

    if len(set(labels)) > 1:
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

    grading_file_path = os.path.join(grading_dir, f"{original_image_name}_grade.txt")
    with open(grading_file_path, "w") as file:
        file.write(f'Grade: {grade}\n')
        file.write(f'Area of blisters: {sarea}\n')
        file.write(f'Frequency of blisters: {len(contours)}\n')

    print(f"Saved grading information to: {grading_file_path}")
    print(f"Total time taken for {original_image_name}: {time.time() - start_time} seconds")

    return grade
# CHECK THE MASKING SECITON..........