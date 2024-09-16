import cv2
import numpy as np
import os
from django.conf import settings

STANDARD_IMAGES_PATH = os.path.join(settings.MEDIA_ROOT, 'standard_images')

standard_img = [
    "2S2.png", "2S3.png", "2S4.png", "2S5.png",
    "3S2.png", "3S3.png", "3S4.png", "3S5.png",
    "4S2.png", "4S3.png", "4S4.png", "4S5.png",
    "5S2.png", "5S3.png", "5S4.png", "5S5.png"
]

def load_standard_images():
    standard_images = []
    for img_name in standard_img:
        img_path = os.path.join(STANDARD_IMAGES_PATH, img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                standard_images.append((img_name, img))
            else:
                print(f"Error: Could not load standard image at {img_path}")
        else:
            print(f"Error: Standard image not found at {img_path}")
    return standard_images

def extract_and_classify_blisters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    classified_contours = {'small': [], 'medium': [], 'large': []}
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Threshold for small blisters
            classified_contours['small'].append(contour)
        elif area < 500:  # Threshold for medium blisters
            classified_contours['medium'].append(contour)
        else:  # Large blisters or defects
            classified_contours['large'].append(contour)
    
    return classified_contours

def detect_and_classify_blisters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    classified_contours = {'small': [], 'medium': [], 'large': []}
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            classified_contours['small'].append(contour)
        elif area < 500:
            classified_contours['medium'].append(contour)
        else:
            classified_contours['large'].append(contour)
    
    # Draw contours on the image
    img_with_contours = img.copy()
    for contour in classified_contours['small']:
        cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 2)  # Green
    for contour in classified_contours['medium']:
        cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 255), 2)  # Yellow
    for contour in classified_contours['large']:
        cv2.drawContours(img_with_contours, [contour], -1, (0, 0, 255), 2)  # Red
    
    return classified_contours, img_with_contours

def compare_blisters(detected_contours, standard_contours):
    score = 0
    for size in ['small', 'medium', 'large']:
        detected_count = len(detected_contours[size])
        standard_count = len(standard_contours[size])
        
        #print(f"Comparing {size}: Detected = {detected_count}, Standard = {standard_count}")
        
        if detected_count == 0 and standard_count == 0:
            score += 1  # Both are zero, treat as a perfect match for this size
        elif detected_count == 0 or standard_count == 0:
            score += 0  # One is zero, other is not, treat as a poor match
        else:
            score += min(detected_count, standard_count) / max(detected_count, standard_count)
    
    return score / 3  # Average score for all sizes

def classify_and_compare(detected_img_path):
    detected_img = cv2.imread(detected_img_path)
    if detected_img is None:
        return None, "Error: Could not load detected image"
    
    detected_contours, detected_img_with_contours = detect_and_classify_blisters(detected_img)
    
    standard_images = load_standard_images()
    
    best_match = None
    best_score = -1
    
    for std_img_name, std_img in standard_images:
        std_contours = extract_and_classify_blisters(std_img)
        score = compare_blisters(detected_contours, std_contours)
        
        if score > best_score:
            best_score = score
            best_match = std_img_name
            best_std_img_with_contours = std_img.copy()
            for contour in std_contours['small']:
                cv2.drawContours(best_std_img_with_contours, [contour], -1, (0, 255, 0), 2)  # Green
            for contour in std_contours['medium']:
                cv2.drawContours(best_std_img_with_contours, [contour], -1, (0, 255, 255), 2)  # Yellow
            for contour in std_contours['large']:
                cv2.drawContours(best_std_img_with_contours, [contour], -1, (0, 0, 255), 2)  # Red
    
    if best_match:
        grade = best_match.split('.')[0]
        result = {
            "standard_image": best_match,
            "similarity_score": best_score,
            "grade": grade,
            "detected_img_with_contours": detected_img_with_contours,
            "best_standard_img_with_contours": best_std_img_with_contours
        }
        return result, None
    else:
        return None, "No matching standard image found"

# the comparison was performed by extracting contours from both the detected image and standard images, 
# classifying these contours into categories (small, medium, large), and then comparing the counts of 
# these categories to calculate a similarity score. This score helps determine how well the detected 
# blisters match the standard images. we can compare the contours as well, rather than number counts