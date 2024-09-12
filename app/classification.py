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

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_images(detected_features, standard_features):
    return cv2.compareHist(detected_features, standard_features, cv2.HISTCMP_CORREL)

def classify_and_grade(detected_img_path):
    detected_img = cv2.imread(detected_img_path)
    if detected_img is None:
        return None, "Error: Could not load detected image"

    standard_images = load_standard_images()
    
    detected_features = extract_features(detected_img)
    
    best_match = None
    best_score = -1
    
    for std_img_name, std_img in standard_images:
        std_features = extract_features(std_img)
        score = compare_images(detected_features, std_features)
        
        if score > best_score:
            best_score = score
            best_match = std_img_name
    
    if best_match:
        grade = best_match.split('.')[0]  
        result = {
            "standard_image": best_match,
            "similarity_score": best_score,
            "grade": grade
        }
        return result, None
    else:
        return None, "No matching standard image found"