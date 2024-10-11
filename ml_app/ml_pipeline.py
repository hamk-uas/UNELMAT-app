import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from django.conf import settings
import os

# Ensure YOLOv5 is in the Python path
import sys
sys.path.append(str(Path(settings.BASE_DIR) / 'ml_app' / 'yolov5'))

from models.experimental import attempt_load
from utils.general import non_max_suppression

from .scripts.blister_comparison import compare_images


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
            standard_images.append((img_name, img_path))
        else:
            print(f"Error: Standard image not found at {img_path}")
    return standard_images

def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights_path, device=device)
    return model, device

def preprocess_image(image_input, width=480, height=640):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (width, height))
    img_tensor = img_resized.transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).float().div(255.0).unsqueeze(0)
    return img_tensor, img_resized

def detect_blisters(model, img_tensor, device, conf_thres=0.25, iou_thres=0.45):
    img_tensor = img_tensor.to(device)
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    return pred[0].cpu().numpy()

def post_process_detections(detections, min_size=100, max_size=10000, min_ratio=0.5, max_ratio=2.0):
    filtered_detections = []
    for *xyxy, conf, cls in detections:
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        ratio = width / height
        if min_size < area < max_size and min_ratio < ratio < max_ratio:
            filtered_detections.append([*xyxy, conf, cls])
    return np.array(filtered_detections)

def classify_blisters(detections):
    classified_contours = {'small': [], 'medium': [], 'large': []}
    for *xyxy, conf, cls in detections:
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        
        if area < 100:
            classified_contours['small'].append((*xyxy, conf, cls))
        elif area < 500:
            classified_contours['medium'].append((*xyxy, conf, cls))
        else:
            classified_contours['large'].append((*xyxy, conf, cls))
    
    return classified_contours

def compare_blisters(detected_contours, standard_contours):
    score = 0
    for size in ['small', 'medium', 'large']:
        detected_count = len(detected_contours[size])
        standard_count = len(standard_contours[size])
        
        if detected_count == 0 and standard_count == 0:
            score += 1
        elif detected_count == 0 or standard_count == 0:
            score += 0
        else:
            score += min(detected_count, standard_count) / max(detected_count, standard_count)
    
    return score / 3

def visualize_results(image_path, img_resized, detections, best_match_info):
    plt.figure(figsize=(10, 10))
    plt.imshow(img_resized)

    # Draw bounding boxes
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

    plt.axis('off')
    plt.tight_layout()

    # Get folder name and create 'yolo' subdirectory
    folder_name = Path(image_path).parent.name
    yolo_folder = Path(settings.MEDIA_ROOT) / 'folders' / folder_name / 'yolo'
    yolo_folder.mkdir(parents=True, exist_ok=True)

    # Save result image in 'yolo' folder
    output_path = yolo_folder / f"result_{Path(image_path).name}"
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    return str(output_path)


def process_image(image_path, model, device, width=480, height=640):
    img_tensor, img_resized = preprocess_image(image_path, width=width, height=height)
    detections = detect_blisters(model, img_tensor, device)
    filtered_detections = post_process_detections(detections)
    
    standard_images = load_standard_images()
    best_match = None
    best_score = -1
    
    for std_img_name, std_img_path in standard_images:
        std_img_tensor, _ = preprocess_image(std_img_path, width=width, height=height)
        std_detections = detect_blisters(model, std_img_tensor, device)
        std_filtered_detections = post_process_detections(std_detections)
        
        score = compare_images(filtered_detections, std_filtered_detections)
        
        if score > best_score:
            best_score = score
            best_match = std_img_name
    
    grade = best_match.split('.')[0] if best_match else "Unknown"
    best_match_info = {
        "standard_image": best_match,
        "similarity_score": best_score,
        "grade": grade
    }
    
    result_image_path = visualize_results(image_path, img_resized, filtered_detections, best_match_info)
    return filtered_detections, best_match_info, result_image_path

def run_pipeline(image_path, width=480, height=640):
    try:
        weights_path = Path(settings.BASE_DIR) / 'ml_app' / 'yolov5'/ 'runs' / 'fine_tune'/ 'new_foto_blister_detection' /'weights' / 'best.pt'
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        model, device = load_model(weights_path)
        detections, best_match_info, result_image_path = process_image(image_path, model, device, width=width, height=height)
        return detections, best_match_info, result_image_path
    except Exception as e:
        print(f"Error in run_pipeline: {str(e)}")
        raise