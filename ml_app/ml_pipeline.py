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
from .scripts.blister_classifier import BlisterClassifier

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
    if len(detections) == 0:
        return np.array([])
        
    filtered_detections = []
    for *xyxy, conf, cls in detections:
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        ratio = width / height
        if min_size < area < max_size and min_ratio < ratio < max_ratio:
            filtered_detections.append([*xyxy, conf, cls])
    return np.array(filtered_detections)

def visualize_results(image_path, img_resized, detections, best_match_info):
    plt.figure(figsize=(10, 10))
    plt.imshow(img_resized)

    # Draw bounding boxes with enhanced color coding
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        area = (x2-x1) * (y2-y1)
        
        # Color coding based on refined size categories
        if area < 100:
            color = 'blue'       # tiny
        elif area < 250:
            color = 'green'      # small
        elif area < 500:
            color = 'yellow'     # medium
        elif area < 1000:
            color = 'orange'     # large
        else:
            color = 'red'        # very large
            
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)

    # Enhanced result display
    if 'detailed_classification' in best_match_info:
        classification = best_match_info['detailed_classification']
        spatial_features = classification.get('spatial_features', {})
        
        title = f"Grade: {best_match_info['grade']}\n"
        title += f"Classification Confidence: {classification['confidence']:.2f}\n"
        title += f"Detections: {len(detections)}"
        
        if spatial_features.get('density', 0) > 0:
            title += f"\nDensity: {spatial_features['density']:.3f}"
    else:
        title = f"Grade: {best_match_info['grade']}\n"
        title += f"Similarity: {best_match_info['similarity_score']:.2f}"

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    # Save result
    folder_name = Path(image_path).parent.name
    yolo_folder = Path(settings.MEDIA_ROOT) / 'folders' / folder_name / 'yolo'
    yolo_folder.mkdir(parents=True, exist_ok=True)
    output_path = yolo_folder / f"result_{Path(image_path).name}"
    plt.savefig(str(output_path), bbox_inches='tight', pad_inches=0)
    plt.close()

    return str(output_path)

def process_image(image_path, model, device, width=480, height=640):
    img_tensor, img_resized = preprocess_image(image_path, width=width, height=height)
    detections = detect_blisters(model, img_tensor, device)
    filtered_detections = post_process_detections(detections)
    
    # Handle zero detections
    if len(filtered_detections) == 0:
        return filtered_detections, {
            "standard_image": "No Detection",
            "similarity_score": 0.0,
            "grade": "Unknown",
            "detailed_classification": {
                'grade': 'Unknown',
                'stage': 0,
                'confidence': 0.0,
                'spatial_features': {'n_clusters': 0, 'avg_cluster_size': 0, 'density': 0}
            }
        }, visualize_results(image_path, img_resized, filtered_detections, {
            "grade": "Unknown",
            "detailed_classification": {'confidence': 0.0, 'stage': 0}
        })

    # Initialize classifier
    classifier = BlisterClassifier()
    
    # Get detailed classification
    classification_info = classifier.classify_image(filtered_detections)
    num_detections = len(filtered_detections)
    
    # Determine potential grades based on detection count
    def get_potential_grades(num_detections, avg_size):
        if num_detections <= 2:
            return ['2S2', '2S3', '2S4']
        elif 3 <= num_detections <= 10:
            return ['2S4', '2S5', '3S2', '3S3']
        elif 11 <= num_detections <= 25:
            return ['3S3', '3S4', '3S5', '4S2']
        elif 26 <= num_detections <= 50:
            return ['4S2', '4S3', '4S4', '4S5']
        else:  # More than 50 detections
            return ['4S4', '4S5', '5S2', '5S3', '5S4', '5S5']

    # Calculate average size
    sizes = [classifier.calculate_size(bbox[:4]) for bbox in filtered_detections]
    avg_size = np.mean(sizes)
    
    # Get potential grades
    potential_grades = get_potential_grades(num_detections, avg_size)
    
    # Compare with standards
    standard_images = load_standard_images()
    best_match = None
    best_score = -1
    detailed_comparisons = {}
    
    # Filter standard images based on potential grades
    filtered_standards = [(name, path) for name, path in standard_images 
                         if any(grade in name for grade in potential_grades)]
    
    for std_img_name, std_img_path in filtered_standards:
        std_img_tensor, _ = preprocess_image(std_img_path, width=width, height=height)
        std_detections = detect_blisters(model, std_img_tensor, device)
        std_filtered_detections = post_process_detections(std_detections)
        
        # Enhanced comparison
        hist1 = classifier.generate_histogram(filtered_detections)
        hist2 = classifier.generate_histogram(std_filtered_detections)
        
        # Calculate similarity with multiple factors
        size_similarity = classifier.compare_distributions(hist1, hist2)
        count_ratio = min(len(filtered_detections), len(std_filtered_detections)) / \
                     max(len(filtered_detections), len(std_filtered_detections))
        
        # Combined similarity score
        if num_detections > 50:
            similarity = 0.6 * size_similarity + 0.4 * count_ratio
        else:
            similarity = 0.7 * size_similarity + 0.3 * count_ratio
        
        detailed_comparisons[std_img_name] = {
            'similarity_score': similarity,
            'histogram_comparison': {
                'test_hist': hist1,
                'standard_hist': hist2
            }
        }
        
        if similarity > best_score:
            best_score = similarity
            best_match = std_img_name
    
    grade = best_match.split('.')[0] if best_match else "Unknown"
    best_match_info = {
        "standard_image": best_match,
        "similarity_score": best_score,
        "grade": grade,
        "detailed_classification": classification_info,
        "comparison_details": detailed_comparisons,
        "num_detections": num_detections,
        "avg_size": avg_size
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