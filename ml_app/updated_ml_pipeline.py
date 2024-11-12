import os
import django
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

# Now import Django-related modules
from django.conf import settings

# Add YOLOv5 to path
sys.path.append(str(Path(settings.BASE_DIR) / 'ml_app' / 'yolov5'))

# Rest of your imports
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.general import non_max_suppression
from scripts.blister_classifier import BlisterClassifier

class UpdatedBlisterPipeline:
    def __init__(self):
        self.standard_img = [
            "2S2.png", "2S3.png", "2S4.png", "2S5.png",
            "3S2.png", "3S3.png", "3S4.png", "3S5.png",
            "4S2.png", "4S3.png", "4S4.png", "4S5.png",
            "5S2.png", "5S3.png", "5S4.png", "5S5.png"
        ]
        # Update this path to your standard images folder
        self.STANDARD_IMAGES_PATH = os.path.join(project_dir, 'media', 'standard_images')
        self.classifier = BlisterClassifier()

    def load_model(self, weights_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = attempt_load(weights_path, device=device)
        return model, device

    def preprocess_image(self, image_input, width=480, height=640):
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

    def detect_blisters(self, model, img_tensor, device, conf_thres=0.25, iou_thres=0.45):
        img_tensor = img_tensor.to(device)
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        return pred[0].cpu().numpy()

    def post_process_detections(self, detections, min_size=100, max_size=10000, min_ratio=0.5, max_ratio=2.0):
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

    def load_standard_images(self):
        standard_images = []
        for img_name in self.standard_img:
            img_path = os.path.join(self.STANDARD_IMAGES_PATH, img_name)
            if os.path.exists(img_path):
                standard_images.append((img_name, img_path))
            else:
                print(f"Error: Standard image not found at {img_path}")
        return standard_images

    def visualize_results(self, image_path, img_resized, detections, best_match_info):
        plt.figure(figsize=(10, 10))
        plt.imshow(img_resized)

        # Draw bounding boxes with colors based on size
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            area = (x2-x1) * (y2-y1)
            
            if area < 100:
                color = 'green'
            elif area < 500:
                color = 'yellow'
            else:
                color = 'red'
                
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)

        # Add classification information
        classification = best_match_info['detailed_classification']
        plt.title(f"Grade: {best_match_info['grade']}\n" +
                 f"Confidence: {classification['confidence']:.2f}\n" +
                 f"Stage: {classification['stage']}")

        plt.axis('off')
        plt.tight_layout()

        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(image_path), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the result
        output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return output_path

    def process_image(self, image_path, model, device, width=480, height=640):
        img_tensor, img_resized = self.preprocess_image(image_path, width=width, height=height)
        detections = self.detect_blisters(model, img_tensor, device)
        filtered_detections = self.post_process_detections(detections)
        
        # Get detailed classification
        classification_info = self.classifier.classify_image(filtered_detections)
        
        # Compare with standards
        standard_images = self.load_standard_images()
        best_match = None
        best_score = -1
        
        for std_img_name, std_img_path in standard_images:
            std_img_tensor, _ = self.preprocess_image(std_img_path, width=width, height=height)
            std_detections = self.detect_blisters(model, std_img_tensor, device)
            std_filtered_detections = self.post_process_detections(std_detections)
            
            # Use classifier's comparison
            hist1 = self.classifier.generate_histogram(filtered_detections)
            hist2 = self.classifier.generate_histogram(std_filtered_detections)
            similarity = self.classifier.compare_distributions(hist1, hist2)
            
            if similarity > best_score:
                best_score = similarity
                best_match = std_img_name
        
        grade = best_match.split('.')[0] if best_match else "Unknown"
        best_match_info = {
            "standard_image": best_match,
            "similarity_score": best_score,
            "grade": grade,
            "detailed_classification": classification_info
        }
        
        result_image_path = self.visualize_results(image_path, img_resized, filtered_detections, best_match_info)
        return filtered_detections, best_match_info, result_image_path

    def run_pipeline(self, image_path, width=480, height=640):
        try:
            # Update this path to point to your weights file
            weights_path = current_dir / 'yolov5' / 'runs' / 'fine_tune' / 'new_foto_blister_detection' / 'weights' / 'best.pt'
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found at {weights_path}")
            
            model, device = self.load_model(weights_path)
            detections, classification_info, result_image_path = self.process_image(
                image_path, 
                model, 
                device, 
                width=width, 
                height=height
            )
            return detections, classification_info, result_image_path
        except Exception as e:
            print(f"Error in run_pipeline: {str(e)}")
            raise

