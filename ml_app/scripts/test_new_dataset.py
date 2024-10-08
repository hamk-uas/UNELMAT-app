import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Add YOLOv5 to the system path
yolov5_path = r"C:\Users\mohan\Desktop\blister\ml_app\yolov5"
sys.path.append(yolov5_path)

# Import YOLOv5 modules
from models.experimental import attempt_load
from utils.general import non_max_suppression

def visualize_detections(img, detections, file_name):
    draw = ImageDraw.Draw(img)
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1-10), f'Blister: {conf:.2f}', fill="red")
    return img

def test_model(model_path, test_images_dir, num_visualize=25, confidence_threshold=0.25):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(model_path, device=device)
    
    # Get list of image files
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    total_images = len(image_files)
    images_with_detections = 0
    total_detections = 0
    confidence_sum = 0
    
    # Randomly select images for visualization
    visualize_indices = random.sample(range(total_images), min(num_visualize, total_images))
    visualized_images = []
    
    # Process images
    for idx, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        img_path = os.path.join(test_images_dir, image_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, confidence_threshold, 0.45)
        
        detections = pred[0].cpu().numpy() if len(pred[0]) else np.array([])
        if len(detections) > 0:
            images_with_detections += 1
            total_detections += len(detections)
            confidence_sum += np.sum(detections[:, 4])
        
        # Visualize selected images
        if idx in visualize_indices:
            visualized_img = visualize_detections(img.copy(), detections, image_file)
            visualized_images.append(visualized_img)
    
    # Calculate and print statistics
    print(f"Total images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Percentage of images with detections: {images_with_detections/total_images*100:.2f}%")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/total_images:.2f}")
    if total_detections > 0:
        print(f"Average confidence: {confidence_sum/total_detections:.4f}")
    
    # Display visualized images in a grid
    rows = int(np.ceil(np.sqrt(len(visualized_images))))
    cols = int(np.ceil(len(visualized_images) / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
    axs = axs.ravel()
    
    for idx, img in enumerate(visualized_images):
        axs[idx].imshow(img)
        axs[idx].axis('off')
    
    for idx in range(len(visualized_images), rows*cols):
        fig.delaxes(axs[idx])
    
    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    model_path = r"C:\Users\mohan\Desktop\blister\ml_app\yolov5\runs\fine_tune\new_foto_blister_detection\weights\best.pt"
    test_images_dir = r"C:\Users\mohan\Desktop\blister\ml_app\datasets\resized_images"
    
    test_model(model_path, test_images_dir, num_visualize=25, confidence_threshold=0.25)