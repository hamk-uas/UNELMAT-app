import os

# Set up paths
base_dir = r"C:\Users\mohan\Desktop\blister\ml_app"
yolov5_dir = os.path.join(base_dir, "yolov5")
data_yaml = os.path.join(base_dir, "datasets", "yolo_dataset", "yolo_dataset.yaml")
weights = os.path.join(yolov5_dir, "runs", "train", "synthetic_blister_detection4", "weights", "best.pt")

# Change to YOLOv5 directory
os.chdir(yolov5_dir)

# Fine-tuning command
command = f"python train.py --img 640 --batch 2 --epochs 50 --data {data_yaml} --weights {weights} --project runs/fine_tune --name real_blister_detection"

# Run fine-tuning
print("Starting fine-tuning...")
os.system(command)
print("Fine-tuning completed.")