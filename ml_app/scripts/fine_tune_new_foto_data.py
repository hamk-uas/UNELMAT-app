import os
import yaml

# Set up paths
base_dir = r"C:\Users\mohan\Desktop\blister\ml_app"
yolov5_dir = os.path.join(base_dir, "yolov5")
new_data_dir = os.path.join(base_dir, "datasets", "new_foto_data")
data_yaml = os.path.join(new_data_dir, "dataset.yaml")
synthetic_weights = os.path.join(yolov5_dir, "runs", "train", "synthetic_blister_detection4", "weights", "best.pt")

# Ensure the YAML file is correctly configured
def check_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Ensure the path is absolute
    data['path'] = os.path.abspath(new_data_dir)
    
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    print(f"Updated YAML file: {yaml_path}")
    print(f"Dataset path: {data['path']}")

# Change to YOLOv5 directory
os.chdir(yolov5_dir)

# Check and update YAML
check_yaml(data_yaml)

# Fine-tuning command
command = (
    f"python train.py "
    f"--img 640 "
    f"--batch 2 "
    f"--epochs 100 "
    f"--data {data_yaml} "
    f"--weights {synthetic_weights} "
    f"--project runs/fine_tune "
    f"--name new_foto_blister_detection "
    f"--cache "
    f"--patience 20 "
    f"--freeze 10 "  # Freeze first 10 layers
)

# Run fine-tuning
print("Starting fine-tuning...")
os.system(command)
print("Fine-tuning completed.")