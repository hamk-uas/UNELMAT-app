import os
import shutil
import random
import yaml

def organize_dataset(image_dir, label_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate split indices
    train_split = int(len(image_files) * split_ratio[0])
    val_split = int(len(image_files) * (split_ratio[0] + split_ratio[1]))

    # Copy images and labels to respective directories
    for i, image_file in enumerate(image_files):
        if i < train_split:
            split = 'train'
        elif i < val_split:
            split = 'val'
        else:
            split = 'test'

        # Copy image
        src_image = os.path.join(image_dir, image_file)
        dst_image = os.path.join(output_dir, 'images', split, image_file)
        shutil.copy2(src_image, dst_image)

        # Copy corresponding label file if it exists
        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_label = os.path.join(label_dir, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(output_dir, 'labels', split, label_file)
            shutil.copy2(src_label, dst_label)

    print(f"Dataset organized in {output_dir}")

def create_yaml(output_dir):
    yaml_content = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['blister']
    }

    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Created YAML file: {yaml_path}")

if __name__ == "__main__":
    image_dir = r"C:\Users\mohan\Downloads\OneDrive_2024-09-16\New photos 2"
    label_dir = r"C:\Users\mohan\Downloads\labels_my-project-name_2024-10-08-04-25-08"
    output_dir = r"C:\Users\mohan\Desktop\blister\ml_app\datasets\new_foto_data"

    organize_dataset(image_dir, label_dir, output_dir)
    create_yaml(output_dir)