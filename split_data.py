import os
import shutil
import random

def organize_dataset(source_path, destination_path, train_ratio=0.8):
    """
    Organize dataset into train and validation splits.
    
    Args:
        source_path: Path to the source directory containing class folders
        destination_path: Path where train and validation folders will be created
        train_ratio: Ratio of images to use for training (default: 0.8)
    """
    # Create main directories
    train_dir = os.path.join(destination_path, 'train')
    val_dir = os.path.join(destination_path, 'val')
    
    # Get class folder names
    class_folders = [f for f in os.listdir(source_path) 
                    if os.path.isdir(os.path.join(source_path, f))]
    
    # Create directory structure
    for split in ['train', 'val']:
        split_dir = os.path.join(destination_path, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class directories inside train and val
        for class_name in class_folders:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    # Process each class
    for class_name in class_folders:
        # Get all images in the class folder
        class_path = os.path.join(source_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split
        n_train = int(len(images) * train_ratio)
        
        # Split images into train and validation
        train_images = images[:n_train]
        val_images = images[n_train:]
        
        # Copy images to respective directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_name, img)
            shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_name, img)
            shutil.copy2(src, dst)
        
        # Print statistics
        print(f"\nClass: {class_name}")
        print(f"Total images: {len(images)}")
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")

def verify_split(destination_path):
    """
    Verify the dataset split by counting images in each directory
    """
    for split in ['train', 'val']:
        split_dir = os.path.join(destination_path, split)
        print(f"\n{split.capitalize()} set:")
        
        total_images = 0
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            n_images = len([f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
            print(f"{class_name}: {n_images} images")
            total_images += n_images
            
        print(f"Total {split} images: {total_images}")

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    source_path = "dataset/dataset"  # Change this to your dataset path
    destination_path = "dataset/split"  # Change this to where you want the organized dataset
    
    # Create the split
    organize_dataset(source_path, destination_path, train_ratio=0.8)
    
    # Verify the split
    verify_split(destination_path)