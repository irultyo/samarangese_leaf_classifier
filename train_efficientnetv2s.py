import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

def seed_everything(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resize_with_aspect_ratio(image, target_size):
    """
    Resize image maintaining aspect ratio and then center crop to target size
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Desired output size (height, width)
    Returns:
        PIL.Image: Resized and cropped image
    """
    target_height, target_width = target_size
    
    # Calculate aspect ratios
    aspect_ratio = image.size[0] / image.size[1]  # width / height
    target_aspect = target_width / target_height
    
    if aspect_ratio > target_aspect:
        # Image is wider than target aspect ratio
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        # Image is taller than target aspect ratio
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    
    # Resize maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop to target size
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    return image.crop((left, top, right, bottom))

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(384, 384)):
        """
        Args:
            data_dir (str): Path to the data directory with class subdirectories
            transform: Optional transforms to apply to images
            target_size (tuple): Target size for images (height, width)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Collect all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total images: {len(self.image_paths)}")
        print("Class distribution:")
        for cls_name in self.classes:
            cls_count = len([label for label in self.labels if label == self.class_to_idx[cls_name]])
            print(f"  {cls_name}: {cls_count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply aspect ratio preserving resize and center crop
        image = resize_with_aspect_ratio(image, self.target_size)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class BinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BinaryClassifier, self).__init__()
        self.efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

def create_data_loaders(dataset, train_ratio=0.8, batch_size=32, num_workers=4):
    """
    Split dataset into train and validation sets and create data loaders
    """
    # Calculate lengths for train and validation sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    # Split the dataset
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print("\nSplit sizes:")
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    return train_losses, val_losses

def plot_metrics(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_plot.png')
    plt.close()

def main():
    # Set seeds for reproducibility
    seed_everything(42)
    
    # Configuration
    data_dir = './dataset'  # Your dataset directory
    target_size = (384, 384)
    batch_size = 32
    num_epochs = 20
    train_ratio = 0.8  # 80% for training, 20% for validation
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = ImageDataset(
        data_dir=data_dir,
        transform=train_transform,  # We'll handle transforms differently for train/val splits
        target_size=target_size
    )
    
    # Create data loaders with splits
    train_loader, val_loader = create_data_loaders(
        dataset,
        train_ratio=train_ratio,
        batch_size=batch_size
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Initialize model
    model = BinaryClassifier()
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train the model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )
    
    # Plot training metrics
    plot_metrics(train_losses, val_losses)
    print("\nTraining completed! Check 'training_plot.png' for the loss curves.")

if __name__ == '__main__':
    main()