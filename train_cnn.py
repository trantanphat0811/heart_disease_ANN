"""
Train CNN model for heart disease prediction from medical images.
Uses PyTorch with ResNet50 backbone for transfer learning.

🚀 QUICK START COMMANDS:
========================
# 1. Install dependencies first
pip install -r requirements.txt

# 2. Run training script
python3 train_cnn.py

# 3. After training, start the server
python3 run_server.py

# 4. Access the system at
open http://localhost:8080

# Notes:
- Training takes ~15-30 minutes on CPU
- GPU (CUDA) will speed up training significantly
- Model will be saved to: /models/cnn_model.pth
- Training logs will display epoch metrics
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
import time
from pathlib import Path
import numpy as np


class MedicalImageDataset(Dataset):
    """Custom dataset for loading medical images with labels."""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load metadata
        metadata_path = os.path.join(root_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Filter by split
        for item in metadata:
            if item['split'] == split:
                self.images.append(item['filename'])
                self.labels.append(item['label'])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class HeartDiseaseCNN(nn.Module):
    """CNN model for heart disease prediction using ResNet50."""
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super(HeartDiseaseCNN, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Optionally freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get number of features from ResNet50
        num_features = self.backbone.fc.in_features
        
        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Binary classification (normal/disease)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    dataset_dir='data/medical_images',
    model_save_path='models/heart_disease_cnn.pt',
    num_epochs=30,
    batch_size=32,
    learning_rate=0.001,
    device=None
):
    """Main training loop."""
    
    # Set device
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
    
    print(f"Using device: {device}")
    
    # Create dataset and dataloaders
    print("\nLoading datasets...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = MedicalImageDataset(dataset_dir, split='train', transform=transform)
    val_dataset = MedicalImageDataset(dataset_dir, split='val', transform=transform)
    test_dataset = MedicalImageDataset(dataset_dir, split='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = HeartDiseaseCNN(pretrained=True, freeze_backbone=False)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 70)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 10
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(
            f"Epoch {epoch+1:2d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history
            }, model_save_path)
            print(f"  ✅ Best model saved (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n⚠️  Early stopping at epoch {epoch+1} (no improvement for {patience_limit} epochs)")
            break
    
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {elapsed/60:.2f} minutes")
    print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Test on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    # Save final history
    history_path = model_save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'test_acc': test_acc,
            'test_loss': test_loss
        }, f, indent=2)
    
    print(f"\n✅ Model saved to: {model_save_path}")
    print(f"📊 History saved to: {history_path}")
    
    # Display completion message
    print("\n" + "="*60)
    print("🎉 CNN MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"📈 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"🏆 Best Epoch: {best_epoch + 1}/50")
    print(f"🧪 Test Set Accuracy: {test_acc:.4f}")
    print("\n🚀 NEXT STEPS:")
    print("  1. Start the server: python3 run_server.py")
    print("  2. Open: http://localhost:8080")
    print("  3. Upload medical images for prediction")
    print("  4. Or use CSV batch prediction")
    print("="*60 + "\n")

    return model, history


if __name__ == '__main__':
    # Generate synthetic dataset if not exists
    dataset_dir = 'data/medical_images'
    if not os.path.exists(dataset_dir):
        print("Generating synthetic medical image dataset...")
        from src.generate_synthetic_images import create_dataset
        create_dataset(dataset_dir, num_samples=1000)
    
    # Train model
    train_model(
        dataset_dir=dataset_dir,
        model_save_path='models/heart_disease_cnn.pt',
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
