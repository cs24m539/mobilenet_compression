import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    Prepare CIFAR-10 dataset with normalization and augmentation.
    
    Data Augmentation for Training:
    - RandomCrop: Crop 32x32 after padding by 4 pixels (creates variety)
    - RandomHorizontalFlip: Flip images horizontally with 50% probability
    - Normalize: Mean and std calculated from CIFAR-10 dataset
    
    Test set uses only normalization (no augmentation).
    """
    
    # CIFAR-10 statistics (pre-calculated)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Pad then crop
        transforms.RandomHorizontalFlip(),      # 50% chance of flip
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Test transforms (only normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader