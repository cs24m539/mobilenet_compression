import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

from models.mobilenet_v2 import MobileNetV2
from utils.dataset import get_cifar10_loaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize wandb
    wandb.init(
        project="mobilenet-cifar10",
        config={
            "architecture": "MobileNetV2",
            "dataset": "CIFAR-10",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "width_multiplier": args.width_multiplier,
            "dropout": args.dropout
        }
    )
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = MobileNetV2(
        num_classes=10,
        width_multiplier=args.width_multiplier,
        dropout=args.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (Cosine Annealing)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save statistics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'best_model.pth')
            print(f'Best model saved with accuracy: {best_acc:.2f}%')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    wandb.log({"training_curves": wandb.Image('training_curves.png')})
    
    print(f'\nTraining completed! Best test accuracy: {best_acc:.2f}%')
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileNetV2 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--width_multiplier', type=float, default=1.0, help='width multiplier')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    args = parser.parse_args()
    main(args)