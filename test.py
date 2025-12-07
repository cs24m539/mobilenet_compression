import torch
import torch.nn as nn
import argparse
import wandb
from tqdm import tqdm

from models.mobilenet_v2 import MobileNetV2
from utils.dataset import get_cifar10_loaders
from utils.quantization import quantize_model, QuantizationConfig, get_model_size_mb


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('\nLoading CIFAR-10 dataset...')
    _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    
    # Load original model
    print('Loading original model...')
    original_model = MobileNetV2(
        num_classes=10,
        width_multiplier=args.width_multiplier,
        dropout=args.dropout
    ).to(device)
    
    checkpoint = torch.load('best_model.pth', map_location=device)
    original_model.load_state_dict(checkpoint['model_state_dict'])
    original_model.eval()
    
    # Calculate ORIGINAL size BEFORE quantization
    original_size_mb = get_model_size_mb(original_model, weight_bits=32)
    
    # Evaluate original model
    print('Evaluating original model...')
    original_accuracy = evaluate_model(original_model, test_loader, device)
    print(f'Original Model Accuracy: {original_accuracy:.2f}%')
    print(f'Original Model Size: {original_size_mb:.2f} MB')
    
    # Apply quantization
    print(f'\n{"="*80}')
    print(f'Applying quantization (W{args.weight_bits}/A{args.activation_bits})...')
    print(f'{"="*80}')
    
    config = QuantizationConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits
    )
    
    # Quantize model (this modifies the model in-place)
    compressed_model = quantize_model(original_model, config)
    compressed_model = compressed_model.to(device)
    compressed_model.eval()
    
    # Calculate compressed size AFTER quantization
    compressed_size_mb = get_model_size_mb(compressed_model, weight_bits=args.weight_bits)
    
    # Evaluate compressed model
    print('\nEvaluating compressed model...')
    compressed_accuracy = evaluate_model(compressed_model, test_loader, device)
    print(f'Compressed Model Accuracy: {compressed_accuracy:.2f}%')
    
    # Calculate metrics
    model_compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0
    weight_compression_ratio = 32.0 / args.weight_bits
    activation_compression_ratio = 32.0 / args.activation_bits
    accuracy_drop = original_accuracy - compressed_accuracy
    
    # Print results
    print(f'\n{"="*80}')
    print('COMPRESSION RESULTS')
    print(f'{"="*80}')
    print(f'Configuration: {args.weight_bits}-bit weights, {args.activation_bits}-bit activations')
    print(f'\nSize Metrics:')
    print(f'  Original Size:              {original_size_mb:.4f} MB')
    print(f'  Compressed Size:            {compressed_size_mb:.4f} MB')
    print(f'  Size Reduction:             {original_size_mb - compressed_size_mb:.4f} MB ({(1-compressed_size_mb/original_size_mb)*100:.1f}%)')
    print(f'\nCompression Ratios:')
    print(f'  Model Compression:          {model_compression_ratio:.2f}x')
    print(f'  Weight Compression:         {weight_compression_ratio:.2f}x (FP32 → {args.weight_bits}-bit)')
    print(f'  Activation Compression:     {activation_compression_ratio:.2f}x (FP32 → {args.activation_bits}-bit)')
    print(f'\nAccuracy Metrics:')
    print(f'  Original Accuracy:          {original_accuracy:.2f}%')
    print(f'  Compressed Accuracy:        {compressed_accuracy:.2f}%')
    print(f'  Accuracy Drop:              {accuracy_drop:.2f}%')
    print(f'  Relative Drop:              {(accuracy_drop/original_accuracy)*100:.2f}%')
    print(f'{"="*80}')
    
    # Log to wandb
    if args.use_wandb:
        wandb.init(
            project="mobilenet-compression",
            name=f"W{args.weight_bits}_A{args.activation_bits}",
            config={
                "weight_bits": args.weight_bits,
                "activation_bits": args.activation_bits,
                "width_multiplier": args.width_multiplier
            }
        )
        
        wandb.log({
            'original_accuracy': original_accuracy,
            'compressed_accuracy': compressed_accuracy,
            'accuracy_drop': accuracy_drop,
            'original_size_mb': original_size_mb,
            'compressed_size_mb': compressed_size_mb,
            'model_compression_ratio': model_compression_ratio,
            'weight_compression_ratio': weight_compression_ratio,
            'activation_compression_ratio': activation_compression_ratio,
            'weight_bits': args.weight_bits,
            'activation_bits': args.activation_bits
        })
        
        wandb.finish()
    
    # Save compressed model if requested
    if args.save_model:
        save_path = f'compressed_model_w{args.weight_bits}_a{args.activation_bits}.pth'
        torch.save({
            'model_state_dict': compressed_model.state_dict(),
            'config': config,
            'accuracy': compressed_accuracy,
            'compression_ratio': model_compression_ratio
        }, save_path)
        print(f'\nCompressed model saved to: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test compressed MobileNetV2')
    parser.add_argument('--weight_bits', type=int, default=8, 
                       help='bits for weight quantization (2-8)')
    parser.add_argument('--activation_bits', type=int, default=8, 
                       help='bits for activation quantization (2-8)')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='batch size for evaluation')
    parser.add_argument('--width_multiplier', type=float, default=1.0, 
                       help='width multiplier for MobileNet')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='dropout rate')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='log results to wandb')
    parser.add_argument('--save_model', action='store_true', 
                       help='save compressed model')
    parser.add_argument('--verbose', action='store_true', 
                       help='print detailed layer information')
    
    args = parser.parse_args()
    main(args)