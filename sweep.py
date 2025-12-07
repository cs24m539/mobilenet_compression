import torch
import wandb
import os
from itertools import product
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
        for inputs, targets in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def run_single_experiment(weight_bits, activation_bits, device, test_loader, original_state_dict, original_size_mb):
    """Run a single compression experiment"""
    
    # Create fresh model
    model = MobileNetV2(num_classes=10, width_multiplier=1.0, dropout=0.2)
    model.load_state_dict(original_state_dict)
    model = model.to(device)
    model.eval()
    
    # Apply compression
    config = QuantizationConfig(weight_bits=weight_bits, activation_bits=activation_bits)
    compressed_model = quantize_model(model, config)
    compressed_model = compressed_model.to(device)
    compressed_model.eval()
    
    # Get compressed size with actual bit-width
    compressed_size_mb = get_model_size_mb(compressed_model, weight_bits=weight_bits)
    
    # Evaluate compressed model
    compressed_accuracy = evaluate_model(compressed_model, test_loader, device)
    
    # Calculate ratios
    model_compression_ratio = original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0
    weight_compression_ratio = 32.0 / weight_bits
    activation_compression_ratio = 32.0 / activation_bits
    
    return {
        'weight_bits': weight_bits,
        'activation_bits': activation_bits,
        'compressed_accuracy': compressed_accuracy,
        'original_size_mb': original_size_mb,
        'compressed_size_mb': compressed_size_mb,
        'model_compression_ratio': model_compression_ratio,
        'weight_compression_ratio': weight_compression_ratio,
        'activation_compression_ratio': activation_compression_ratio
    }


def main():
    """Run compression sweep"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading CIFAR-10 dataset...')
    _, test_loader = get_cifar10_loaders(batch_size=128)
    
    # Load original model
    print('Loading original model...')
    original_model = MobileNetV2(num_classes=10, width_multiplier=1.0, dropout=0.2).to(device)
    
    checkpoint = torch.load('best_model.pth', map_location=device)
    original_model.load_state_dict(checkpoint['model_state_dict'])
    original_model.eval()
    
    # Save state dict for reuse
    original_state_dict = original_model.state_dict()
    
    # Calculate ORIGINAL size (FP32) BEFORE any quantization
    original_size_mb = get_model_size_mb(original_model, weight_bits=32)
    print(f'Original Model Size: {original_size_mb:.2f} MB')
    
    # Calculate baseline accuracy once
    print('Calculating baseline accuracy...')
    baseline_accuracy = evaluate_model(original_model, test_loader, device)
    print(f'Baseline Accuracy: {baseline_accuracy:.2f}%\n')
    
    # Define sweep configurations
    # Only use powers of 2 and 8-bit (these work reliably)
    weight_bits_options = [8, 4, 2]
    activation_bits_options = [8, 4, 2]
    
    configurations = list(product(weight_bits_options, activation_bits_options))
    
    print(f'Running {len(configurations)} experiments...')
    print(f'Weight bits options: {weight_bits_options}')
    print(f'Activation bits options: {activation_bits_options}\n')
    
    # Run experiments
    all_results = []
    
    # Disable wandb auto-logging to avoid errors
    os.environ['WANDB_SILENT'] = 'true'
    
    for idx, (w_bits, a_bits) in enumerate(configurations, 1):
        print(f'\n{"="*60}')
        print(f'Experiment {idx}/{len(configurations)}: W{w_bits}_A{a_bits}')
        print(f'{"="*60}')
        
        try:
            # Initialize wandb with simpler settings
            run = wandb.init(
                project="mobilenet-compression-sweep",
                name=f"W{w_bits}_A{a_bits}",
                config={'weight_bits': w_bits, 'activation_bits': a_bits},
                reinit=True,
                settings=wandb.Settings(start_method="thread")
            )
            
            # Run experiment
            results = run_single_experiment(
                w_bits, a_bits, device, test_loader, original_state_dict, original_size_mb
            )
            
            # Add baseline accuracy
            results['original_accuracy'] = baseline_accuracy
            results['accuracy_drop'] = baseline_accuracy - results['compressed_accuracy']
            
            all_results.append(results)
            
            # Log to wandb
            wandb.log(results)
            
            print(f'Compressed Accuracy: {results["compressed_accuracy"]:.2f}%')
            print(f'Accuracy Drop: {results["accuracy_drop"]:.2f}%')
            print(f'Original Size: {results["original_size_mb"]:.2f} MB')
            print(f'Compressed Size: {results["compressed_size_mb"]:.2f} MB')
            print(f'Compression Ratio: {results["model_compression_ratio"]:.2f}x')
            
            wandb.finish()
            
        except Exception as e:
            print(f'Error in experiment W{w_bits}_A{a_bits}: {str(e)}')
            import traceback
            traceback.print_exc()
            try:
                wandb.finish()
            except:
                pass
            continue
    
    # Print summary
    print('\n' + '='*100)
    print('SUMMARY OF ALL EXPERIMENTS')
    print('='*100)
    print(f'{"W_bits":<8} {"A_bits":<8} {"Accuracy":<10} {"Drop":<8} {"Orig_MB":<10} {"Comp_MB":<10} {"Ratio":<8}')
    print('-' * 100)
    
    for r in all_results:
        print(f'{r["weight_bits"]:<8} {r["activation_bits"]:<8} '
              f'{r["compressed_accuracy"]:<10.2f} {r["accuracy_drop"]:<8.2f} '
              f'{r["original_size_mb"]:<10.2f} {r["compressed_size_mb"]:<10.2f} '
              f'{r["model_compression_ratio"]:<8.2f}')
    
    # Find best configurations
    if all_results:
        print(f'\n{"="*100}')
        print('BEST CONFIGURATIONS:')
        print(f'{"="*100}')
        
        best_accuracy = max(all_results, key=lambda x: x['compressed_accuracy'])
        print(f'\nBest Accuracy: W{best_accuracy["weight_bits"]}_A{best_accuracy["activation_bits"]} '
              f'= {best_accuracy["compressed_accuracy"]:.2f}%')
        
        best_compression = max(all_results, key=lambda x: x['model_compression_ratio'])
        print(f'Best Compression: W{best_compression["weight_bits"]}_A{best_compression["activation_bits"]} '
              f'= {best_compression["model_compression_ratio"]:.2f}x '
              f'(Accuracy: {best_compression["compressed_accuracy"]:.2f}%)')
        
        # Best trade-off (high compression with <2% accuracy drop)
        good_tradeoffs = [r for r in all_results if r['accuracy_drop'] < 2.0]
        if good_tradeoffs:
            best_tradeoff = max(good_tradeoffs, key=lambda x: x['model_compression_ratio'])
            print(f'Best Trade-off (<2% drop): W{best_tradeoff["weight_bits"]}_A{best_tradeoff["activation_bits"]} '
                  f'= {best_tradeoff["model_compression_ratio"]:.2f}x '
                  f'(Drop: {best_tradeoff["accuracy_drop"]:.2f}%)')


if __name__ == '__main__':
    main()