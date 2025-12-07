# MobileNet-v2 Compression on CIFAR-10

This repository implements MobileNet-v2 training and compression using quantization techniques for the CIFAR-10 dataset.

## Environment Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy matplotlib wandb pandas tqdm
```

## Repository Structure
```
mobilenet_compression/
├── models/
│   └── mobilenet_v2.py          # MobileNet-v2 architecture
├── utils/
│   ├── dataset.py               # CIFAR-10 data loading
│   └── quantization.py          # Quantization implementation
├── train.py                      # Training script
├── test.py                       # Single compression test
├── sweep.py                      # Multiple compression experiments
├── results/                      # Results and figures
└── README.md
```

## Usage

### Step 1: Train Baseline Model

Train MobileNet-v2 on CIFAR-10 without compression:
```bash
# Login to wandb (first time only)
wandb login

# Train with default settings
python train.py --epochs 200 --batch_size 128 --lr 0.1 --seed 42

# Train with custom settings
python train.py --epochs 150 --batch_size 256 --lr 0.05 --width_multiplier 0.75
```

**Training Configuration:**
- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.1 with Cosine Annealing
- Weight Decay: 4e-5
- Batch Size: 128
- Epochs: 200
- Data Augmentation: RandomCrop, RandomHorizontalFlip
- Normalization: CIFAR-10 mean/std

### Step 2: Test Single Compression Configuration

Test a specific quantization configuration:
```bash
# 8-bit weights and activations
python test.py --weight_bits 8 --activation_bits 8 --use_wandb

# 4-bit weights, 8-bit activations
python test.py --weight_bits 4 --activation_bits 8 --use_wandb

# 2-bit weights, 4-bit activations (extreme compression)
python test.py --weight_bits 2 --activation_bits 4 --use_wandb
```

### Step 3: Run Compression Sweep

Run multiple experiments to generate parallel coordinates plot:
```bash
python sweep.py
```

This will test various combinations of weight and activation bit-widths and automatically log results to WandB.

## Reproducibility

**Random Seed:** All experiments use seed 42 by default
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
```

**Dependency Versions:**
```
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
matplotlib==3.7.1
wandb==0.15.3
```

## Results

### Baseline Model (No Compression)
- Test Accuracy: 93.71%
- Model Size: ~9 MB
- Parameters: ~2.3M

### Compression Results

| W_bits | A_bits | Accuracy | Drop  | Orig_MB | Comp_MB | Ratio |
|-------|--------|----------|-------|---------|---------|-------|
|   8   |   8    |  93.78   | 0.07  |  8.66   |  2.40   | 3.61  |
|   8   |   4    |  93.78   | 0.07  |  8.66   |  2.40   | 3.61  |
|   8   |   2    |  93.78   | 0.07  |  8.66   |  2.40   | 3.61  |
|   4   |   8    |  92.08   | 1.77  |  8.66   |  1.36   | 6.38  |
|   4   |   4    |  92.08   | 1.77  |  8.66   |  1.36   | 6.38  |
|   4   |   2    |  92.08   | 1.77  |  8.66   |  1.36   | 6.38  |
|   2   |   8    |  10.00   | 83.85 |  8.66   |  0.84   | 10.35 |
|   2   |   4    |  10.00   | 83.85 |  8.66   |  0.84   | 10.35 |
|   2   |   2    |  10.00   | 83.85 |  8.66   |  0.84   | 10.35 |



## Compression Method Explanation

### Quantization Approach

We implement **symmetric uniform quantization**:

1. **Weight Quantization:**
   - Map FP32 weights to lower bit-width integers
   - Formula: `Q = round((W - min) / scale)`
   - Store: quantized values + scale + zero_point

2. **Activation Quantization:**
   - Quantize intermediate activations during inference
   - Reduces memory bandwidth requirements

3. **Layer-wise Quantization:**
   - Exceptions: First layer, BatchNorm, Final classifier kept in FP32
   - Reason: These layers have minimal size but significant accuracy impact

### Storage Overhead

For each quantized layer:
- **Metadata:** scale (4 bytes), zero_point (4 bytes)
- **Quantized weights:** N × (bits/8) bytes
- **Overhead:** ~8 bytes per layer (negligible)

Example for 1000 weights with 8-bit quantization:
- Original: 1000 × 4 = 4000 bytes
- Quantized: 1000 × 1 + 8 = 1008 bytes
- Compression: 3.97x

## WandB Integration

All experiments are logged to Weights & Biases:

1. **Training Metrics:** Loss, accuracy, learning rate
2. **Compression Metrics:** Size, compression ratio, accuracy drop
3. **Parallel Coordinates Plot:** Visualize trade-offs

Access your results at: https://wandb.ai
