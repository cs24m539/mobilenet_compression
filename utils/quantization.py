import torch
import torch.nn as nn
import numpy as np

class QuantizationConfig:
    """Configuration for quantization parameters"""
    def __init__(self, weight_bits=8, activation_bits=8):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits


class QuantizedLinear(nn.Module):
    """Quantized Linear Layer"""
    def __init__(self, original_layer, config):
        super().__init__()
        self.config = config
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.weight_bits = config.weight_bits
        
        # Quantize weights
        weight = original_layer.weight.data
        scale, zero_point, quantized_weight = self.quantize_tensor(weight, config.weight_bits)
        
        # Store as uint8 (actual storage doesn't matter, we'll calculate theoretical size)
        self.register_buffer('quantized_weight', quantized_weight.to(torch.uint8))
        self.register_buffer('weight_scale', scale)
        self.register_buffer('weight_zero_point', zero_point)
        self.register_buffer('weight_shape', torch.tensor(weight.shape))
        
        # Handle bias
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None
    
    @staticmethod
    def quantize_tensor(tensor, bits):
        """Quantize tensor to specified bit-width"""
        qmin = 0
        qmax = 2 ** bits - 1
        
        min_val = tensor.min()
        max_val = tensor.max()
        
        scale = (max_val - min_val) / (qmax - qmin)
        scale = max(scale, 1e-8)
        
        zero_point = qmin - min_val / scale
        zero_point = torch.round(zero_point).clamp(qmin, qmax)
        
        quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
        
        return scale, zero_point, quantized
    
    @staticmethod
    def dequantize_tensor(quantized_tensor, scale, zero_point):
        """Convert quantized integers back to floats"""
        return (quantized_tensor.float() - zero_point) * scale
    
    def forward(self, x):
        # Dequantize weights
        weight = self.dequantize_tensor(
            self.quantized_weight.float(),
            self.weight_scale,
            self.weight_zero_point
        ).to(x.device).reshape(self.weight_shape.tolist())
        
        # Perform linear operation
        output = torch.nn.functional.linear(x, weight, self.bias)
        
        return output


class QuantizedConv2d(nn.Module):
    """Quantized Convolutional Layer"""
    def __init__(self, original_layer, config):
        super().__init__()
        self.config = config
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        self.weight_bits = config.weight_bits
        
        # Quantize weights
        weight = original_layer.weight.data
        scale, zero_point, quantized_weight = self.quantize_tensor(weight, config.weight_bits)
        
        # Store as uint8
        self.register_buffer('quantized_weight', quantized_weight.to(torch.uint8))
        self.register_buffer('weight_scale', scale)
        self.register_buffer('weight_zero_point', zero_point)
        self.register_buffer('weight_shape', torch.tensor(weight.shape))
        
        # Handle bias
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None
    
    @staticmethod
    def quantize_tensor(tensor, bits):
        """Quantize tensor to specified bit-width"""
        qmin = 0
        qmax = 2 ** bits - 1
        
        min_val = tensor.min()
        max_val = tensor.max()
        
        scale = (max_val - min_val) / (qmax - qmin)
        scale = max(scale, 1e-8)
        
        zero_point = qmin - min_val / scale
        zero_point = torch.round(zero_point).clamp(qmin, qmax)
        
        quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
        
        return scale, zero_point, quantized
    
    @staticmethod
    def dequantize_tensor(quantized_tensor, scale, zero_point):
        """Convert quantized integers back to floats"""
        return (quantized_tensor.float() - zero_point) * scale
    
    def forward(self, x):
        # Dequantize weights
        weight = self.dequantize_tensor(
            self.quantized_weight.float(),
            self.weight_scale,
            self.weight_zero_point
        ).to(x.device).reshape(self.weight_shape.tolist())
        
        # Perform convolution
        output = torch.nn.functional.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )
        
        return output


def quantize_model(model, config):
    """
    Replace Conv2d and Linear layers with quantized versions
    """
    original_device = next(model.parameters()).device
    
    # Track what we're quantizing
    quantized_layers = []
    
    for name, module in model.named_modules():
        # Skip first conv and final classifier
        if 'features.0' in name or 'classifier' in name:
            continue
        
        if isinstance(module, nn.Conv2d):
            quantized_layers.append((name, 'conv', module))
        elif isinstance(module, nn.Linear):
            quantized_layers.append((name, 'linear', module))
    
    # Replace modules
    for name, layer_type, original_module in quantized_layers:
        # Navigate to parent
        parts = name.split('.')
        if len(parts) > 1:
            parent = model.get_submodule('.'.join(parts[:-1]))
            child_name = parts[-1]
        else:
            parent = model
            child_name = name
        
        # Create quantized version
        if layer_type == 'conv':
            new_module = QuantizedConv2d(original_module, config)
        else:
            new_module = QuantizedLinear(original_module, config)
        
        # Replace
        setattr(parent, child_name, new_module)
    
    # Move to device
    model = model.to(original_device)
    
    return model


def get_model_size_mb(model, weight_bits=32):
    """
    Calculate actual model size considering quantization bit-width
    
    This calculates the THEORETICAL size based on bit-width, not PyTorch's storage
    """
    total_size_bytes = 0
    
    # Calculate FP32 parameter sizes (non-quantized parts)
    for name, param in model.named_parameters():
        # Only count parameters that are NOT quantized weights
        # (biases and parameters in first/last layers)
        num_elements = param.numel()
        element_size = param.element_size()
        size_bytes = num_elements * element_size
        total_size_bytes += size_bytes
    
    # Calculate quantized buffer sizes
    for name, buffer in model.named_buffers():
        if 'quantized_weight' in name:
            # This is a quantized weight buffer
            # Calculate THEORETICAL size based on bit-width
            num_elements = buffer.numel()
            theoretical_bytes = (num_elements * weight_bits) / 8.0
            total_size_bytes += theoretical_bytes
        elif 'weight_scale' in name or 'weight_zero_point' in name:
            # Metadata (scales and zero points) - always FP32
            size_bytes = buffer.numel() * buffer.element_size()
            total_size_bytes += size_bytes
        elif 'weight_shape' in name:
            # Shape info - int64
            size_bytes = buffer.numel() * buffer.element_size()
            total_size_bytes += size_bytes
        else:
            # Other buffers (BN parameters, etc.)
            size_bytes = buffer.numel() * buffer.element_size()
            total_size_bytes += size_bytes
    
    return total_size_bytes / (1024 ** 2)  # Convert to MB