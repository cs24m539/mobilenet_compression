import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (Core building block of MobileNet-v2)
    
    Steps:
    1. Expansion: 1x1 conv increases channels (expand_ratio times)
    2. Depthwise: 3x3 depthwise conv (one filter per channel)
    3. Projection: 1x1 conv reduces channels back
    4. Skip connection: Add input if stride=1 and same channels
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Expansion layer (pointwise)
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Projection layer (pointwise)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNet-v2 for CIFAR-10 (10 classes)
    
    Configuration:
    - width_multiplier: Controls number of channels (default: 1.0)
    - num_classes: Output classes (10 for CIFAR-10)
    - dropout: Dropout rate before final classifier
    """
    def __init__(self, num_classes=10, width_multiplier=1.0, dropout=0.2):
        super(MobileNetV2, self).__init__()
        
        # Building first layer
        input_channel = int(32 * width_multiplier)
        last_channel = int(1280 * width_multiplier)
        
        # Inverted residual settings
        # t: expansion factor, c: output channels, n: number of blocks, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Changed stride from 2 to 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # First convolution layer
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),  # stride=1 for CIFAR
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # Build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        # Last convolution layer
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)