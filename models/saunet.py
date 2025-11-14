"""
SA-UNet: U-Net with Spatial Attention Mechanism
Uses spatial attention to focus on important regions in feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """PyTorch implementation of DropBlock2D"""
    
    def __init__(self, block_size=7, keep_prob=0.9):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
    
    def forward(self, x):
        if not self.training:
            return x
        
        # Calculate gamma (drop rate)
        gamma = (1 - self.keep_prob) / (self.block_size ** 2)
        
        # Get dimensions
        batch_size, channels, height, width = x.shape
        
        # Calculate the number of activation units to drop
        gamma = gamma * (height * width) / ((height - self.block_size + 1) * (width - self.block_size + 1))
        
        # Sample from Bernoulli distribution
        mask = torch.bernoulli(torch.full((batch_size, channels, height, width), gamma, device=x.device))
        
        # Apply block mask
        mask = F.max_pool2d(
            mask, 
            kernel_size=self.block_size, 
            stride=1, 
            padding=self.block_size // 2
        )
        
        # Invert mask (1 means keep, 0 means drop)
        mask = 1 - mask
        
        # Normalize to maintain expected value
        normalize_scale = mask.numel() / (mask.sum() + 1e-7)
        
        return x * mask * normalize_scale


class SpatialAttention(nn.Module):
    """PyTorch implementation of Spatial Attention mechanism"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Average pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling along channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate
        concat = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(concat))
        
        # Apply attention
        return x * attention


class SAUNetEncoderBlock(nn.Module):
    """SA-UNet Encoder Block with configurable parameters"""
    
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9, 
                 activation='relu', use_batchnorm=True):
        super(SAUNetEncoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.dropblock1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.dropblock2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Configurable activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.dropblock1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.dropblock2(x)
        x = self.bn2(x)
        skip = self.activation(x)
        
        # Pooling for next level
        x = self.pool(skip)
        
        return x, skip


class SAUNetDecoderBlock(nn.Module):
    """SA-UNet Decoder Block with configurable parameters"""
    
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9, 
                 activation='relu', use_batchnorm=True):
        super(SAUNetDecoderBlock, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                         stride=2, padding=1, output_padding=1)
        
        # After concatenation, input channels will be out_channels + out_channels
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=True)
        self.dropblock1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.dropblock2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Configurable activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        # Upsample
        x = self.upconv(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # First conv block
        x = self.conv1(x)
        x = self.dropblock1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.dropblock2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class ConfigurableSAUNet(nn.Module):
    """
    Configurable SA-UNet (Spatial Attention U-Net) for hyperparameter optimization
    Incorporates spatial attention mechanism in the bottleneck to focus on important features
    """
    
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1, 
                 start_neurons=16,
                 block_size=7,
                 keep_prob=0.9,
                 activation='relu',
                 use_batchnorm=True,
                 dropout_rate=0.0,
                 attention_kernel_size=7):
        super(ConfigurableSAUNet, self).__init__()
        
        self.start_neurons = start_neurons
        
        # Encoder
        self.enc1 = SAUNetEncoderBlock(
            input_channels, start_neurons, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.enc2 = SAUNetEncoderBlock(
            start_neurons, start_neurons * 2, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.enc3 = SAUNetEncoderBlock(
            start_neurons * 2, start_neurons * 4, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        
        # Bottleneck with Spatial Attention
        self.bottleneck_conv1 = nn.Conv2d(start_neurons * 4, start_neurons * 8, 
                                          kernel_size=3, padding=1)
        self.bottleneck_dropblock1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bottleneck_bn1 = nn.BatchNorm2d(start_neurons * 8) if use_batchnorm else nn.Identity()
        
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention(kernel_size=attention_kernel_size)
        
        self.bottleneck_conv2 = nn.Conv2d(start_neurons * 8, start_neurons * 8, 
                                          kernel_size=3, padding=1)
        self.bottleneck_dropblock2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bottleneck_bn2 = nn.BatchNorm2d(start_neurons * 8) if use_batchnorm else nn.Identity()
        
        # Configurable activation for bottleneck
        if activation == 'relu':
            self.bottleneck_activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.bottleneck_activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.bottleneck_activation = nn.Tanh()
        else:
            self.bottleneck_activation = nn.ReLU(inplace=True)
        
        # Decoder
        self.dec3 = SAUNetDecoderBlock(
            start_neurons * 8, start_neurons * 4, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.dec2 = SAUNetDecoderBlock(
            start_neurons * 4, start_neurons * 2, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.dec1 = SAUNetDecoderBlock(
            start_neurons * 2, start_neurons, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        
        # Output layer
        self.output_conv = nn.Conv2d(start_neurons, num_classes, kernel_size=1)
        
        # Optional dropout before output
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        
        # Bottleneck with Spatial Attention
        x = self.bottleneck_conv1(x3)
        x = self.bottleneck_dropblock1(x)
        x = self.bottleneck_bn1(x)
        x = self.bottleneck_activation(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        x = self.bottleneck_conv2(x)
        x = self.bottleneck_dropblock2(x)
        x = self.bottleneck_bn2(x)
        x = self.bottleneck_activation(x)
        
        # Decoder path
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        x = self.dropout(x)
        x = self.output_conv(x)
        
        return x
