"""
SD-UNet: U-Net with Structured Dropout (DropBlock)
Uses DropBlock2D for better regularization than standard dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """PyTorch implementation of DropBlock2D for SD-UNet"""
    
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
        
        # Apply block mask using max pooling
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


class SDUNetEncoderBlock(nn.Module):
    """SD-UNet Encoder Block with configurable parameters"""
    
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9, 
                 activation='relu', use_batchnorm=False):
        super(SDUNetEncoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.dropblock1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.dropblock2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Configurable activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
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


class SDUNetDecoderBlock(nn.Module):
    """SD-UNet Decoder Block with configurable parameters"""
    
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9, 
                 activation='relu', use_batchnorm=False):
        super(SDUNetDecoderBlock, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                         stride=2, padding=1, output_padding=1)
        
        # After concatenation, input channels will be out_channels + out_channels
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, 
                               padding=1, bias=not use_batchnorm)
        self.dropblock1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=not use_batchnorm)
        self.dropblock2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Configurable activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
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


class ConfigurableSDUNet(nn.Module):
    """
    Configurable SD-UNet (StructuredDropout U-Net) for hyperparameter optimization
    Based on the original SD-UNet architecture but fully implemented in PyTorch
    """
    
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1, 
                 start_neurons=16,
                 block_size=7,
                 keep_prob=0.9,
                 activation='relu',
                 use_batchnorm=False,
                 dropout_rate=0.0):
        super(ConfigurableSDUNet, self).__init__()
        
        self.start_neurons = start_neurons
        
        # Encoder Path
        self.enc1 = SDUNetEncoderBlock(
            input_channels, start_neurons, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.enc2 = SDUNetEncoderBlock(
            start_neurons, start_neurons * 2, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.enc3 = SDUNetEncoderBlock(
            start_neurons * 2, start_neurons * 4, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        
        # Bottleneck (Middle)
        self.bottleneck_conv1 = nn.Conv2d(start_neurons * 4, start_neurons * 8, 
                                          kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bottleneck_dropblock1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bottleneck_bn1 = nn.BatchNorm2d(start_neurons * 8) if use_batchnorm else nn.Identity()
        
        self.bottleneck_conv2 = nn.Conv2d(start_neurons * 8, start_neurons * 8, 
                                          kernel_size=3, padding=1, bias=not use_batchnorm)
        self.bottleneck_dropblock2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bottleneck_bn2 = nn.BatchNorm2d(start_neurons * 8) if use_batchnorm else nn.Identity()
        
        # Configurable activation for bottleneck
        if activation == 'relu':
            self.bottleneck_activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.bottleneck_activation = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.bottleneck_activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.bottleneck_activation = nn.Tanh()
        else:
            self.bottleneck_activation = nn.ReLU(inplace=True)
        
        # Decoder Path
        self.dec3 = SDUNetDecoderBlock(
            start_neurons * 8, start_neurons * 4, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.dec2 = SDUNetDecoderBlock(
            start_neurons * 4, start_neurons * 2, 
            block_size=block_size, keep_prob=keep_prob, 
            activation=activation, use_batchnorm=use_batchnorm
        )
        self.dec1 = SDUNetDecoderBlock(
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
        
        # Bottleneck
        x = self.bottleneck_conv1(x3)
        x = self.bottleneck_dropblock1(x)
        x = self.bottleneck_bn1(x)
        x = self.bottleneck_activation(x)
        
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
        
        # Return logits for training (sigmoid will be applied in loss function)
        return x
