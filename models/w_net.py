"""
Classical W-Net with 18 Layers
W-Net architecture for image segmentation (two U-Nets connected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):
    """Basic convolutional block for W-Net"""
    
    def __init__(self, in_channels, out_channels, activation='relu', 
                 use_batchnorm=True, dropout_rate=0.0):
        super(Conv2dBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Configurable activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class WNet18(nn.Module):
    """
    W-Net: Two U-Nets connected in series
    Forms a 'W' shape architecture for improved segmentation
    """
    
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1, 
                 start_filters=32,  # Reduced to keep model size reasonable
                 activation='relu',
                 use_batchnorm=True,
                 dropout_rate=0.0):
        super(WNet18, self).__init__()
        
        # Filter sizes
        f = [start_filters * (2**i) for i in range(6)]
        
        # ============= FIRST U-NET (Left V of W) =============
        # Encoder 1
        self.enc1_1 = Conv2dBlock(input_channels, f[0], activation, use_batchnorm, dropout_rate)
        self.pool1_1 = nn.MaxPool2d(2)
        
        self.enc1_2 = Conv2dBlock(f[0], f[1], activation, use_batchnorm, dropout_rate)
        self.pool1_2 = nn.MaxPool2d(2)
        
        self.enc1_3 = Conv2dBlock(f[1], f[2], activation, use_batchnorm, dropout_rate)
        self.pool1_3 = nn.MaxPool2d(2)
        
        self.enc1_4 = Conv2dBlock(f[2], f[3], activation, use_batchnorm, dropout_rate)
        self.pool1_4 = nn.MaxPool2d(2)
        
        # Bottleneck 1
        self.bottleneck1 = Conv2dBlock(f[3], f[4], activation, use_batchnorm, dropout_rate)
        
        # Decoder 1 (partial - only up to middle)
        self.up1_4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec1_4 = Conv2dBlock(f[4], f[3], activation, use_batchnorm, dropout_rate)
        
        self.up1_3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec1_3 = Conv2dBlock(f[3], f[2], activation, use_batchnorm, dropout_rate)
        
        # ============= MIDDLE BOTTLENECK (Center of W) =============
        self.middle_bottleneck = Conv2dBlock(f[2], f[2], activation, use_batchnorm, dropout_rate)
        
        # ============= SECOND U-NET (Right V of W) =============
        # Encoder 2
        self.enc2_3 = Conv2dBlock(f[2], f[2], activation, use_batchnorm, dropout_rate)
        self.pool2_3 = nn.MaxPool2d(2)
        
        self.enc2_4 = Conv2dBlock(f[2], f[3], activation, use_batchnorm, dropout_rate)
        self.pool2_4 = nn.MaxPool2d(2)
        
        # Bottleneck 2
        self.bottleneck2 = Conv2dBlock(f[3], f[4], activation, use_batchnorm, dropout_rate)
        
        # Decoder 2 (full)
        self.up2_4 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec2_4 = Conv2dBlock(f[4], f[3], activation, use_batchnorm, dropout_rate)
        
        self.up2_3 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec2_3 = Conv2dBlock(f[3], f[2], activation, use_batchnorm, dropout_rate)
        
        self.up2_2 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec2_2 = Conv2dBlock(f[2], f[1], activation, use_batchnorm, dropout_rate)
        
        self.up2_1 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec2_1 = Conv2dBlock(f[1], f[0], activation, use_batchnorm, dropout_rate)
        
        # Output
        self.output_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        # ============= FIRST U-NET (Encoder-Decoder 1) =============
        # Encoder path
        e1_1 = self.enc1_1(x)
        x = self.pool1_1(e1_1)
        
        e1_2 = self.enc1_2(x)
        x = self.pool1_2(e1_2)
        
        e1_3 = self.enc1_3(x)
        x = self.pool1_3(e1_3)
        
        e1_4 = self.enc1_4(x)
        x = self.pool1_4(e1_4)
        
        # Bottleneck 1
        x = self.bottleneck1(x)
        
        # Partial decoder (go back up to middle level)
        x = self.up1_4(x)
        x = torch.cat([x, e1_4], dim=1)
        x = self.dec1_4(x)
        
        x = self.up1_3(x)
        x = torch.cat([x, e1_3], dim=1)
        x = self.dec1_3(x)
        
        # ============= MIDDLE BOTTLENECK =============
        x = self.middle_bottleneck(x)
        
        # ============= SECOND U-NET (Encoder-Decoder 2) =============
        # Encoder path (go back down)
        e2_3 = self.enc2_3(x)
        x = self.pool2_3(e2_3)
        
        e2_4 = self.enc2_4(x)
        x = self.pool2_4(e2_4)
        
        # Bottleneck 2
        x = self.bottleneck2(x)
        
        # Full decoder path
        x = self.up2_4(x)
        x = torch.cat([x, e2_4], dim=1)
        x = self.dec2_4(x)
        
        x = self.up2_3(x)
        x = torch.cat([x, e2_3], dim=1)
        x = self.dec2_3(x)
        
        x = self.up2_2(x)
        x = torch.cat([x, e1_2], dim=1)  # Skip from first U-Net
        x = self.dec2_2(x)
        
        x = self.up2_1(x)
        x = torch.cat([x, e1_1], dim=1)  # Skip from first U-Net
        x = self.dec2_1(x)
        
        # Output
        x = self.output_conv(x)
        
        return x


# Configurable version
ConfigurableWNet18 = WNet18