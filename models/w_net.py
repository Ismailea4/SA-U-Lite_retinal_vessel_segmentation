"""
Classical W-Net with 18 Layers
Standard W-Net architecture for image segmentation
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
        elif activation == 'tanh':
            self.activation = nn.Tanh()
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
    Classical W-Net with 18 layers for retinal blood vessel segmentation
    
    Architecture:
    - 5 encoder levels (2 conv layers each) = 10 layers
    - 1 bottleneck (2 conv layers) = 2 layers  
    - 4 decoder levels (2 conv layers each) = 8 layers
    - 1 output layer = 1 layer
    Total: 18 layers (plus skip connections)
    """
    
    def __init__(self, 
                 input_channels=3, 
                 num_classes=1, 
                 start_filters=64,
                 activation='relu',
                 use_batchnorm=True,
                 dropout_rate=0.0):
        super(WNet18, self).__init__()
        
        # Calculate filter sizes for each level
        filters = [start_filters * (2**i) for i in range(6)]  # [64, 128, 256, 512, 1024, 2048]
        
        # Encoder Path (Contracting Path)
        self.encoder1 = Conv2dBlock(input_channels, filters[0], activation, use_batchnorm, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = Conv2dBlock(filters[0], filters[1], activation, use_batchnorm, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = Conv2dBlock(filters[1], filters[2], activation, use_batchnorm, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = Conv2dBlock(filters[2], filters[3], activation, use_batchnorm, dropout_rate)
        self.pool4 = nn.MaxPool2d(2)
        
        self.encoder5 = Conv2dBlock(filters[3], filters[4], activation, use_batchnorm, dropout_rate)
        self.pool5 = nn.MaxPool2d(2)
        
        # Bottleneck (Bridge)
        self.bottleneck = Conv2dBlock(filters[4], filters[5], activation, use_batchnorm, dropout_rate)

        # Bottleneck (upper)
        self.bottleneck_upper = Conv2dBlock(filters[3], filters[2], activation, use_batchnorm, dropout_rate)
        
        # Decoder Path (Expanding Path)
        self.upconv5 = nn.ConvTranspose2d(filters[5], filters[4], kernel_size=2, stride=2)
        self.decoder5 = Conv2dBlock(filters[5], filters[4], activation, use_batchnorm, dropout_rate)
        
        self.upconv4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.decoder4 = Conv2dBlock(filters[4], filters[3], activation, use_batchnorm, dropout_rate)
        
        self.upconv3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.decoder3 = Conv2dBlock(filters[3], filters[2], activation, use_batchnorm, dropout_rate)
        
        self.upconv2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.decoder2 = Conv2dBlock(filters[2], filters[1], activation, use_batchnorm, dropout_rate)
        
        self.upconv1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.decoder1 = Conv2dBlock(filters[1], filters[0], activation, use_batchnorm, dropout_rate)
        
        # Output layer
        self.output_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        #First V part of the W-net

        # Encoder path with skip connections
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        
        enc5 = self.encoder5(x)
        x = self.pool5(enc5)
        
        # Bottleneck 1 (bridge)
        x = self.bottleneck(x)

        # Decoder path with skip connections between the bottleneck 1 and 2
        x = self.upconv5(x)
        x = torch.cat([x, enc5], dim=1)  # Skip connection
        x = self.decoder5(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)  # Skip connection
        x = self.decoder4(x)
        
        #Bottleneck 2 (upper)
        x = self.bottleneck_upper(x)

        #Second V part of the W-net
        
        # Encoder path with skip connections between the bottleneck 2 and 3
        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        
        enc5 = self.encoder5(x)
        x = self.pool5(enc5)

        # Bottleneck 3(bridge)
        x = self.bottleneck(x)

        # Decoder path with skip connections
        x = self.upconv5(x)
        x = torch.cat([x, enc5], dim=1)  # Skip connection
        x = self.decoder5(x)
        
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)  # Skip connection
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)  # Skip connection
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.decoder1(x)
        
        # Output
        x = self.output_conv(x)
        
        return x

# Configurable version
ConfigurableWNet18 = WNet18