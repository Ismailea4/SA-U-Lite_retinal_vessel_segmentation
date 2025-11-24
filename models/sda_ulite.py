"""
SDA-U-Lite: U-Lite with Spatial Attention and DropBlock
Combines the lightweight axial depthwise convolutions of U-Lite 
with both Spatial Attention mechanism and DropBlock regularization
This is the most advanced variant combining all beneficial features.
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


class AxialDW(nn.Module):
    """Axial Depthwise Convolution"""
    
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups=dim, dilation=dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups=dim, dilation=dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x


class EncoderBlockWithSDAtten(nn.Module):
    """U-Lite Encoder Block enhanced with both DropBlock and Spatial Attention"""
    
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7), block_size=7, keep_prob=0.9, 
                 attention_kernel_size=7, activation='gelu'):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=mixer_kernel)
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.spatial_attention = SpatialAttention(kernel_size=attention_kernel_size)
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2, 2))
        
        # Configurable activation
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.dropblock(x)
        x = self.spatial_attention(x)
        skip = self.bn(x)
        x = self.act(self.down(self.pw(skip)))
        return x, skip


class DecoderBlockWithSDAtten(nn.Module):
    """U-Lite Decoder Block enhanced with both DropBlock and Spatial Attention"""
    
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7), block_size=7, keep_prob=0.9,
                 attention_kernel_size=7, activation='gelu'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=mixer_kernel)
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.spatial_attention = SpatialAttention(kernel_size=attention_kernel_size)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)
        
        # Configurable activation
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.GELU()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.pw(x)
        x = self.bn(x)
        x = self.dw(x)
        x = self.dropblock(x)
        x = self.spatial_attention(x)
        x = self.act(self.pw2(x))
        return x


class BottleNeckBlockWithSDAtten(nn.Module):
    """U-Lite Bottleneck Block enhanced with both DropBlock and Spatial Attention"""
    
    def __init__(self, dim, block_size=7, keep_prob=0.9, attention_kernel_size=7, activation='gelu'):
        super().__init__()

        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
        
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.spatial_attention = SpatialAttention(kernel_size=attention_kernel_size)
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        
        # Configurable activation
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.dropblock(x)
        x = self.bn(x)
        x = self.pw2(x)
        x = self.spatial_attention(x)
        x = self.act(x)
        return x


class ConfigurableSDAULite(nn.Module):
    """
    Configurable SDA-U-Lite (Spatial DropBlock Attention U-Lite) for hyperparameter optimization
    Combines U-Lite's efficient architecture with both Spatial Attention and DropBlock regularization
    This is the most comprehensive variant with all advanced features.
    """
    
    def __init__(self, 
                 input_channel=3, 
                 num_classes=1, 
                 base_channels=16,
                 block_size=7,
                 keep_prob=0.9,
                 attention_kernel_size=7,
                 activation='gelu',
                 dropout_rate=0.0):
        super().__init__()
        
        # Encoder
        self.conv_in = nn.Conv2d(input_channel, base_channels, kernel_size=7, padding='same')
        
        self.e1 = EncoderBlockWithSDAtten(
            base_channels, base_channels * 2,
            block_size=block_size, keep_prob=keep_prob, 
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        self.e2 = EncoderBlockWithSDAtten(
            base_channels * 2, base_channels * 4,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        self.e3 = EncoderBlockWithSDAtten(
            base_channels * 4, base_channels * 8,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        self.e4 = EncoderBlockWithSDAtten(
            base_channels * 8, base_channels * 16,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )

        # Bottleneck with both DropBlock and Spatial Attention
        self.b5 = BottleNeckBlockWithSDAtten(
            base_channels * 16,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )

        # Decoder
        self.d4 = DecoderBlockWithSDAtten(
            base_channels * 16, base_channels * 8,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        self.d3 = DecoderBlockWithSDAtten(
            base_channels * 8, base_channels * 4,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        self.d2 = DecoderBlockWithSDAtten(
            base_channels * 4, base_channels * 2,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        self.d1 = DecoderBlockWithSDAtten(
            base_channels * 2, base_channels,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        
        # Optional dropout before output
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.conv_out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        # Bottleneck
        x = self.b5(x)

        # Decoder
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        
        # Dropout and output
        x = self.dropout(x)
        x = self.conv_out(x)
        return x


# Alias for backward compatibility
SDAULite = ConfigurableSDAULite