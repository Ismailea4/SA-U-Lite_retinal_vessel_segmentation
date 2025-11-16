"""
SDA-W-Lite: W-Lite with Spatial Attention and DropBlock
Combines the W-Lite architecture (two U-Lite networks in series) 
with Spatial Attention on bottlenecks and DropBlock regularization
This is the most advanced W-Lite variant with all beneficial features.
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


class EncoderBlockWithDropBlock(nn.Module):
    """U-Lite Encoder Block enhanced with DropBlock"""
    
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7), block_size=7, keep_prob=0.9, activation='gelu'):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=mixer_kernel)
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
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
        skip = self.bn(x)
        x = self.act(self.down(self.pw(skip)))
        return x, skip


class DecoderBlockWithDropBlock(nn.Module):
    """U-Lite Decoder Block enhanced with DropBlock"""
    
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7), block_size=7, keep_prob=0.9, activation='gelu'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=mixer_kernel)
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
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
        x = self.act(self.pw2(x))
        return x


class BottleNeckBlockWithSA(nn.Module):
    """U-Lite Bottleneck with Spatial Attention and DropBlock"""
    
    def __init__(self, dim, block_size=7, keep_prob=0.9, attention_kernel_size=7, activation='gelu'):
        super().__init__()

        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)
        
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1)
        self.spatial_attention = SpatialAttention(kernel_size=attention_kernel_size)
        
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


class ConfigurableSDAWLite(nn.Module):
    """
    SDA-W-Lite: W-Lite with Spatial Attention and DropBlock
    Combines W-Lite architecture (two U-Lite networks in W shape) with:
    - Spatial Attention on all 3 bottlenecks (bottleneck1, middle_bottleneck, bottleneck2)
    - DropBlock regularization on all encoder and decoder blocks
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
        super(ConfigurableSDAWLite, self).__init__()
        
        # ============= FIRST U-LITE (Left V of W) =============
        self.conv_in = nn.Conv2d(input_channel, base_channels, kernel_size=7, padding='same')
        
        # Encoder 1 with DropBlock
        self.e1_1 = EncoderBlockWithDropBlock(
            base_channels, base_channels * 2, 
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.e1_2 = EncoderBlockWithDropBlock(
            base_channels * 2, base_channels * 4,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.e1_3 = EncoderBlockWithDropBlock(
            base_channels * 4, base_channels * 8,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.e1_4 = EncoderBlockWithDropBlock(
            base_channels * 8, base_channels * 16,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        
        # Bottleneck 1 with Spatial Attention + DropBlock
        self.b1 = BottleNeckBlockWithSA(
            base_channels * 16,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        
        # Partial Decoder 1 with DropBlock
        self.d1_4 = DecoderBlockWithDropBlock(
            base_channels * 16, base_channels * 8,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.d1_3 = DecoderBlockWithDropBlock(
            base_channels * 8, base_channels * 4,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        
        # ============= MIDDLE BOTTLENECK (Center of W) =============
        # Middle Bottleneck with Spatial Attention + DropBlock
        self.middle_bottleneck = BottleNeckBlockWithSA(
            base_channels * 4,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        
        # ============= SECOND U-LITE (Right V of W) =============
        # Encoder 2 with DropBlock
        self.e2_3 = EncoderBlockWithDropBlock(
            base_channels * 4, base_channels * 8,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.e2_4 = EncoderBlockWithDropBlock(
            base_channels * 8, base_channels * 16,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        
        # Bottleneck 2 with Spatial Attention + DropBlock
        self.b2 = BottleNeckBlockWithSA(
            base_channels * 16,
            block_size=block_size, keep_prob=keep_prob,
            attention_kernel_size=attention_kernel_size, activation=activation
        )
        
        # Full Decoder 2 with DropBlock
        self.d2_4 = DecoderBlockWithDropBlock(
            base_channels * 16, base_channels * 8,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.d2_3 = DecoderBlockWithDropBlock(
            base_channels * 8, base_channels * 4,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.d2_2 = DecoderBlockWithDropBlock(
            base_channels * 4, base_channels * 2,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        self.d2_1 = DecoderBlockWithDropBlock(
            base_channels * 2, base_channels,
            block_size=block_size, keep_prob=keep_prob, activation=activation
        )
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Output
        self.conv_out = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # ============= FIRST U-LITE (Encoder-Decoder 1) =============
        x = self.conv_in(x)
        
        # Encoder path 1
        x, skip1_1 = self.e1_1(x)
        x, skip1_2 = self.e1_2(x)
        x, skip1_3 = self.e1_3(x)
        x, skip1_4 = self.e1_4(x)
        
        # Bottleneck 1 with Spatial Attention
        x = self.b1(x)
        
        # Partial decoder 1
        x = self.d1_4(x, skip1_4)
        x = self.d1_3(x, skip1_3)
        
        # ============= MIDDLE BOTTLENECK with Spatial Attention =============
        x = self.middle_bottleneck(x)
        
        # ============= SECOND U-LITE (Encoder-Decoder 2) =============
        # Encoder path 2
        x, skip2_3 = self.e2_3(x)
        x, skip2_4 = self.e2_4(x)
        
        # Bottleneck 2 with Spatial Attention
        x = self.b2(x)
        
        # Full decoder path 2
        x = self.d2_4(x, skip2_4)
        x = self.d2_3(x, skip2_3)
        x = self.d2_2(x, skip1_2)  # Skip from first U-Lite
        x = self.d2_1(x, skip1_1)  # Skip from first U-Lite
        
        # Dropout and output
        x = self.dropout(x)
        x = self.conv_out(x)
        
        return x


# Alias for backward compatibility
SDAWLite = ConfigurableSDAWLite