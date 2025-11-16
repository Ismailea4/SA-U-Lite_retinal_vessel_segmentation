"""
W-Lite: W-Net architecture using U-Lite building blocks
Two U-Lite networks connected in series forming a 'W' shape for improved segmentation
Combines the efficiency of U-Lite with the refinement capability of W-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EncoderBlock(nn.Module):
    """U-Lite Encoding block with downsampling"""
    
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7), activation='gelu'):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel=mixer_kernel)
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
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip


class DecoderBlock(nn.Module):
    """U-Lite Decoding block with upsampling"""
    
    def __init__(self, in_c, out_c, mixer_kernel=(7, 7), activation='gelu'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel=mixer_kernel)
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
        x = self.act(self.pw2(x))
        return x


class BottleNeckBlock(nn.Module):
    """U-Lite Bottleneck with axial dilated DW convolution"""
    
    def __init__(self, dim, activation='gelu'):
        super().__init__()

        gc = dim // 4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel=(3, 3), dilation=1)
        self.dw2 = AxialDW(gc, mixer_kernel=(3, 3), dilation=2)
        self.dw3 = AxialDW(gc, mixer_kernel=(3, 3), dilation=3)

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
        x = self.bn(x)
        x = self.pw2(x)
        x = self.act(x)
        return x


class WLite(nn.Module):
    """
    W-Lite: Two U-Lite networks connected in series
    Forms a 'W' shape architecture for improved segmentation
    Combines U-Lite efficiency with W-Net refinement capability
    """
    
    def __init__(self, 
                 input_channel=3, 
                 num_classes=1, 
                 base_channels=16,
                 activation='gelu',
                 dropout_rate=0.0):
        super(WLite, self).__init__()
        
        # ============= FIRST U-LITE (Left V of W) =============
        # Initial convolution
        self.conv_in = nn.Conv2d(input_channel, base_channels, kernel_size=7, padding='same')
        
        # Encoder 1
        self.e1_1 = EncoderBlock(base_channels, base_channels * 2, activation=activation)
        self.e1_2 = EncoderBlock(base_channels * 2, base_channels * 4, activation=activation)
        self.e1_3 = EncoderBlock(base_channels * 4, base_channels * 8, activation=activation)
        self.e1_4 = EncoderBlock(base_channels * 8, base_channels * 16, activation=activation)
        
        # Bottleneck 1
        self.b1 = BottleNeckBlock(base_channels * 16, activation=activation)
        
        # Partial Decoder 1 (only go up to middle level)
        self.d1_4 = DecoderBlock(base_channels * 16, base_channels * 8, activation=activation)
        self.d1_3 = DecoderBlock(base_channels * 8, base_channels * 4, activation=activation)
        
        # ============= MIDDLE BOTTLENECK (Center of W) =============
        self.middle_bottleneck = BottleNeckBlock(base_channels * 4, activation=activation)
        
        # ============= SECOND U-LITE (Right V of W) =============
        # Encoder 2 (go back down from middle)
        self.e2_3 = EncoderBlock(base_channels * 4, base_channels * 8, activation=activation)
        self.e2_4 = EncoderBlock(base_channels * 8, base_channels * 16, activation=activation)
        
        # Bottleneck 2
        self.b2 = BottleNeckBlock(base_channels * 16, activation=activation)
        
        # Full Decoder 2 (go all the way up)
        self.d2_4 = DecoderBlock(base_channels * 16, base_channels * 8, activation=activation)
        self.d2_3 = DecoderBlock(base_channels * 8, base_channels * 4, activation=activation)
        self.d2_2 = DecoderBlock(base_channels * 4, base_channels * 2, activation=activation)
        self.d2_1 = DecoderBlock(base_channels * 2, base_channels, activation=activation)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Output
        self.conv_out = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # ============= FIRST U-LITE (Encoder-Decoder 1) =============
        # Initial
        x = self.conv_in(x)
        
        # Encoder path 1
        x, skip1_1 = self.e1_1(x)
        x, skip1_2 = self.e1_2(x)
        x, skip1_3 = self.e1_3(x)
        x, skip1_4 = self.e1_4(x)
        
        # Bottleneck 1
        x = self.b1(x)
        
        # Partial decoder 1 (go back up to middle level)
        x = self.d1_4(x, skip1_4)
        x = self.d1_3(x, skip1_3)
        
        # ============= MIDDLE BOTTLENECK =============
        x = self.middle_bottleneck(x)
        
        # ============= SECOND U-LITE (Encoder-Decoder 2) =============
        # Encoder path 2 (go back down)
        x, skip2_3 = self.e2_3(x)
        x, skip2_4 = self.e2_4(x)
        
        # Bottleneck 2
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


class ConfigurableWLite(nn.Module):
    """
    Configurable W-Lite for hyperparameter optimization
    Supports different activation functions, dropout rates, and base channels
    """
    
    def __init__(self, 
                 input_channel=3, 
                 num_classes=1, 
                 base_channels=16,
                 activation='gelu',
                 dropout_rate=0.0):
        super(ConfigurableWLite, self).__init__()
        
        # ============= FIRST U-LITE (Left V of W) =============
        self.conv_in = nn.Conv2d(input_channel, base_channels, kernel_size=7, padding='same')
        
        # Encoder 1
        self.e1_1 = EncoderBlock(base_channels, base_channels * 2, activation=activation)
        self.e1_2 = EncoderBlock(base_channels * 2, base_channels * 4, activation=activation)
        self.e1_3 = EncoderBlock(base_channels * 4, base_channels * 8, activation=activation)
        self.e1_4 = EncoderBlock(base_channels * 8, base_channels * 16, activation=activation)
        
        # Bottleneck 1
        self.b1 = BottleNeckBlock(base_channels * 16, activation=activation)
        
        # Partial Decoder 1
        self.d1_4 = DecoderBlock(base_channels * 16, base_channels * 8, activation=activation)
        self.d1_3 = DecoderBlock(base_channels * 8, base_channels * 4, activation=activation)
        
        # ============= MIDDLE BOTTLENECK =============
        self.middle_bottleneck = BottleNeckBlock(base_channels * 4, activation=activation)
        
        # ============= SECOND U-LITE (Right V of W) =============
        # Encoder 2
        self.e2_3 = EncoderBlock(base_channels * 4, base_channels * 8, activation=activation)
        self.e2_4 = EncoderBlock(base_channels * 8, base_channels * 16, activation=activation)
        
        # Bottleneck 2
        self.b2 = BottleNeckBlock(base_channels * 16, activation=activation)
        
        # Full Decoder 2
        self.d2_4 = DecoderBlock(base_channels * 16, base_channels * 8, activation=activation)
        self.d2_3 = DecoderBlock(base_channels * 8, base_channels * 4, activation=activation)
        self.d2_2 = DecoderBlock(base_channels * 4, base_channels * 2, activation=activation)
        self.d2_1 = DecoderBlock(base_channels * 2, base_channels, activation=activation)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Output
        self.conv_out = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # First U-Lite
        x = self.conv_in(x)
        x, skip1_1 = self.e1_1(x)
        x, skip1_2 = self.e1_2(x)
        x, skip1_3 = self.e1_3(x)
        x, skip1_4 = self.e1_4(x)
        
        x = self.b1(x)
        
        x = self.d1_4(x, skip1_4)
        x = self.d1_3(x, skip1_3)
        
        # Middle
        x = self.middle_bottleneck(x)
        
        # Second U-Lite
        x, skip2_3 = self.e2_3(x)
        x, skip2_4 = self.e2_4(x)
        
        x = self.b2(x)
        
        x = self.d2_4(x, skip2_4)
        x = self.d2_3(x, skip2_3)
        x = self.d2_2(x, skip1_2)
        x = self.d2_1(x, skip1_1)
        
        x = self.dropout(x)
        x = self.conv_out(x)
        
        return x