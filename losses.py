"""
Loss functions for training
"""

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss"""
    
    def __init__(self, dice_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # BCE Loss
        bce_loss = self.bce(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + 1) / (pred_sigmoid.sum() + target.sum() + 1)
        
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss
