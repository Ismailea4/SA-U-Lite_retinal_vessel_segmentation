"""
Evaluation metrics for vessel segmentation
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU (Intersection over Union)"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def sensitivity(pred, target, smooth=1e-6):
    """Calculate Sensitivity (Recall/True Positive Rate)"""
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    false_negative = ((1 - pred) * target).sum()
    return (true_positive + smooth) / (true_positive + false_negative + smooth)


def specificity(pred, target, smooth=1e-6):
    """Calculate Specificity (True Negative Rate)"""
    pred = (pred > 0.5).float()
    true_negative = ((1 - pred) * (1 - target)).sum()
    false_positive = (pred * (1 - target)).sum()
    return (true_negative + smooth) / (true_negative + false_positive + smooth)


def accuracy(pred, target):
    """Calculate Accuracy"""
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total


def f1_score(pred, target, smooth=1e-6):
    """Calculate F1 Score"""
    pred = (pred > 0.5).float()
    true_positive = (pred * target).sum()
    false_positive = (pred * (1 - target)).sum()
    false_negative = ((1 - pred) * target).sum()
    
    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
    
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    return f1


def calculate_auc(pred_prob, target):
    """Calculate AUC (Area Under ROC Curve)"""
    pred_prob_flat = pred_prob.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    
    try:
        auc = roc_auc_score(target_flat, pred_prob_flat)
        return auc
    except:
        return 0.0


def compute_all_metrics(pred_prob, target):
    """Compute all metrics at once"""
    metrics = {
        'Dice': dice_coefficient(pred_prob, target).item(),
        'IoU': iou_score(pred_prob, target).item(),
        'SE': sensitivity(pred_prob, target).item(),
        'SP': specificity(pred_prob, target).item(),
        'ACC': accuracy(pred_prob, target).item(),
        'F1': f1_score(pred_prob, target).item(),
        'AUC': calculate_auc(pred_prob, target)
    }
    return metrics
