"""
Dataset and DataLoader utilities
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class RetinalVesselDataset(Dataset):
    """Generic dataset class for retinal vessel segmentation"""
    
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of image files
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith(('.jpg', '.png', '.jpeg', '.ppm'))
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask - handle different naming conventions
        base_name = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]
        
        # Try different label naming patterns
        possible_mask_names = [
            base_name + '_label' + ext,
            base_name + '_label.png',
            base_name + '.png',
            img_name
        ]
        
        mask_path = None
        for mask_name in possible_mask_names:
            potential_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            raise FileNotFoundError(f"No mask found for {img_name}")
        
        mask = Image.open(mask_path).convert('L')
        
        # Resize
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)
        
        # Convert to numpy
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        # Convert to tensor format (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Binary mask
        mask = (mask > 0.5).float()
        
        return image, mask


def get_dataloaders(dataset_config, batch_size=4):
    """Create train, validation, and test dataloaders"""
    
    train_dataset = RetinalVesselDataset(
        image_dir=dataset_config['train_images'],
        mask_dir=dataset_config['train_labels'],
        target_size=dataset_config['image_size']
    )
    
    val_dataset = RetinalVesselDataset(
        image_dir=dataset_config['val_images'],
        mask_dir=dataset_config['val_labels'],
        target_size=dataset_config['image_size']
    )
    
    test_dataset = RetinalVesselDataset(
        image_dir=dataset_config['test_images'],
        mask_dir=dataset_config['test_labels'],
        target_size=dataset_config['image_size']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader
