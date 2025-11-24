"""
Training and evaluation utilities
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import json
import pandas as pd
import numpy as np

from metrics import dice_coefficient, compute_all_metrics
from losses import DiceBCELoss


class ModelTrainer:
    """Handles model training, validation, and testing"""
    
    def __init__(self, model, model_name, device, save_dir):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir
        
        # Create directories
        self.models_dir = os.path.join(save_dir, 'models')
        self.hist_dir = os.path.join(save_dir, 'train_hist')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.hist_dir, exist_ok=True)
        
        # History tracking
        self.history = {
            'epoch': [],
            'loss_train': [],
            'loss_val': [],
            'dice_train': [],
            'dice_val': []
        }
        
        self.best_dice = 0.0
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_dice = 0
        
        for images, masks in tqdm(train_loader, desc=f"Training {self.model_name}", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # Handle deep supervision outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(outputs)
                total_dice += dice_coefficient(pred_sigmoid, masks).item()
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        
        return avg_loss, avg_dice
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_dice = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validating {self.model_name}", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Handle deep supervision outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, masks)
                total_loss += loss.item()
                
                # Metrics
                pred_sigmoid = torch.sigmoid(outputs)
                total_dice += dice_coefficient(pred_sigmoid, masks).item()
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        return avg_loss, avg_dice
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.001):
        """Full training loop"""
        criterion = DiceBCELoss(dice_weight=0.5)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        print(f"\n{'='*80}")
        print(f"Training {self.model_name}")
        print(f"{'='*80}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Train
            train_loss, train_dice = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_dice = self.validate(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_dice)
            
            # Save history
            self.history['epoch'].append(epoch + 1)
            self.history['loss_train'].append(train_loss)
            self.history['loss_val'].append(val_loss)
            self.history['dice_train'].append(train_dice)
            self.history['dice_val'].append(val_dice)
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.save_checkpoint(epoch, optimizer, val_dice)
                print(f"  ✓ Best model saved! (Dice: {self.best_dice:.4f})")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch, optimizer, val_dice):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.models_dir, f'best_{self.model_name}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'model_name': self.model_name
        }, checkpoint_path)
    
    def save_history(self):
        """Save training history to CSV and JSON"""
        # Save as CSV
        csv_path = os.path.join(self.hist_dir, f'{self.model_name}_train_hist.csv')
        df = pd.DataFrame(self.history)
        df.to_csv(csv_path, index=False)
        
        # Save summary as JSON
        summary = {
            'model_name': self.model_name,
            'total_epochs': len(self.history['epoch']),
            'best_val_dice': self.best_dice,
            'final_train_loss': self.history['loss_train'][-1],
            'final_val_loss': self.history['loss_val'][-1],
            'final_train_dice': self.history['dice_train'][-1],
            'final_val_dice': self.history['dice_val'][-1]
        }
        
        json_path = os.path.join(self.hist_dir, f'{self.model_name}_history_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def test(self, test_loader):
        """Evaluate on test set"""
        # Load best model
        checkpoint_path = os.path.join(self.models_dir, f'best_{self.model_name}.pth')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        all_metrics = {
            'Dice': [], 'IoU': [], 'SE': [], 'SP': [], 
            'ACC': [], 'F1': [], 'AUC': []
        }
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc=f"Testing {self.model_name}", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                
                # Handle deep supervision outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                pred_sigmoid = torch.sigmoid(outputs)
                
                # Calculate all metrics
                batch_metrics = compute_all_metrics(pred_sigmoid, masks)
                
                for key in all_metrics:
                    all_metrics[key].append(batch_metrics[key])
        
        # Calculate mean metrics
        mean_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        mean_metrics['model_name'] = self.model_name
        
        return mean_metrics