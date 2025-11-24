"""
Main benchmarking script
Run experiments with selected models, datasets, and epochs
"""

import os
import torch
import pandas as pd
import json

from config import (
    DATASETS, MODEL_REGISTRY, TRAINING_CONFIG, 
    RESULTS_DIR, AVAILABLE_EPOCHS
)
from data_loader import get_dataloaders
from trainer import ModelTrainer


class BenchmarkRunner:
    """Orchestrates benchmarking experiments"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def run_experiment(self, dataset_name, model_names, num_epochs):
        """
        Run benchmarking experiment
        
        Args:
            dataset_name: Name of dataset (CHASE, DRIVE, STARE, HRF)
            model_names: List of model names to train
            num_epochs: Number of training epochs
        """
        
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if num_epochs not in AVAILABLE_EPOCHS:
            print(f"⚠ Warning: {num_epochs} epochs not in standard list {AVAILABLE_EPOCHS}")
        
        # Create results directory structure
        save_dir = os.path.join(RESULTS_DIR, dataset_name, f"{num_epochs}_epochs")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK EXPERIMENT")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Models: {', '.join(model_names)}")
        print(f"Epochs: {num_epochs}")
        print(f"Save directory: {save_dir}")
        print(f"{'='*80}\n")
        
        # Load dataset
        dataset_config = DATASETS[dataset_name]
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_config, batch_size=TRAINING_CONFIG['batch_size']
        )
        
        print(f"✓ Dataset loaded:")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        
        # Results collection
        all_test_results = []
        
        # Train each model
        for model_name in model_names:
            if model_name not in MODEL_REGISTRY:
                print(f"⚠ Warning: Model '{model_name}' not found in registry, skipping...")
                continue
            
            print(f"\n{'='*80}")
            print(f"Model: {model_name}")
            print(f"{'='*80}")
            
            # Create model
            model_config = MODEL_REGISTRY[model_name]
            model = model_config['class'](**model_config['params']).to(self.device)
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            
            # Create trainer
            trainer = ModelTrainer(model, model_name, self.device, save_dir)
            
            # Train
            history = trainer.train(
                train_loader, val_loader, num_epochs,
                learning_rate=TRAINING_CONFIG['learning_rate']
            )
            
            # Test
            test_metrics = trainer.test(test_loader)
            test_metrics['parameters'] = total_params
            all_test_results.append(test_metrics)
            
            print(f"\n✓ {model_name} completed")
            print(f"  Best Val Dice: {trainer.best_dice:.4f}")
            print(f"  Test Dice: {test_metrics['Dice']:.4f}")
        
        # Save overall performance
        self.save_performance(save_dir, all_test_results)
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED")
        print(f"Results saved to: {save_dir}")
        print(f"{'='*80}\n")
    
    def save_performance(self, save_dir, test_results):
        """Save overall performance metrics (append mode)"""
        
        csv_path = os.path.join(save_dir, 'performance.csv')
        json_path = os.path.join(save_dir, 'performance.json')
        
        # ========== CSV HANDLING (APPEND MODE) ==========
        df_new = pd.DataFrame(test_results)
        
        if os.path.exists(csv_path):
            # Append to existing CSV
            df_existing = pd.read_csv(csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            print(f"\n✓ Appended to existing CSV: {csv_path}")
        else:
            # Create new CSV
            df_new.to_csv(csv_path, index=False)
            print(f"\n✓ Created new CSV: {csv_path}")
        
        # ========== JSON HANDLING (APPEND MODE) ==========
        if os.path.exists(json_path):
            # Load existing JSON
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
            
            # Append new results
            if isinstance(existing_data, list):
                combined_data = existing_data + test_results
            else:
                # Handle old format (single dict)
                combined_data = [existing_data] + test_results
            
            # Save combined data
            with open(json_path, 'w') as f:
                json.dump(combined_data, f, indent=2)
            
            print(f"✓ Appended to existing JSON: {json_path}")
        else:
            # Create new JSON
            with open(json_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            print(f"✓ Created new JSON: {json_path}")
        
        # ========== PRINT SUMMARY TABLE ==========
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY (ALL MODELS)")
        print(f"{'='*80}")
        
        # Load and display all results
        df_all = pd.read_csv(csv_path)
        print(df_all.to_string(index=False))
        print(f"{'='*80}")
        print(f"Total models trained: {len(df_all)}")
        print(f"{'='*80}")


def main():
    """Main entry point with interactive selection"""
    
    print("="*80)
    print("MODEL BENCHMARKING SYSTEM")
    print("="*80)
    
    # Select dataset
    print("\nAvailable datasets:")
    for idx, dataset in enumerate(DATASETS.keys(), 1):
        print(f"  {idx}. {dataset}")
    
    dataset_choice = input("\nSelect dataset (1-4): ")
    dataset_name = list(DATASETS.keys())[int(dataset_choice) - 1]
    
    # Select models
    print("\nAvailable models:")
    for idx, model in enumerate(MODEL_REGISTRY.keys(), 1):
        print(f"  {idx}. {model}")
    
    model_choice = input("\nSelect models (comma-separated, e.g., 1,3,5 or 'all'): ")
    
    if model_choice.lower() == 'all':
        model_names = list(MODEL_REGISTRY.keys())
    else:
        indices = [int(x.strip()) - 1 for x in model_choice.split(',')]
        model_names = [list(MODEL_REGISTRY.keys())[i] for i in indices]
    
    # Select epochs
    print(f"\nAvailable epochs: {AVAILABLE_EPOCHS}")
    epochs_choice = input(f"Select number of epochs (or custom number): ")
    num_epochs = int(epochs_choice)
    
    # Confirmation
    print(f"\n{'='*80}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*80}\n")
    
    confirm = input("Proceed with this configuration? (y/n): ")
    
    if confirm.lower() == 'y':
        runner = BenchmarkRunner()
        runner.run_experiment(dataset_name, model_names, num_epochs)
    else:
        print("Experiment cancelled.")


if __name__ == '__main__':
    main()
