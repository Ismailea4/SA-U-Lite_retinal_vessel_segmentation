"""
Visualization script for model training results
Plots training history (loss and dice) for all models in a dataset/epoch combination

Usage:
    Interactive mode:
        python plot_result.py
    
    Command-line mode:
        python plot_result.py --dataset CHASE --epochs 10_epochs
        python plot_result.py -d DRIVE -e 50_epochs --no-save
        python plot_result.py --list-datasets
        python plot_result.py --list-epochs CHASE
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
from pathlib import Path


class ResultsPlotter:
    """Handles plotting of training results"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir).parent if Path(results_dir).name == 'plot_result.py' else Path(results_dir)
        self.available_datasets = self._get_available_datasets()
        self.available_epochs = {}
        
        for dataset in self.available_datasets:
            self.available_epochs[dataset] = self._get_available_epochs(dataset)
    
    def _get_available_datasets(self):
        """Get list of available datasets"""
        datasets = []
        if not self.results_dir.exists():
            return datasets
        
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name not in ['__pycache__', '.git']:
                datasets.append(item.name)
        
        return sorted(datasets)
    
    def _get_available_epochs(self, dataset):
        """Get list of available epoch folders for a dataset"""
        epochs = []
        dataset_path = self.results_dir / dataset
        
        if not dataset_path.exists():
            return epochs
        
        for item in dataset_path.iterdir():
            if item.is_dir() and 'epoch' in item.name:
                epochs.append(item.name)
        
        return sorted(epochs)
    
    def _get_model_histories(self, dataset, epoch_folder):
        """Get all model training histories from a dataset/epoch folder"""
        train_hist_dir = self.results_dir / dataset / epoch_folder / 'train_hist'
        
        if not train_hist_dir.exists():
            raise FileNotFoundError(f"Training history directory not found: {train_hist_dir}")
        
        # Find all CSV files
        csv_files = list(train_hist_dir.glob('*_train_hist.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"No training history CSV files found in {train_hist_dir}")
        
        histories = {}
        
        for csv_file in csv_files:
            model_name = csv_file.stem.replace('_train_hist', '')
            
            try:
                df = pd.read_csv(csv_file)
                histories[model_name] = df
                
                # Load summary JSON if available
                json_file = csv_file.with_name(f"{model_name}_history_summary.json")
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        summary = json.load(f)
                        histories[model_name].attrs['summary'] = summary
                
            except Exception as e:
                print(f"⚠ Warning: Could not load {csv_file.name}: {e}")
        
        return histories
    
    def plot_training_history(self, dataset, epoch_folder, save_fig=True):
        """Plot training history for all models"""
        
        # Validate inputs
        if dataset not in self.available_datasets:
            raise ValueError(f"Dataset '{dataset}' not found. Available: {self.available_datasets}")
        
        if epoch_folder not in self.available_epochs[dataset]:
            raise ValueError(f"Epoch folder '{epoch_folder}' not found for {dataset}. Available: {self.available_epochs[dataset]}")
        
        # Get histories
        print(f"\n{'='*80}")
        print(f"Loading training histories: {dataset} / {epoch_folder}")
        print(f"{'='*80}")
        
        histories = self._get_model_histories(dataset, epoch_folder)
        
        if not histories:
            print("⚠ No training histories found!")
            return
        
        print(f"✓ Found {len(histories)} model(s): {', '.join(histories.keys())}")
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training History - {dataset} Dataset ({epoch_folder})', 
                     fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
        
        # Plot each model
        for idx, (model_name, df) in enumerate(histories.items()):
            color = colors[idx]
            
            # Training Loss
            axes[0, 0].plot(df['epoch'], df['loss_train'], 
                           label=model_name, color=color, linewidth=2, marker='o', markersize=4)
            
            # Validation Loss
            axes[0, 1].plot(df['epoch'], df['loss_val'], 
                           label=model_name, color=color, linewidth=2, marker='s', markersize=4)
            
            # Training Dice
            axes[1, 0].plot(df['epoch'], df['dice_train'], 
                           label=model_name, color=color, linewidth=2, marker='^', markersize=4)
            
            # Validation Dice
            axes[1, 1].plot(df['epoch'], df['dice_val'], 
                           label=model_name, color=color, linewidth=2, marker='D', markersize=4)
        
        # Configure subplots
        axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Dice Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        axes[1, 1].set_title('Validation Dice Score', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Dice Score')
        axes[1, 1].legend(loc='lower right')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            output_dir = self.results_dir / dataset / epoch_folder
            output_path = output_dir / 'training_history.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Figure saved to: {output_path}")
        
        plt.show()
        
        # Print summary statistics
        self._print_summary_stats(histories)
    
    def _print_summary_stats(self, histories):
        """Print summary statistics for all models"""
        
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY STATISTICS")
        print(f"{'='*80}\n")
        
        summary_data = []
        
        for model_name, df in histories.items():
            stats = {
                'Model': model_name,
                'Epochs': len(df),
                'Final Train Loss': df['loss_train'].iloc[-1],
                'Final Val Loss': df['loss_val'].iloc[-1],
                'Final Train Dice': df['dice_train'].iloc[-1],
                'Final Val Dice': df['dice_val'].iloc[-1],
                'Best Val Dice': df['dice_val'].max(),
                'Best Epoch': df.loc[df['dice_val'].idxmax(), 'epoch']
            }
            summary_data.append(stats)
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print(f"\n{'='*80}")
    
    def plot_performance_comparison(self, dataset, epoch_folder, save_fig=True):
        """Plot performance metrics comparison from performance.csv"""
        
        performance_path = self.results_dir / dataset / epoch_folder / 'performance.csv'
        
        if not performance_path.exists():
            print(f"⚠ Performance file not found: {performance_path}")
            return
        
        df = pd.read_csv(performance_path)
        
        print(f"\n{'='*80}")
        print(f"Performance Comparison - {dataset} Dataset ({epoch_folder})")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        print(f"\n{'='*80}")
        
        # Plot metrics
        metrics = ['Dice', 'IoU', 'SE', 'SP', 'ACC', 'F1', 'AUC']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Performance Metrics - {dataset} Dataset ({epoch_folder})', 
                     fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[idx]
                bars = ax.bar(df['model_name'], df[metric], alpha=0.7, 
                             color=plt.cm.tab10(np.linspace(0, 1, len(df))))
                
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.set_ylabel('Score')
                ax.set_ylim([0, 1.05])
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
        
        # Parameters plot
        if 'parameters' in df.columns:
            ax = axes[7]
            bars = ax.bar(df['model_name'], df['parameters']/1000, alpha=0.7,
                         color=plt.cm.tab10(np.linspace(0, 1, len(df))))
            ax.set_title('Parameters (K)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count (K)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}K',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        if save_fig:
            output_dir = self.results_dir / dataset / epoch_folder
            output_path = output_dir / 'performance_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Figure saved to: {output_path}")
        
        plt.show()
    
    def interactive_plot(self):
        """Interactive mode to select dataset and epoch"""
        
        print("="*80)
        print("TRAINING RESULTS VISUALIZATION")
        print("="*80)
        
        if not self.available_datasets:
            print("\n✗ No datasets found in results directory!")
            print(f"   Expected location: {self.results_dir}")
            return
        
        # Select dataset
        print("\nAvailable datasets:")
        for idx, dataset in enumerate(self.available_datasets, 1):
            epochs = self.available_epochs[dataset]
            print(f"  {idx}. {dataset} ({len(epochs)} epoch folder(s))")
        
        try:
            dataset_choice = int(input("\nSelect dataset (number): ")) - 1
            dataset = self.available_datasets[dataset_choice]
        except (ValueError, IndexError):
            print("✗ Invalid selection!")
            return
        
        # Select epoch folder
        available_epochs = self.available_epochs[dataset]
        
        if not available_epochs:
            print(f"\n✗ No epoch folders found for {dataset}!")
            return
        
        print(f"\nAvailable epoch folders for {dataset}:")
        for idx, epoch in enumerate(available_epochs, 1):
            print(f"  {idx}. {epoch}")
        
        try:
            epoch_choice = int(input("\nSelect epoch folder (number): ")) - 1
            epoch_folder = available_epochs[epoch_choice]
        except (ValueError, IndexError):
            print("✗ Invalid selection!")
            return
        
        # Plot results
        try:
            self.plot_training_history(dataset, epoch_folder)
            self.plot_performance_comparison(dataset, epoch_folder)
        except Exception as e:
            print(f"\n✗ Error plotting results: {e}")
            import traceback
            traceback.print_exc()


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Visualize training results for U-Net model benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python plot_result.py
  
  # Plot specific dataset and epoch
  python plot_result.py --dataset CHASE --epochs 10_epochs
  python plot_result.py -d DRIVE -e 50_epochs
  
  # Plot without saving figures
  python plot_result.py -d CHASE -e 10_epochs --no-save
  
  # List available datasets
  python plot_result.py --list-datasets
  
  # List available epoch folders for a dataset
  python plot_result.py --list-epochs CHASE
        """
    )
    
    parser.add_argument('-d', '--dataset',
                       type=str,
                       help='Dataset name (CHASE, DRIVE, STARE, HRF)')
    
    parser.add_argument('-e', '--epochs',
                       type=str,
                       help='Epoch folder name (e.g., 10_epochs, 25_epochs)')
    
    parser.add_argument('--no-save',
                       action='store_true',
                       help='Do not save figures to disk (only display)')
    
    parser.add_argument('--list-datasets',
                       action='store_true',
                       help='List all available datasets and exit')
    
    parser.add_argument('--list-epochs',
                       type=str,
                       metavar='DATASET',
                       help='List available epoch folders for a dataset and exit')
    
    parser.add_argument('--results-dir',
                       type=str,
                       default='results',
                       help='Path to results directory (default: results)')
    
    return parser.parse_args()


def list_datasets(plotter):
    """List all available datasets"""
    
    print("\n" + "="*80)
    print("AVAILABLE DATASETS")
    print("="*80 + "\n")
    
    if not plotter.available_datasets:
        print("✗ No datasets found in results directory!")
        print(f"  Location: {plotter.results_dir}")
        return
    
    for dataset in plotter.available_datasets:
        epochs = plotter.available_epochs[dataset]
        print(f"  • {dataset:<15} ({len(epochs)} epoch folder(s))")
        if epochs:
            for epoch in epochs:
                print(f"    - {epoch}")
    
    print("\n" + "="*80)


def list_epochs_for_dataset(plotter, dataset):
    """List available epoch folders for a specific dataset"""
    
    if dataset not in plotter.available_datasets:
        print(f"\n✗ Dataset '{dataset}' not found!")
        print(f"  Available datasets: {', '.join(plotter.available_datasets)}")
        return
    
    epochs = plotter.available_epochs[dataset]
    
    print("\n" + "="*80)
    print(f"AVAILABLE EPOCH FOLDERS FOR {dataset}")
    print("="*80 + "\n")
    
    if not epochs:
        print(f"✗ No epoch folders found for {dataset}!")
        return
    
    for epoch in epochs:
        train_hist_dir = plotter.results_dir / dataset / epoch / 'train_hist'
        if train_hist_dir.exists():
            num_models = len(list(train_hist_dir.glob('*_train_hist.csv')))
            print(f"  • {epoch:<20} ({num_models} model(s))")
        else:
            print(f"  • {epoch:<20} (no training history)")
    
    print("\n" + "="*80)


def display_configuration(dataset, epoch_folder, save_fig):
    """Display the configuration before plotting"""
    
    print("\n" + "="*80)
    print("PLOT CONFIGURATION")
    print("="*80)
    print(f"  Dataset:      {dataset}")
    print(f"  Epoch Folder: {epoch_folder}")
    print(f"  Save Figures: {'Yes' if save_fig else 'No'}")
    print("="*80)


def main():
    """Main entry point"""
    
    args = parse_arguments()
    
    # Change to script directory if needed
    script_dir = Path(__file__).parent
    if script_dir.name == 'results':
        os.chdir(script_dir)
    
    # Initialize plotter
    plotter = ResultsPlotter(args.results_dir)
    
    # Handle list operations
    if args.list_datasets:
        list_datasets(plotter)
        return
    
    if args.list_epochs:
        list_epochs_for_dataset(plotter, args.list_epochs)
        return
    
    # Command-line mode
    if args.dataset and args.epochs:
        
        # Validate dataset
        if args.dataset not in plotter.available_datasets:
            print(f"\n✗ Error: Dataset '{args.dataset}' not found!")
            print(f"  Available datasets: {', '.join(plotter.available_datasets)}")
            print("\n  Use --list-datasets to see all available datasets")
            return
        
        # Validate epoch folder
        if args.epochs not in plotter.available_epochs[args.dataset]:
            print(f"\n✗ Error: Epoch folder '{args.epochs}' not found for {args.dataset}!")
            print(f"  Available epoch folders: {', '.join(plotter.available_epochs[args.dataset])}")
            print(f"\n  Use --list-epochs {args.dataset} to see all available epoch folders")
            return
        
        # Display configuration
        display_configuration(args.dataset, args.epochs, not args.no_save)
        
        # Plot results
        try:
            plotter.plot_training_history(args.dataset, args.epochs, save_fig=not args.no_save)
            plotter.plot_performance_comparison(args.dataset, args.epochs, save_fig=not args.no_save)
            print("\n✓ Plotting completed successfully!")
        except Exception as e:
            print(f"\n✗ Error plotting results: {e}")
            import traceback
            traceback.print_exc()
    
    # Interactive mode
    elif not args.dataset and not args.epochs:
        plotter.interactive_plot()
    
    # Incomplete arguments
    else:
        print("\n✗ Error: Both --dataset and --epochs are required for command-line mode!")
        print("  Use --help for usage information")
        print("  Or run without arguments for interactive mode")


if __name__ == '__main__':
    main()