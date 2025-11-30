"""
Command-line interface for running benchmarks
Usage: python run_benchmark.py --dataset CHASE --models SA-U-Lite,W-Lite --epochs 50
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from benchmark import BenchmarkRunner
from config import DATASETS, MODEL_REGISTRY, AVAILABLE_EPOCHS


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Run model benchmarking experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model on CHASE for 50 epochs
  python run_benchmark.py --dataset CHASE --models SA-U-Lite --epochs 50
  
  # Train multiple models on DRIVE for 25 epochs
  python run_benchmark.py --dataset DRIVE --models SA-U-Lite,W-Lite,SDA-U-Lite --epochs 25
  
  # Train all models on STARE for 10 epochs
  python run_benchmark.py --dataset STARE --models all --epochs 10
  
  # Quick test with single model
  python run_benchmark.py -d HRF -m U-Lite -e 5
        """
    )
    
    # Dataset argument
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help='Dataset to use for training'
    )
    
    # Models argument
    parser.add_argument(
        '-m', '--models',
        type=str,
        required=True,
        help=f"Model(s) to train (comma-separated or 'all'). Available: {', '.join(MODEL_REGISTRY.keys())}"
    )
    
    # Epochs argument
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        required=True,
        help=f"Number of training epochs. Recommended: {AVAILABLE_EPOCHS}"
    )
    
    # Batch size (optional)
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=4,
        help='Batch size for training (default: 4)'
    )
    
    # Learning rate (optional)
    parser.add_argument(
        '-lr', '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    # List available options
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    # Dry run
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without running training'
    )
    
    # Verbose mode
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Optimal hyperparameters mode
    parser.add_argument(
        '--optimal',
        action='store_true',
        help='Use optimal hyperparameters from HPO results (results/hpo_results/abc_hpo_results.csv)'
    )
    
    parser.add_argument(
        '--hpo-results',
        type=str,
        default='results/hpo_results/abc_hpo_results.csv',
        help='Path to HPO results CSV (default: results/hpo_results/abc_hpo_results.csv)'
    )
    
    return parser.parse_args()


def list_datasets():
    """List available datasets"""
    print("\n" + "="*80)
    print("AVAILABLE DATASETS")
    print("="*80)
    
    for dataset_name, config in DATASETS.items():
        print(f"\n{dataset_name}:")
        print(f"  Image size: {config['image_size']}")
        print(f"  Train: {config['train_images']}")
        print(f"  Val: {config['val_images']}")
        print(f"  Test: {config['test_images']}")
    
    print("\n" + "="*80)


def list_models():
    """List available models"""
    print("\n" + "="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    
    for idx, (model_name, config) in enumerate(MODEL_REGISTRY.items(), 1):
        print(f"\n{idx}. {model_name}")
        print(f"   Class: {config['class'].__name__}")
        print(f"   Parameters:")
        for key, value in config['params'].items():
            print(f"     - {key}: {value}")
    
    print("\n" + "="*80)
    print(f"Total: {len(MODEL_REGISTRY)} models")
    print("="*80)


def load_optimal_params(hpo_csv_path, dataset_name, model_names):
    """Load and apply optimal hyperparameters from HPO results."""
    if not Path(hpo_csv_path).exists():
        print(f"\n⚠ Warning: HPO results file not found: {hpo_csv_path}")
        print("  Proceeding with default parameters from config.py")
        return False
    
    try:
        hpo_df = pd.read_csv(hpo_csv_path)
        print(f"\n✓ Loaded HPO results from {hpo_csv_path}")
        
        # Filter for current dataset and models
        filtered = hpo_df[(hpo_df['dataset'] == dataset_name) & (hpo_df['model'].isin(model_names))]
        
        if len(filtered) == 0:
            print(f"⚠ No optimal parameters found for {dataset_name} with models {model_names}")
            print("  Proceeding with default parameters from config.py")
            return False
        
        print(f"\n📊 Applying optimal hyperparameters:\n")
        
        # Apply parameters for each model
        for _, row in filtered.iterrows():
            model_name = row['model']
            if model_name not in MODEL_REGISTRY:
                continue
            
            model_entry = MODEL_REGISTRY[model_name]
            
            print(f"  {model_name}:")
            print(f"    HPO Score: {row['best_score']:.4f}")
            
            # Map and update parameters
            param_mapping = {
                'base_channels': 'base_channels',
                'attention_kernel_size': 'attention_kernel_size',
                'block_size': 'block_size',
                'keep_prob': 'keep_prob',
                'dropout_rate': 'dropout_rate',
                'activation': 'activation',
            }
            
            for hpo_key, model_key in param_mapping.items():
                if hpo_key in row and model_key in model_entry['params']:
                    old_val = model_entry['params'][model_key]
                    new_val = row[hpo_key]
                    model_entry['params'][model_key] = new_val
                    print(f"    {model_key}: {old_val} → {new_val}")
        
        print()
        return True
        
    except Exception as e:
        print(f"\n⚠ Error loading HPO results: {e}")
        print("  Proceeding with default parameters from config.py")
        return False

# NEW: create alias entries with '_opti' suffix so saved results carry the suffix
def alias_models_with_suffix(model_names, suffix='_opti'):
    """Create MODEL_REGISTRY aliases with a suffix and return the suffixed names."""
    aliased_names = []
    for name in model_names:
        if name not in MODEL_REGISTRY:
            continue
        alias = f"{name}{suffix}"
        # If alias already exists, overwrite params to keep it in sync
        MODEL_REGISTRY[alias] = {
            'class': MODEL_REGISTRY[name]['class'],
            'params': dict(MODEL_REGISTRY[name]['params'])
        }
        aliased_names.append(alias)
    if aliased_names:
        print("\nRenaming models for this run:")
        for orig, ali in zip(model_names, aliased_names):
            print(f"  {orig} -> {ali}")
    return aliased_names


def parse_models(models_str):
    """Parse model names from string"""
    
    if models_str.lower() == 'all':
        return list(MODEL_REGISTRY.keys())
    
    # Split by comma and clean
    model_names = [m.strip() for m in models_str.split(',')]
    
    # Validate model names
    invalid_models = [m for m in model_names if m not in MODEL_REGISTRY]
    
    if invalid_models:
        print(f"\n✗ Error: Invalid model name(s): {', '.join(invalid_models)}")
        print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        sys.exit(1)
    
    return model_names


def display_configuration(dataset, models, epochs, batch_size, learning_rate):
    """Display experiment configuration"""
    
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Dataset:       {dataset}")
    print(f"Models:        {', '.join(models)} ({len(models)} model(s))")
    print(f"Epochs:        {epochs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("="*80 + "\n")


def main():
    """Main entry point"""
    
    args = parse_arguments()
    
    # Handle list commands
    if args.list_datasets:
        list_datasets()
        sys.exit(0)
    
    if args.list_models:
        list_models()
        sys.exit(0)
    
    # Parse model names
    model_names = parse_models(args.models)

    # Load optimal parameters if requested
    opt_loaded = False
    if args.optimal:
        print("\n" + "="*80)
        print("OPTIMAL HYPERPARAMETER MODE")
        print("="*80)
        opt_loaded = load_optimal_params(args.hpo_results, args.dataset, model_names)
        # Append '_opti' to names only if optimal params were successfully loaded
        if opt_loaded:
            model_names = alias_models_with_suffix(model_names, suffix='_opti')
        else:
            print("\nNo optimal parameters applied; proceeding without renaming.")

    # Display configuration
    display_configuration(
        args.dataset,
        model_names,
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # Dry run mode
    if args.dry_run:
        print("✓ Dry run mode - configuration validated successfully")
        print("Remove --dry-run flag to start training")
        sys.exit(0)
    
    # Confirmation
    try:
        confirm = input("Proceed with this configuration? (y/n): ")
        if confirm.lower() != 'y':
            print("Experiment cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nExperiment cancelled.")
        sys.exit(0)
    
    # Run benchmark
    try:
        runner = BenchmarkRunner()
        
        # Update training config if specified
        if args.batch_size != 4 or args.learning_rate != 0.001:
            from config import TRAINING_CONFIG
            TRAINING_CONFIG['batch_size'] = args.batch_size
            TRAINING_CONFIG['learning_rate'] = args.learning_rate
        
        runner.run_experiment(
            dataset_name=args.dataset,
            model_names=model_names,
            num_epochs=args.epochs
        )
        
        print("\n" + "="*80)
        print("✓ BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()