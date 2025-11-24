"""
Command-line interface for running benchmarks
Usage: python run_benchmark.py --dataset CHASE --models SA-U-Lite,W-Lite --epochs 50
"""

import argparse
import sys
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