# Retinal Vessel Segmentation Benchmark

A modular system for training and comparing U-Net/U-Lite variants on CHASE, DRIVE, STARE, and HRF datasets.

## 📁 Repository Structure

```
SA-U-Lite_retinal_vessel_segmentation/
├── benchmark.py                # Interactive benchmarking runner
├── run_benchmark.py            # CLI for batch benchmarking
├── plot_result.py              # CLI for visualizing results
├── config.py                   # Datasets, model registry, training config
├── data_loader.py              # Dataset loading utilities
├── losses.py                   # Loss functions
├── metrics.py                  # Evaluation metrics
├── trainer.py                  # Training loop and logging
├── models/                     # Model implementations
│   ├── saulite.py
│   ├── saunet.py
│   ├── sda_ulite.py
│   ├── sda_wlite.py
│   ├── sdulite.py
│   ├── sdunet.py
│   ├── ulite.py
│   ├── unet18.py
│   ├── w_lite.py
│   └── w_net.py
└── results/                    # Auto-generated experiment outputs
    ├── CHASE/DRIVE/STARE/HRF/
    │   └── <epochs>/          # e.g., 10_epochs, 25_epochs, 50_epochs, 100_epochs
    │       ├── models/        # best_<model>.pth
    │       ├── train_hist/    # <model>_train_hist.csv & summary json
    │       ├── performance.csv
    │       └── performance.json
```

## 🚀 Quick Start

### Interactive Training

Run the interactive benchmarking script:

```powershell
python .\benchmark.py
```

You will be prompted to choose dataset(s), model(s), and epochs.

### Command-Line Benchmark (CLI)

Use `run_benchmark.py` to run experiments directly from the terminal:

```powershell
# Basic run
python .\run_benchmark.py --dataset CHASE --models SA-U-Lite --epochs 50

# Multiple models
python .\run_benchmark.py -d DRIVE -m SA-U-Lite,W-Lite,SDA-U-Lite -e 25

# All models
python .\run_benchmark.py -d STARE -m all -e 10

# Customize training hyperparameters
python .\run_benchmark.py -d CHASE -m SA-U-Lite -e 50 -b 4 -lr 0.001

# Preview configuration without training
python .\run_benchmark.py -d HRF -m U-Lite -e 5 --dry-run

# List options
python .\run_benchmark.py --list-datasets
python .\run_benchmark.py --list-models
```

Notes:
- `--models` accepts comma-separated names or `all`.
- Valid datasets and models come from `config.py` (`DATASETS`, `MODEL_REGISTRY`).

### Results Visualization (CLI)

Use `plot_result.py` to visualize training curves and performance metrics:

```powershell
# Interactive mode (prompted selection)
python .\plot_result.py

# Plot a specific dataset/epoch
python .\plot_result.py --dataset CHASE --epochs 50_epochs
python .\plot_result.py -d DRIVE -e 100_epochs

# Do not save figures (only display)
python .\plot_result.py -d STARE -e 25_epochs --no-save

# Discover available datasets and epoch folders
python .\plot_result.py --list-datasets
python .\plot_result.py --list-epochs CHASE
```

Outputs:
- Saves `training_history.png` and `performance_comparison.png` into `results/<dataset>/<epochs>/` unless `--no-save` is used.
- Reads model histories from `train_hist` and metrics from `performance.csv`.

## 📊 Available Models

- **SA-U-Lite**: U-Lite with spatial attention
- **SDA-U-Lite**: Spatial attention + DropBlock
- **U-Lite** / **UNet18** / **W-Lite** / **W-Net**: Baseline and W-shaped variants
- Additional variants: `sdulite.py`, `sdunet.py`, `saunet.py`

## 📁 Results Layout

After training, results are organized as:

```
results/
└── <DATASET>/
    └── <EPOCHS>/              # e.g., 100_epochs
        ├── models/            # best_<Model>.pth
        ├── train_hist/        # CSV + JSON summaries
        ├── performance.csv    # Tabular comparison across models
        └── performance.json   # Same metrics in JSON
```

### Training History CSV Columns

- `epoch`, `loss_train`, `loss_val`, `dice_train`, `dice_val`

### Performance Metrics Columns

- `model_name`, `Dice`, `IoU`, `SE`, `SP`, `ACC`, `F1`, `AUC`, `parameters`

## ⚙️ Configuration

Edit `config.py` to:

- Manage datasets in `DATASETS` (paths, image sizes)
- Register models in `MODEL_REGISTRY`
- Tune `TRAINING_CONFIG` (batch size, learning rate, optimizer)

Add a new dataset example:

```python
DATASETS['YOUR_DATASET'] = {
    'train_images': 'path/to/train/images',
    'train_labels': 'path/to/train/labels',
    'val_images': 'path/to/val/images',
    'val_labels': 'path/to/val/labels',
    'test_images': 'path/to/test/images',
    'test_labels': 'path/to/test/labels',
    'image_size': (256, 256)
}
```

Register a new model example:

```python
from models.your_model import YourModel

MODEL_REGISTRY['Your-Model-Name'] = {
    'class': YourModel,
    'params': {
        'param1': value1,
        'param2': value2
    }
}
```

## 🔍 Programmatic Usage

```python
from benchmark import BenchmarkRunner

runner = BenchmarkRunner()
runner.run_experiment(
    dataset_name='CHASE',
    model_names=['SA-U-Lite', 'W-Lite'],
    num_epochs=50
)
```

## 📝 Notes

- Best checkpoints saved in `results/<dataset>/<epochs>/models/`
- CSV training logs and JSON summaries saved per model in `train_hist/`
- Performance computed on the test set with the best checkpoint

## 🐛 Troubleshooting

- Dataset not found: verify paths in `config.py`
- Import errors: ensure model files exist in `models/` and are registered
- CUDA OOM: reduce `batch_size` or choose lighter models (e.g., `U-Lite`)

## 📊 Metrics Explained

- Dice, IoU, Sensitivity (SE), Specificity (SP), Accuracy (ACC), F1, AUC

---

Happy Benchmarking! 🚀
