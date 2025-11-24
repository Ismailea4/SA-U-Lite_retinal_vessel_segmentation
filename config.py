"""
Configuration file for model benchmarking
Centralized configuration for datasets, models, and training parameters
"""

import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dataset paths - using dataset/ subdirectory
DATASETS = {
    'CHASE': {
        'train_images': 'dataset/CHASE/train/images',
        'train_labels': 'dataset/CHASE/train/labels',
        'val_images': 'dataset/CHASE/validate/images',
        'val_labels': 'dataset/CHASE/validate/labels',
        'test_images': 'dataset/CHASE/test/images',
        'test_labels': 'dataset/CHASE/test/labels',
        'image_size': (256, 256)
    },
    'DRIVE': {
        'train_images': 'dataset/DRIVE/train/images',
        'train_labels': 'dataset/DRIVE/train/labels',
        'val_images': 'dataset/DRIVE/validate/images',
        'val_labels': 'dataset/DRIVE/validate/labels',
        'test_images': 'dataset/DRIVE/test/images',
        'test_labels': 'dataset/DRIVE/test/labels',
        'image_size': (256, 256)
    },
    'STARE': {
        'train_images': 'dataset/STARE/train/images',
        'train_labels': 'dataset/STARE/train/labels',
        'val_images': 'dataset/STARE/validate/images',
        'val_labels': 'dataset/STARE/validate/labels',
        'test_images': 'dataset/STARE/test/images',
        'test_labels': 'dataset/STARE/test/labels',
        'image_size': (256, 256)
    },
    'HRF': {
        'train_images': 'dataset/HRF/train/images',
        'train_labels': 'dataset/HRF/train/labels',
        'val_images': 'dataset/HRF/validate/images',
        'val_labels': 'dataset/HRF/validate/labels',
        'test_images': 'dataset/HRF/test/images',
        'test_labels': 'dataset/HRF/test/labels',
        'image_size': (256, 256)
    }
}

# ==================== MODEL CONFIGURATIONS ====================
from models.unet18 import ConfigurableUNet18
from models.ulite import ConfigurableULite
from models.sdulite import ConfigurableULiteDropBlock
from models.sda_ulite import ConfigurableSDAULite
from models.w_net import ConfigurableWNet18
from models.w_lite import ConfigurableWLite
from models.w_lite2 import ConfigurableWLite2
from models.sda_wlite import ConfigurableSDAWLite
from models.enhance_saunet import DetailFocusedSAUNet

MODEL_REGISTRY = {
    'UNet18': {
        'class': ConfigurableUNet18,
        'params': {
            'input_channels': 3,
            'num_classes': 1,
            'base_channels': 16,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'U-Lite': {
        'class': ConfigurableULite,
        'params': {
            'input_channel': 3,
            'num_classes': 1,
            'base_channels': 16,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'U-Lite-DropBlock': {
        'class': ConfigurableULiteDropBlock,
        'params': {
            'input_channel': 3,
            'num_classes': 1,
            'base_channels': 16,
            'block_size': 7,
            'keep_prob': 0.9,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'SDA-U-Lite': {
        'class': ConfigurableSDAULite,
        'params': {
            'input_channel': 3,
            'num_classes': 1,
            'base_channels': 16,
            'block_size': 7,
            'keep_prob': 0.9,
            'attention_kernel_size': 7,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'W-Net': {
        'class': ConfigurableWNet18,
        'params': {
            'input_channels': 3,
            'num_classes': 1,
            'base_channels': 16,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'W-Lite': {
        'class': ConfigurableWLite,
        'params': {
            'input_channel': 3,
            'num_classes': 1,
            'base_channels': 16,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'W-Lite2': {
        'class': ConfigurableWLite2,
        'params': {
            'input_channel': 3,
            'num_classes': 1,
            'base_channels': 16,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'SDA-W-Lite': {
        'class': ConfigurableSDAWLite,
        'params': {
            'input_channel': 3,
            'num_classes': 1,
            'base_channels': 16,
            'block_size': 7,
            'keep_prob': 0.9,
            'attention_kernel_size': 7,
            'activation': 'gelu',
            'dropout_rate': 0.1
        }
    },
    'Detail-SA-UNet': {
        'class': DetailFocusedSAUNet,
        'params': {
            'input_channels': 3,
            'num_classes': 1,
            'start_neurons': 16
        }
    }
}

# ==================== TRAINING CONFIGURATIONS ====================
TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'scheduler': 'reduce_on_plateau',
    'scheduler_params': {
        'mode': 'max',
        'factor': 0.5,
        'patience': 5,
        'verbose': True
    },
    'loss': 'dice_bce',
    'loss_params': {
        'dice_weight': 0.5
    }
}

# ==================== AVAILABLE EPOCHS ====================
AVAILABLE_EPOCHS = [10, 25, 50, 100]
