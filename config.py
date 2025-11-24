"""
Configuration file for model benchmarking
Centralized configuration for datasets, models, and training parameters
"""

import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dataset paths
DATASETS = {
    'CHASE': {
        'train_images': 'CHASE/train/images',
        'train_labels': 'CHASE/train/labels',
        'val_images': 'CHASE/validate/images',
        'val_labels': 'CHASE/validate/labels',
        'test_images': 'CHASE/test/images',
        'test_labels': 'CHASE/test/labels',
        'image_size': (256, 256)
    },
    'DRIVE': {
        'train_images': 'DRIVE/train/images',
        'train_labels': 'DRIVE/train/labels',
        'val_images': 'DRIVE/validate/images',
        'val_labels': 'DRIVE/validate/labels',
        'test_images': 'DRIVE/test/images',
        'test_labels': 'DRIVE/test/labels',
        'image_size': (256, 256)
    },
    'STARE': {
        'train_images': 'STARE/train/images',
        'train_labels': 'STARE/train/labels',
        'val_images': 'STARE/validate/images',
        'val_labels': 'STARE/validate/labels',
        'test_images': 'STARE/test/images',
        'test_labels': 'STARE/test/labels',
        'image_size': (256, 256)
    },
    'HRF': {
        'train_images': 'HRF/train/images',
        'train_labels': 'HRF/train/labels',
        'val_images': 'HRF/validate/images',
        'val_labels': 'HRF/validate/labels',
        'test_images': 'HRF/test/images',
        'test_labels': 'HRF/test/labels',
        'image_size': (256, 256)
    }
}

# ==================== MODEL CONFIGURATIONS ====================
from models.saulite import ConfigurableSAULite
from models.sdulite import ConfigurableULiteDropBlock
from models.sda_ulite import ConfigurableSDAULite
from models.w_lite import ConfigurableWLite
from models.sda_wlite import ConfigurableSDAWLite
from models.enhance_saunet import DetailFocusedSAUNet

MODEL_REGISTRY = {
    'SA-U-Lite': {
        'class': ConfigurableSAULite,
        'params': {
            'input_channel': 3,
            'num_classes': 1,
            'base_channels': 16,
            'activation': 'gelu',
            'dropout_rate': 0.1,
            'attention_kernel_size': 7
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
