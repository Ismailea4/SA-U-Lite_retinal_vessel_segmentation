# U-Net Model Variants

This folder contains organized implementations of different U-Net architectures for retinal blood vessel segmentation.

## Model Files

### 1. ulite.py - U-Lite (Lightweight U-Net)
**Architecture**: Lightweight U-Net with axial depthwise convolutions

**Key Features**:
- Axial depthwise convolutions for efficient feature extraction
- Bottleneck block with dilated convolutions
- GELU activation by default
- Significantly fewer parameters than standard U-Net

**Classes**:
- `ULite`: Base lightweight U-Net
- `ConfigurableULite`: Variant with configurable hyperparameters
- `AxialDW`: Axial depthwise convolution module
- `EncoderBlock`, `DecoderBlock`, `BottleNeckBlock`: Building blocks

**Usage**:
```python
from models.ulite import ConfigurableULite

model = ConfigurableULite(
    input_channel=3,
    num_classes=1,
    base_channels=128,
    activation='elu',
    dropout_rate=0.1
)
```

### 2. unet18.py - Classical U-Net 18 Layers
**Architecture**: Standard U-Net with 18 convolutional layers

**Key Features**:
- 5 encoder levels with skip connections
- 1 bottleneck layer
- 4 decoder levels
- Configurable activation functions and batch normalization

**Classes**:
- `ClassicalUNet18`: Main architecture
- `Conv2dBlock`: Reusable convolutional block

**Usage**:
```python
from models.unet18 import ClassicalUNet18

model = ClassicalUNet18(
    input_channels=3,
    num_classes=1,
    start_filters=64,
    activation='relu',
    use_batchnorm=True,
    dropout_rate=0.2
)
```

### 3. sdunet.py - SD-UNet (Structured Dropout U-Net)
**Architecture**: U-Net with DropBlock for structured regularization

**Key Features**:
- DropBlock2D for better regularization than standard dropout
- Configurable block size and keep probability
- Pure PyTorch implementation (no Keras dependencies)

**Classes**:
- `ConfigurableSDUNet`: Main architecture
- `DropBlock2D`: Structured dropout module
- `SDUNetEncoderBlock`, `SDUNetDecoderBlock`: Building blocks

**Usage**:
```python
from models.sdunet import ConfigurableSDUNet

model = ConfigurableSDUNet(
    input_channels=3,
    num_classes=1,
    start_neurons=64,
    block_size=9,
    keep_prob=0.9,
    activation='elu',
    use_batchnorm=True,
    dropout_rate=0.1
)
```

### 4. saunet.py - SA-UNet (Spatial Attention U-Net)
**Architecture**: U-Net with spatial attention mechanism

**Key Features**:
- Spatial attention module in bottleneck
- Focuses on important regions in feature maps
- DropBlock regularization
- Configurable attention kernel size

**Classes**:
- `ConfigurableSAUNet`: Main architecture
- `SpatialAttention`: Spatial attention module
- `SAUNetEncoderBlock`, `SAUNetDecoderBlock`: Building blocks

**Usage**:
```python
from models.saunet import ConfigurableSAUNet

model = ConfigurableSAUNet(
    input_channels=3,
    num_classes=1,
    start_neurons=64,
    block_size=5,
    keep_prob=0.9,
    activation='relu',
    use_batchnorm=True,
    dropout_rate=0.2,
    attention_kernel_size=7
)
```

## Optimal Hyperparameters (from ABC Optimization)

Based on ABC optimization results on the CHASE dataset:

### U-Lite (Best: Dice=0.5422)
```python
{
    'optimizer': 'rmsprop',
    'learning_rate': 0.001,
    'activation': 'elu',
    'base_channels': 128,
    'dropout_rate': 0.1
}
```

### Classical U-Net 18 (Best: Dice=0.3666)
```python
{
    'optimizer': 'rmsprop',
    'learning_rate': 0.001,
    'activation': 'relu',
    'start_filters': 64,
    'use_batchnorm': True,
    'dropout_rate': 0.2
}
```

### SD-UNet (Best: Dice=0.2913)
```python
{
    'optimizer': 'rmsprop',
    'learning_rate': 0.0001,
    'activation': 'elu',
    'start_neurons': 64,
    'block_size': 9,
    'keep_prob': 0.9,
    'use_batchnorm': True,
    'dropout_rate': 0.1
}
```

### SA-UNet (Best: Dice=0.3373)
```python
{
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'activation': 'relu',
    'start_neurons': 64,
    'block_size': 5,
    'keep_prob': 0.9,
    'use_batchnorm': True,
    'dropout_rate': 0.2,
    'attention_kernel_size': 7
}
```

## Common Parameters

All models support the following common parameters:

- **input_channels** (int): Number of input channels (default: 3 for RGB)
- **num_classes** (int): Number of output classes (default: 1 for binary segmentation)
- **activation** (str): Activation function ('relu', 'elu', 'tanh', 'gelu', 'leaky_relu')
- **dropout_rate** (float): Dropout rate before output layer (0.0 to 0.5)

## Model-Specific Parameters

### U-Lite
- `base_channels`: Starting number of channels (16, 32, 64, 128)

### Classical U-Net 18
- `start_filters`: Starting number of filters (32, 64, 128)
- `use_batchnorm`: Whether to use batch normalization

### SD-UNet & SA-UNet
- `start_neurons`: Starting number of neurons/channels
- `block_size`: Size of DropBlock (5, 7, 9)
- `keep_prob`: Probability of keeping activation in DropBlock (0.8-0.95)
- `use_batchnorm`: Whether to use batch normalization

### SA-UNet Only
- `attention_kernel_size`: Kernel size for spatial attention (3, 5, 7)

## Integration with Notebook

To use these models in the main notebook:

```python
# Import models
from models import ConfigurableULite, ClassicalUNet18, ConfigurableSDUNet, ConfigurableSAUNet

# Create and train models
model = ConfigurableULite(input_channel=3, num_classes=1, base_channels=128)
```

## File Organization

```
models/
├── __init__.py          # Package initialization and exports
├── ulite.py            # U-Lite implementation
├── unet18.py           # Classical U-Net 18 implementation
├── sdunet.py           # SD-UNet implementation
├── saunet.py           # SA-UNet implementation
└── README.md           # This file
```

## Notes

- All models are implemented in pure PyTorch
- No Keras dependencies
- Compatible with ABC optimization framework
- Designed for retinal blood vessel segmentation but adaptable to other segmentation tasks
