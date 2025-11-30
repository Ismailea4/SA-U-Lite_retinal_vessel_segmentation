import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

# Assumes config.py and models/ exist as per your README structure
from config import MODEL_REGISTRY

def load_image(path, is_mask=False):
    """
    Loads and resizes image for plotting.
    is_mask: If True, loads as grayscale.
    """
    if not path.exists():
        return np.zeros((512, 512)) if is_mask else np.zeros((512, 512, 3))
    
    if is_mask:
        img = Image.open(path).convert('L') # Grayscale for masks
    else:
        img = Image.open(path).convert('RGB')
        
    img = img.resize((512, 512)) 
    arr = np.array(img) / 255.0
    return arr

def preprocess(img):
    """Prepares image for model inference (HWC -> CHW, Tensor)."""
    if np.all(img == 0): return None
    arr = np.transpose(img, (2, 0, 1))  # HWC to CHW
    arr = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return arr

def get_first_test_image(dataset):
    """Gets the first available image from the dataset test folder."""
    test_img_dir = Path(f'dataset/{dataset}/test/images')
    if not test_img_dir.exists():
        return None
    
    img_paths = sorted(list(test_img_dir.glob('*')))
    if img_paths:
        return img_paths[0] 
    return None

def get_corresponding_label(dataset, image_path):
    """Finds the label file corresponding to the image."""
    if image_path is None: return None
    
    label_dir = Path(f'dataset/{dataset}/test/labels')
    if not label_dir.exists():
        return None

    # Search for file with same name (ignoring extension difference if any)
    # This handles case where image is .jpg but label is .png or .gif
    target_stem = image_path.stem
    for label_path in label_dir.glob('*'):
        if label_path.stem == target_stem+"_label" or label_path.stem == target_stem:
            return label_path
            
    return None

def load_model_for_inference(model_name, dataset, epoch):
    """Loads a specific model architecture with weights trained on a specific dataset."""
    if model_name not in MODEL_REGISTRY:
        print(f"Error: {model_name} not found in MODEL_REGISTRY.")
        return None
        
    params = MODEL_REGISTRY[model_name]['params']
    model_class = MODEL_REGISTRY[model_name]['class']
    model = model_class(**params)
    
    weight_path = Path(f'results/{dataset}/{epoch}_epochs/models/best_{model_name}.pth')
    
    if not weight_path.exists():
        # print(f"Warning: Weights not found at {weight_path}")
        return None

    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load {model_name} for {dataset}: {e}")
        return None

def plot_grid(datasets, models, epoch):
    n_cols = len(datasets)
    # Rows: Original + Ground Truth + Models
    n_rows = len(models) + 2 
    
    # Large figure size, zero spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Shape handling for matplotlib axes
    if n_cols == 1: axes = axes.reshape(-1, 1)
    if n_rows == 1: axes = axes.reshape(1, -1)
    if len(axes.shape) == 1:
        axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)

    print(f"Generating Grid: {n_rows} Rows x {n_cols} Columns...")

    for col_idx, dataset in enumerate(datasets):
        
        # 1. Row 0: Original Image
        img_path = get_first_test_image(dataset)
        label_path = get_corresponding_label(dataset, img_path)
        
        # Plot Original
        if img_path:
            img = load_image(img_path)
            axes[0, col_idx].imshow(img)
            # Column Title
            axes[0, col_idx].set_title(dataset, fontsize=20, fontweight='bold', pad=15) 
        else:
            img = None
            axes[0, col_idx].text(0.5, 0.5, 'Img Missing', ha='center')
            axes[0, col_idx].set_title(dataset, fontsize=20, fontweight='bold')

        axes[0, col_idx].axis('off')

        # Row Label 0
        if col_idx == 0:
            axes[0, col_idx].text(-0.05, 0.5, "Original", va='center', ha='right', 
                                transform=axes[0, col_idx].transAxes, fontsize=16, fontweight='bold')

        # 2. Row 1: Ground Truth
        if label_path:
            lbl = load_image(label_path, is_mask=True)
            axes[1, col_idx].imshow(lbl, cmap='gray')
        else:
            axes[1, col_idx].text(0.5, 0.5, 'No GT', ha='center')
            axes[1, col_idx].set_facecolor('#f0f0f0')
            
        axes[1, col_idx].axis('off')

        # Row Label 1
        if col_idx == 0:
            axes[1, col_idx].text(-0.05, 0.5, "Ground Truth", va='center', ha='right', 
                                transform=axes[1, col_idx].transAxes, fontsize=16, fontweight='bold')

        # Prepare input for models
        inp = preprocess(img) if img is not None else None

        # 3. Row 2+: Models
        for i, model_name in enumerate(models):
            row_idx = i + 2
            ax = axes[row_idx, col_idx]
            
            model = load_model_for_inference(model_name, dataset, epoch)
            
            if model and inp is not None:
                with torch.no_grad():
                    out = model(inp)
                    out = torch.sigmoid(out)
                    mask = out.squeeze().cpu().numpy()
                ax.imshow(mask, cmap='gray')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.set_facecolor('#e0e0e0')

            ax.axis('off')

            # Row Labels for Models
            if col_idx == 0:
                ax.text(-0.05, 0.5, model_name, va='center', ha='right', 
                        transform=ax.transAxes, fontsize=16, fontweight='bold')

    output_filename = f"comparison_grid_{epoch}ep.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='Comma-separated datasets (e.g. CHASE,DRIVE)')
    parser.add_argument('--models', type=str, required=True, help='Comma-separated models (e.g. SA-U-Lite,U-Lite)')
    parser.add_argument('--epoch', type=int, default=5, help='Epoch folder to look for (default: 5)')
    args = parser.parse_args()
    
    d_list = [d.strip() for d in args.datasets.split(',')]
    m_list = [m.strip() for m in args.models.split(',')]
    
    plot_grid(d_list, m_list, args.epoch)