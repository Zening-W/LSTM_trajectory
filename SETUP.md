# Environment Setup Guide

## Quick Setup

### Option 1: Using pip (Recommended)
```bash
# Create a virtual environment (recommended)
python -m venv lstm_env
source lstm_env/bin/activate  # On Windows: lstm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Create conda environment
conda create -n lstm_env python=3.8
conda activate lstm_env

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## System Requirements

- **Python**: 3.7+ (3.8+ recommended)
- **CUDA**: Optional but recommended for GPU acceleration
- **RAM**: 8GB+ recommended for training
- **Storage**: At least 2GB free space for models and data

## GPU Support (Optional)

If you have an NVIDIA GPU and want to use CUDA acceleration:

1. **Check CUDA compatibility**:
   ```bash
   nvidia-smi
   ```

2. **Install PyTorch with CUDA support**:
   - Visit [PyTorch website](https://pytorch.org/get-started/locally/)
   - Select your CUDA version
   - Use the provided installation command

## Project Structure

```
lstm_traj/
├── train.py          # Main training script
├── model.py          # LSTM model definition
├── dataloader.py     # Data loading and preprocessing
├── utils.py          # Utility functions
├── vis.py            # Visualization scripts
├── clip_vdo.py       # Video processing
├── create_video.py   # Video creation
├── data/             # Data directory (create if needed)
├── runs/             # Training outputs
├── test/             # Test outputs
└── videos/           # Generated videos
```

## Data Requirements

The project expects trajectory data in the following format:
- Text files with space-separated values
- Columns: frame_idx, cls, id, bbox_left, bbox_top, bbox_w, bbox_h
- Place data files in the `data/` directory

## Usage

1. **Training**:
   ```bash
   python train.py
   ```

2. **Testing/Visualization**:
   ```bash
   python train.py  # Uncomment test() function
   ```

3. **Video Processing**:
   ```bash
   python clip_vdo.py
   python create_video.py
   ```

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch_size in train.py
2. **Import errors**: Ensure all dependencies are installed
3. **Data not found**: Check data directory structure

### Windows-specific:
- Use `lstm_env\Scripts\activate` instead of `source lstm_env/bin/activate`
- Install Visual C++ build tools if needed for some packages

## Verification

To verify your setup is working:

```python
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("All imports successful!")
``` 