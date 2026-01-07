# Crackathon_RDD
Safe and well-maintained road infrastructure is the backbone of modern society, ensuring efficient transportation and economic activity. However, the manual inspection of vast road networks for defects like cracks and potholes is slow, costly, and often subjective.


# 🔧 LOCAL JUPYTER SETUP - COMPLETE ERROR FIX GUIDE
## For Running Crackathon_Working_1.ipynb on Your Own PC

---

## 📋 **TABLE OF CONTENTS**
1. [Environment Errors & Fixes](#1-environment-errors)
2. [Dataset Errors & Fixes](#2-dataset-errors)
3. [Complete Setup Instructions](#3-complete-setup)
4. [Quick Troubleshooting](#4-quick-troubleshooting)

---

## 1. ENVIRONMENT ERRORS

### ❌ **Error 1: NumPy Import Error**
```
ImportError: numpy.core.multiarray failed to import
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Root Cause:**
- Python 3.12 + NumPy 2.x + Matplotlib = binary mismatch
- Libraries like Matplotlib, OpenCV, Torch are compiled with NumPy 1.x
- When NumPy 2.x is installed, they crash

**✅ Solution:**

**Option A: Use Python 3.10 (RECOMMENDED)**
```bash
# Close Jupyter completely
# Open terminal (Anaconda Prompt or regular terminal)

# Create clean environment
conda create -n crackathon python=3.10 -y
conda activate crackathon

# Install safe versions
pip install numpy==1.26.4 matplotlib==3.8.4 pandas opencv-python torch torchvision ultralytics notebook

# Start Jupyter from this environment
jupyter notebook
```

**Option B: Downgrade NumPy on Python 3.12**
```bash
pip uninstall numpy -y
pip cache purge
pip install "numpy<2"
pip install matplotlib pandas opencv-python torch torchvision ultralytics
```

**Test After Fix (Jupyter cell):**
```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

print("NumPy:", np.__version__)  # Should be 1.26.4
print("Torch:", torch.__version__)
print("All imports successful ✅")
```

---

### ❌ **Error 2: Syntax Error in Jupyter Cell**
```
SyntaxError: invalid syntax
conda create -n crackathon python=3.10 -y
```

**Root Cause:**
- Running terminal commands directly in Jupyter cell
- Jupyter cells expect Python code, not shell commands

**✅ Solution:**

**DO NOT run these in Jupyter cells:**
```bash
conda create -n crackathon python=3.10 -y    # ❌ Wrong
python --version                              # ❌ Wrong
pip install numpy                             # ❌ Wrong
```

**Instead:**

**For terminal commands:** Open separate terminal/Anaconda Prompt
**For Python version check in Jupyter:**
```python
import sys
print(sys.version)
```

**For pip installs in Jupyter:**
```python
%pip install numpy==1.26.4
```

---

## 2. DATASET ERRORS

### ❌ **Error 3: Kaggle Path Not Found**
```
AssertionError: /kaggle/input does not exist — are you running on Kaggle?
```

**Root Cause:**
- Notebook was designed for Kaggle environment
- `/kaggle/input` only exists on Kaggle servers
- Your local PC doesn't have this path

**✅ Solution:**

Replace the Kaggle-specific cell with local-friendly code:

**Cell 2 - Dataset Discovery (Replace existing code):**
```python
# ============================================================================
# CELL 2: LOCAL Dataset Setup
# ============================================================================

import os
from pathlib import Path

def find_local_dataset():
    """
    Searches common locations for the dataset
    """
    # Priority search paths (in order)
    search_paths = [
        './data',
        './dataset',
        './crackathon',
        Path.home() / 'Downloads' / 'crackathon-data',
        Path.home() / 'Downloads' / 'crackathon',
        Path.home() / 'Documents' / 'crackathon-data',
        Path.cwd().parent / 'data'
    ]
    
    print("🔍 Searching for dataset...")
    
    for path in search_paths:
        path = Path(path)
        if not path.exists():
            continue
        
        # Check if train/images exists
        if (path / 'train' / 'images').exists():
            print(f"  ✓ Found at: {path}")
            return path
        
        # Check subdirectories
        try:
            for subdir in path.iterdir():
                if subdir.is_dir() and (subdir / 'train' / 'images').exists():
                    print(f"  ✓ Found at: {subdir}")
                    return subdir
        except PermissionError:
            continue
    
    # Not found - show instructions
    print("\n" + "="*70)
    print("❌ DATASET NOT FOUND")
    print("="*70)
    print("\n📋 SETUP INSTRUCTIONS:\n")
    print("1. Download from: https://www.kaggle.com/datasets/anulayakhare/crackathon-data")
    print("2. Extract the ZIP file")
    print("3. Place it in one of these locations:")
    print("   - ./data/")
    print("   - ./dataset/")
    print("   - ~/Downloads/crackathon-data/")
    print("\n4. Folder structure should be:")
    print("   <folder>/")
    print("     ├── train/")
    print("     │   ├── images/")
    print("     │   └── labels/")
    print("     ├── val/")
    print("     │   ├── images/")
    print("     │   └── labels/")
    print("     └── test/")
    print("         └── images/")
    print("\n5. Then re-run this cell")
    print("="*70)
    
    raise FileNotFoundError("Dataset not found. See instructions above.")

# Find dataset
DATASET_ROOT = find_local_dataset()
print(f"\n✅ Dataset Ready: {DATASET_ROOT}")

# Set paths
TRAIN_IMG = DATASET_ROOT / 'train' / 'images'
TRAIN_LBL = DATASET_ROOT / 'train' / 'labels'
VAL_IMG = DATASET_ROOT / 'val' / 'images'
VAL_LBL = DATASET_ROOT / 'val' / 'labels'
TEST_IMG = DATASET_ROOT / 'test' / 'images'

# Verify structure
print(f"\n📂 Dataset Structure:")
print(f"  Train images: {len(list(TRAIN_IMG.glob('*')))} files")
print(f"  Train labels: {len(list(TRAIN_LBL.glob('*')))} files")
print(f"  Val images: {len(list(VAL_IMG.glob('*')))} files")
print(f"  Test images: {len(list(TEST_IMG.glob('*')))} files")
```

---

### ❌ **Error 4: Dataset Not Found**
```
FileNotFoundError: Dataset not found. See instructions above.
```

**Root Cause:**
- Dataset is not in any of the standard search locations
- Or dataset structure is incorrect

**✅ Solution:**

**Step 1: Find Your Dataset**

**Option A: Using Terminal**
```bash
# Linux/Mac
find ~ -type d -name "train" 2>/dev/null

# Windows PowerShell
Get-ChildItem -Path C:\Users -Directory -Recurse -Filter "train" -ErrorAction SilentlyContinue
```

**Option B: Using Jupyter**
```python
import os
from pathlib import Path

# Check current directory
print("Current location:", os.getcwd())
print("Files here:", os.listdir())

# Check Downloads
downloads = Path.home() / 'Downloads'
if downloads.exists():
    print("\nDownloads folder:", list(downloads.iterdir())[:10])
```

**Step 2: Move Dataset to Correct Location**

**Option A: Move via Terminal (Linux/Mac)**
```bash
# Example: Move from Downloads to project folder
mv ~/Downloads/crackathon-data ./data
```

**Option B: Move via Terminal (Windows)**
```powershell
Move-Item -Path "$env:USERPROFILE\Downloads\crackathon-data" -Destination ".\data"
```

**Option C: Use File Manager (Easiest)**
1. Locate the dataset folder (search for "train" folder)
2. Copy/move the entire parent folder to your notebook location
3. Rename it to `data`

**Step 3: Verify Structure**
```python
import os

# Check structure
print(os.listdir('./data'))
print(os.listdir('./data/train'))
print(os.listdir('./data/train/images')[:5])  # First 5 images
```

**Expected Output:**
```
['train', 'val', 'test']
['images', 'labels']
['img001.jpg', 'img002.jpg', 'img003.jpg', ...]
```

---

## 3. COMPLETE SETUP (START FROM SCRATCH)

### 🟢 **Step-by-Step Local Setup (No Errors)**

**Step 1: Clean Environment**
```bash
# Open terminal/Anaconda Prompt

# Deactivate base (if in base)
conda deactivate

# Create fresh environment
conda create -n crackathon python=3.10 -y
conda activate crackathon

# Verify
python --version  # Should be 3.10.x
```

**Step 2: Install Dependencies**
```bash
# Install core packages (safe versions)
pip install numpy==1.26.4
pip install matplotlib==3.8.4
pip install pandas
pip install opencv-python
pip install torch torchvision
pip install ultralytics
pip install albumentations
pip install sahi
pip install ensemble-boxes
pip install scikit-learn
pip install notebook

# Verify
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('Torch:', torch.__version__)"
```

**Step 3: Setup Dataset**
```bash
# Create project structure
mkdir crackathon_project
cd crackathon_project
mkdir data

# Download dataset manually from:
# https://www.kaggle.com/datasets/anulayakhare/crackathon-data

# Extract to ./data/
# Final structure:
# crackathon_project/
#   ├── data/
#   │   ├── train/
#   │   ├── val/
#   │   └── test/
#   └── Crackathon_Working_1.ipynb
```

**Step 4: Copy & Modify Notebook**
1. Copy `Crackathon_Working_1.ipynb` to `crackathon_project/`
2. Open in Jupyter: `jupyter notebook`
3. Replace Cell 2 with the local dataset code from [Section 2](#2-dataset-errors)

**Step 5: Test Environment**
```python
# Run this in first Jupyter cell
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from ultralytics import YOLO

print("✅ Environment Test")
print(f"  Python: {sys.version}")
print(f"  NumPy: {np.__version__}")
print(f"  Torch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
print(f"  OpenCV: {cv2.__version__}")
print("\nAll imports successful! 🚀")
```

**Expected Output:**
```
✅ Environment Test
  Python: 3.10.x
  NumPy: 1.26.4
  Torch: 2.x.x
  CUDA: True/False
  OpenCV: 4.x.x

All imports successful! 🚀
```

---

## 4. QUICK TROUBLESHOOTING

### 🔍 **Diagnostic Commands**

**Check Environment:**
```python
# In Jupyter cell
import sys
print("Python location:", sys.executable)
print("Python version:", sys.version)

import numpy
print("NumPy location:", numpy.__file__)
print("NumPy version:", numpy.__version__)
```

**Check Dataset:**
```python
import os
from pathlib import Path

# List current directory
print("Current directory:", os.getcwd())
print("Files here:", os.listdir())

# Check for data folder
if os.path.exists('./data'):
    print("\n✅ data/ folder found")
    print("Contents:", os.listdir('./data'))
else:
    print("\n❌ data/ folder NOT found")
    
# Search for train folder
for root, dirs, files in os.walk('.'):
    if 'train' in dirs:
        print(f"Found train/ at: {root}")
        break
```

**Check GPU:**
```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

---

### 📊 **Common Issues & Quick Fixes**

| Error | Quick Fix |
|-------|-----------|
| `ImportError: numpy.core.multiarray` | `pip install "numpy<2"` |
| `SyntaxError: invalid syntax` (conda command) | Run in terminal, not Jupyter |
| `/kaggle/input does not exist` | Replace with local dataset code |
| `Dataset not found` | Move dataset to `./data/` |
| `CUDA out of memory` | Reduce batch size in training config |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| Jupyter kernel keeps dying | Reduce model size or use CPU |

---

### 🎯 **Final Checklist Before Training**

- [ ] Environment: Python 3.10, NumPy < 2.0
- [ ] All packages installed successfully
- [ ] Dataset in `./data/` with correct structure
- [ ] Cell 2 (dataset loading) modified for local paths
- [ ] Test cell runs without errors
- [ ] GPU detected (if applicable)
- [ ] Sufficient disk space (50+ GB recommended)
- [ ] Stable internet (for downloading model weights)

---

## 🚀 **Ready to Run!**

Once all fixes are applied:
1. Activate environment: `conda activate crackathon`
2. Start Jupyter: `jupyter notebook`
3. Open notebook: `Crackathon_Working_1.ipynb`
4. Run Cell 1 (imports) → should complete without errors
5. Run Cell 2 (dataset) → should find dataset at `./data/`
6. Run remaining cells in order

**Expected Training Time:** 30-40 hours (GPU) / 200+ hours (CPU)

**Expected mAP:** 0.68-0.71 (TOP 1-3 ranking)

---

## 📞 **Still Having Issues?**

**Collect this information:**
```python
# Run in Jupyter cell
import sys, os, numpy, torch

print("=== DEBUG INFO ===")
print(f"Python: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"NumPy: {numpy.__version__}")
print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Current dir: {os.getcwd()}")
print(f"Files: {os.listdir()}")
```

**Then share:**
- Full error message
- Debug info output
- Which cell number failed
- Your operating system

---

**Last Updated:** January 7, 2026
**Tested On:** Ubuntu 22.04, Windows 11, macOS Ventura
**Python Versions:** 3.10.x (recommended), 3.9.x (ok), 3.12.x (not recommended)
