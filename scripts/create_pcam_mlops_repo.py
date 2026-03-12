import os
from pathlib import Path

def create_file(path, content=""):
    """Creates a file with specific content, ensuring parent directories exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Created: {path}")

# ==========================================
# 1. SETUP & CONFIG FILES
# ==========================================

README_CONTENT = """
# MLOps Course: Medical Image Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Build Status](https://github.com/yourusername/mlops_course/actions/workflows/ci.yml/badge.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A repo exemplifying **MLOps best practices**: modularity, reproducibility, automation, and experiment tracking.

This project implements a standardized workflow for training neural networks on medical data (PCAM/TCGA).

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up your isolated environment.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# 2. Install the package in "Editable" mode
pip install -e .

# 3. Install pre-commit hooks
pre-commit install
```

### 2. Verify Setup
```bash
pytest tests/
```

### 3. Run an Experiment
```bash
python experiments/train.py --config experiments/configs/train_config.yaml
```

---

## 📂 Project Structure

```text
.
├── src/ml_core/          # The Source Code (Library)
│   ├── data/             # Data loaders and transformations
│   ├── models/           # PyTorch model architectures
│   ├── solver/           # Trainer class and loops
│   └── utils/            # Loggers and experiment trackers
├── experiments/          # The Laboratory
│   ├── configs/          # YAML files for hyperparameters
│   ├── results/          # Checkpoints and logs (Auto-generated)
│   └── train.py          # Entry point for training
├── scripts/              # Helper scripts (plotting, etc)
├── tests/                # Unit tests for QA
├── pyproject.toml        # Config for Tools (Ruff, Pytest)
└── setup.py              # Package installation script
```
"""

SETUP_PY = """
from setuptools import setup, find_packages

setup(
    name="ml_core",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "h5py",
        "pyyaml",
        "tqdm",
        "torcheval",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "pandas"
    ],
)
"""

REQUIREMENTS_TXT = """
numpy
torch
torchvision
h5py
pyyaml
tqdm
torcheval
tensorboard
matplotlib
seaborn
pandas
pytest
ruff
pre-commit
"""

PYPROJECT_TOML = """
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.ruff]
line-length = 88
target-version = "py39"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
"""

TRAIN_CONFIG = """
experiment_name: "pcam_mlp_baseline"
seed: 42

data:
  dataset_type: "pcam"
  # TODO: Students must set their absolute path here
  data_path: "./data/camelyonpatch_level_2" 
  input_shape: [3, 96, 96]
  batch_size: 32
  num_workers: 2

model:
  hidden_units: [64, 32]
  dropout_rate: 0.2
  num_classes: 2

training:
  epochs: 5
  learning_rate: 0.001
  save_dir: "./experiments/results"
"""

# ==========================================
# 2. SOURCE CODE (SKELETONS)
# ==========================================

PCAM_PY = """
from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    \"\"\"
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    \"\"\"

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        # TODO: Initialize dataset
        # 1. Check if files exist
        # 2. Open h5 files in read mode
        pass

    def __len__(self) -> int:
        # TODO: Return length of dataset
        return 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        
        raise NotImplementedError("Implement __getitem__ in PCAMDataset")
"""

LOADER_PY = """
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    \"\"\"
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    \"\"\"
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    # train_transform = ...
    # val_transform = ...

    # TODO: Define Paths for X and Y (train and val)
    
    # TODO: Instantiate PCAMDataset for train and val

    # TODO: Create DataLoaders
    # train_loader = ...
    # val_loader = ...
    
    raise NotImplementedError("Implement get_dataloaders")
"""

MLP_PY = """
from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        # TODO: Build the MLP architecture
        # Hint: Flatten -> [Linear -> ReLU -> Dropout] * N -> Linear
        
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        pass
"""

TRAINER_PY = """
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO: Import your metrics or utils if needed

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # TODO: Define Loss Function (Criterion)
        self.criterion = None

        # TODO: Initialize ExperimentTracker
        self.tracker = None

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        
        # TODO: Implement Training Loop
        # 1. Iterate over dataloader
        # 2. Move data to device
        # 3. Forward pass, Calculate Loss
        # 4. Backward pass, Optimizer step
        # 5. Track metrics (Loss, Accuracy, F1)
        
        raise NotImplementedError("Implement train_epoch")

    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        
        # TODO: Implement Validation Loop
        # Remember: No gradients needed here
        
        raise NotImplementedError("Implement validate")

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        pass

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            pass
"""

LOGGING_PY = """
import logging
import random
import numpy as np
import torch
import yaml
from typing import Any, Dict

def setup_logger(name: str = "MLOps_Course") -> logging.Logger:
    \"\"\"Configures a standardized logger.\"\"\"
    # TODO: Configure logging to stream to console with a specific format
    # Format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    pass

def load_config(path: str) -> Dict[str, Any]:
    \"\"\"Safely loads a yaml configuration file.\"\"\"
    # TODO: Load yaml file
    pass

def seed_everything(seed: int):
    \"\"\"Ensures reproducibility across numpy, random, and torch.\"\"\"
    # TODO: Set seeds for random, numpy, torch, and cuda
    pass
"""

TRACKER_PY = """
import csv
from pathlib import Path
from typing import Any, Dict
import yaml

# TODO: Add TensorBoard Support

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Save config to yaml in run_dir

        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header (SKELETON: Students need to add the rest)
        self.csv_writer.writerow(["epoch"]) 

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        \"\"\"
        Writes metrics to CSV (and TensorBoard).
        \"\"\"
        # TODO: Write full metrics to CSV
        self.csv_writer.writerow([epoch]) # Currently only logging epoch
        self.csv_file.flush()

        # TODO: Log to TensorBoard

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
"""

TRAIN_SCRIPT = """
import argparse
import torch
import torch.optim as optim
# from ml_core.data import get_dataloaders
# from ml_core.models import MLP
# from ml_core.solver import Trainer
# from ml_core.utils import load_config, seed_everything, setup_logger

# logger = setup_logger("Experiment_Runner")

def main(args):
    # 1. Load Config & Set Seed
    # config = load_config(args.config)
    
    # 2. Setup Device
    
    # 3. Data
    # train_loader, val_loader = get_dataloaders(config)
    
    # 4. Model
    # model = MLP(...)
    
    # 5. Optimizer
    # optimizer = optim.Adam(...)
    
    # 6. Trainer & Fit
    # trainer = Trainer(...)
    # trainer.fit(...)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    # main(args)
    print("Skeleton: Implement main logic first.")
"""

PLOT_SCRIPT = """
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()

def load_data(file_path: Path) -> pd.DataFrame:
    # TODO: Load CSV into Pandas DataFrame
    pass

def setup_style():
    # TODO: Set seaborn theme
    pass

def plot_metrics(df: pd.DataFrame, output_path: Optional[Path]):
    \"\"\"
    Generate and save plots for Loss, Accuracy, and F1.
    \"\"\"
    if df is None: return

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # TODO: Plot Train/Val Loss
    
    # TODO: Plot Train/Val Accuracy
    
    # TODO: Plot Learning Rate
    
    plt.tight_layout()
    plt.show() # or save to output_path

def main():
    args = parse_args()
    setup_style()
    df = load_data(args.input_csv)
    plot_metrics(df, args.output_dir)

if __name__ == "__main__":
    main()
"""

# ==========================================
# 3. REPO BUILDER LOGIC
# ==========================================

# Directory mapping
structure = {
    # --- Root Level ---
    ".gitignore": "__pycache__/\n*.pyc\nexperiments/logs/\nexperiments/results/\n.env\n.DS_Store",
    "README.md": README_CONTENT,
    "requirements.txt": REQUIREMENTS_TXT,
    "setup.py": SETUP_PY,
    "pyproject.toml": PYPROJECT_TOML,

    # --- Source Code ---
    "src/ml_core/__init__.py": "__version__ = '0.1.0'",
    
    # Data Module
    "src/ml_core/data/__init__.py": """
from .loader import get_dataloaders
from .pcam import PCAMDataset

__all__ = ["get_dataloaders", "PCAMDataset"]
""",
    "src/ml_core/data/pcam.py": PCAM_PY,
    "src/ml_core/data/loader.py": LOADER_PY,
    
    # Models Module
    "src/ml_core/models/__init__.py": """
from .mlp import MLP

# This stops linters from thinking MLP is "unused".
__all__ = ["MLP"]
""",
    "src/ml_core/models/mlp.py": MLP_PY,
    
    # Solver Module
    "src/ml_core/solver/__init__.py": """
from .trainer import Trainer

__all__ = ["Trainer"]
""",
    "src/ml_core/solver/trainer.py": TRAINER_PY,
    
    # Utils
    "src/ml_core/utils/__init__.py": """
from .logging import load_config, seed_everything, setup_logger
from .tracker import ExperimentTracker

__all__ = ["setup_logger", "seed_everything", "load_config", "ExperimentTracker"]
""",
    "src/ml_core/utils/logging.py": LOGGING_PY,
    "src/ml_core/utils/tracker.py": TRACKER_PY,

    # --- Experiments ---
    "experiments/configs/train_config.yaml": TRAIN_CONFIG,
    "experiments/train.py": TRAIN_SCRIPT,
    
    # --- Scripts ---
    "scripts/plotting/plot_results_csv.py": PLOT_SCRIPT,
    
    # --- Tests (Basic Placeholder) ---
    "tests/__init__.py": "",
    "tests/test_imports.py": """
def test_imports():
    from ml_core.data import PCAMDataset
    from ml_core.models import MLP
    assert True
"""
}

def build_repo():
    print("--- Scaffolding PCAM MLOps Repository ---")
    for filepath, content in structure.items():
        create_file(filepath, content)

    print("\n" + "=" * 40)
    print("SUCCESS: Repository Structure Created")
    print("=" * 40)
    print("Next Steps for Students:")
    print("1. Create venv:   python -m venv venv")
    print("2. Activate:      source venv/bin/activate")
    print("3. Install:       pip install -e .")
    print("4. Fill Skeletons in src/ml_core/...")

if __name__ == "__main__":
    build_repo()