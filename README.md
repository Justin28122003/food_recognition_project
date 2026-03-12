<<<<<<< HEAD
# Food Recognition Challenge
=======
# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Build Status](https://github.com/yourusername/mlops_course/actions/workflows/ci.yml/badge.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A repo exemplifying **MLOps best practices**: modularity, reproducibility, automation, and experiment tracking.

This project implements a standardized workflow for training neural networks on medical data (PCAM/TCGA). 

The idea is that you fill in the repository with the necessary functions so you can execute the ```train.py``` function. Please also fill in this ```README.md``` clearly to setup, install and run your code. 

Don't forget to setup CI and linting!

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up your isolated environment.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
>>>>>>> f50f7e83 (Import files from MLOps_2026)
