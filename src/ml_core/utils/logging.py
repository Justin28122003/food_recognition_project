from typing import Any, Dict

import logging
import random

import numpy as np
import torch
import yaml

def setup_logger(name: str = "MLOps_Course") -> logging.Logger:
    """Configures a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(path: str) -> Dict[str, Any]:
    """Safely loads a yaml configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int):
    """Ensures reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
