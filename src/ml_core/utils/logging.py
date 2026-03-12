import logging
import random
import numpy as np
import torch
import yaml
from typing import Any, Dict

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
    # TODO: Set seeds for random, numpy, torch, and cuda
    pass
