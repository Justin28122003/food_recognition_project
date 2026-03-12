import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger


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
        
        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed

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
            
	# Remember to handle the trackers properly
