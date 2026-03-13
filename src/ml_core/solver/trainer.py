import time
from typing import Any, Dict, Optional, Tuple

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
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.logger = setup_logger("Trainer")
        label_smoothing = config["training"].get("label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.tracker = ExperimentTracker(
            experiment_name=config["experiment_name"],
            config=config,
            base_dir=config["training"].get("save_dir", "experiments/results"),
        )
        self.best_val_acc = float("-inf")
        self.best_val_loss = float("inf")

    def _run_epoch(
        self,
        dataloader: DataLoader,
        training: bool,
        epoch_idx: int,
    ) -> Tuple[float, float, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        start_time = time.time()

        progress_bar = tqdm(
            dataloader,
            desc=f"{'Train' if training else 'Val'} {epoch_idx + 1}",
            leave=False,
        )

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for inputs, targets in progress_bar:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                if training:
                    loss.backward()
                    self.optimizer.step()

                batch_size = targets.size(0)
                total_examples += batch_size
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == targets).sum().item()

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(total_correct / total_examples):.4f}",
                )

        epoch_loss = total_loss / max(total_examples, 1)
        epoch_acc = total_correct / max(total_examples, 1)
        epoch_time = time.time() - start_time
        return epoch_loss, epoch_acc, epoch_time

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        return self._run_epoch(dataloader=dataloader, training=True, epoch_idx=epoch_idx)

    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        return self._run_epoch(dataloader=dataloader, training=False, epoch_idx=epoch_idx)

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        last_checkpoint = self.tracker.get_checkpoint_path("last.pt")
        torch.save(checkpoint, last_checkpoint)

        if is_best:
            best_checkpoint = self.tracker.get_checkpoint_path("best.pt")
            torch.save(checkpoint, best_checkpoint)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]

        self.logger.info("Starting training for %s epochs", epochs)

        try:
            for epoch in range(epochs):
                train_loss, train_acc, train_time = self.train_epoch(train_loader, epoch)
                val_loss, val_acc, val_time = self.validate(val_loader, epoch)

                if self.scheduler is not None:
                    self.scheduler.step()

                learning_rate = self.optimizer.param_groups[0]["lr"]
                metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_epoch_seconds": train_time,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_epoch_seconds": val_time,
                    "lr": learning_rate,
                }
                self.tracker.log_metrics(epoch=epoch + 1, metrics=metrics)

                is_best = val_acc > self.best_val_acc or (
                    val_acc == self.best_val_acc and val_loss < self.best_val_loss
                )
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss

                self.save_checkpoint(
                    epoch=epoch + 1,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    is_best=is_best,
                )

                self.logger.info(
                    "Epoch %s/%s | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.6f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    learning_rate,
                )
        finally:
            self.tracker.close()
