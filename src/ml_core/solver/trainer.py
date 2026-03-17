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
        training_cfg = config["training"]
        self.scheduler_interval = training_cfg.get("scheduler_interval", "epoch").lower()
        self.mixup_alpha = float(training_cfg.get("mixup_alpha", 0.0))
        self.cutmix_alpha = float(training_cfg.get("cutmix_alpha", 0.0))
        self.mix_prob = float(training_cfg.get("mix_prob", 0.0))

    @staticmethod
    def _sample_mix_lambda(alpha: float) -> float:
        beta_dist = torch.distributions.Beta(alpha, alpha)
        return float(beta_dist.sample().item())

    @staticmethod
    def _rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = torch.randint(0, width, (1,)).item()
        cy = torch.randint(0, height, (1,)).item()

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, width)
        y2 = min(cy + cut_h // 2, height)
        return x1, y1, x2, y2

    def _maybe_mix_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        if self.mix_prob <= 0:
            return inputs, targets, targets, 1.0

        if torch.rand(1).item() >= self.mix_prob:
            return inputs, targets, targets, 1.0

        batch_size = inputs.size(0)
        index = torch.randperm(batch_size, device=inputs.device)
        targets_a = targets
        targets_b = targets[index]

        use_cutmix = self.cutmix_alpha > 0 and (
            self.mixup_alpha <= 0 or torch.rand(1).item() < 0.5
        )

        if use_cutmix:
            lam = self._sample_mix_lambda(self.cutmix_alpha)
            _, _, height, width = inputs.shape
            x1, y1, x2, y2 = self._rand_bbox(width=width, height=height, lam=lam)
            mixed_inputs = inputs.clone()
            mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
            patch_area = (x2 - x1) * (y2 - y1)
            lam_adjusted = 1.0 - patch_area / float(width * height)
            return mixed_inputs, targets_a, targets_b, lam_adjusted

        if self.mixup_alpha > 0:
            lam = self._sample_mix_lambda(self.mixup_alpha)
            mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
            return mixed_inputs, targets_a, targets_b, lam

        return inputs, targets, targets, 1.0

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

                    mixed_inputs, targets_a, targets_b, lam = self._maybe_mix_batch(inputs, targets)
                    logits = self.model(mixed_inputs)
                    loss = lam * self.criterion(logits, targets_a) + (1.0 - lam) * self.criterion(
                        logits,
                        targets_b,
                    )
                else:
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)

                if training:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None and self.scheduler_interval == "batch":
                        self.scheduler.step()

                batch_size = targets.size(0)
                total_examples += batch_size
                total_loss += loss.item() * batch_size
                if training and (self.mixup_alpha > 0 or self.cutmix_alpha > 0):
                    predictions = logits.argmax(dim=1)
                    total_correct += (
                        lam * (predictions == targets_a).sum().item()
                        + (1.0 - lam) * (predictions == targets_b).sum().item()
                    )
                else:
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

                if self.scheduler is not None and self.scheduler_interval == "epoch":
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
