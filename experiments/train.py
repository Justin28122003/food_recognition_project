import argparse
from pathlib import Path

import torch
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import create_model
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything, setup_logger

logger = setup_logger("Experiment_Runner")


def build_optimizer(model: torch.nn.Module, config):
    training_cfg = config["training"]
    optimizer_name = training_cfg.get("optimizer", "adamw").lower()
    learning_rate = training_cfg.get("learning_rate", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 1e-4)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if optimizer_name == "sgd":
        return optim.SGD(
            trainable_parameters,
            lr=learning_rate,
            momentum=training_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, config, steps_per_epoch: int):
    training_cfg = config["training"]
    scheduler_name = training_cfg.get("scheduler", "cosine").lower()

    if scheduler_name == "none":
        return None

    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_cfg.get("step_size", 5),
            gamma=training_cfg.get("gamma", 0.1),
        )

    if scheduler_name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_cfg.get("max_learning_rate", training_cfg.get("learning_rate", 1e-3)),
            epochs=training_cfg["epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=training_cfg.get("onecycle_pct_start", 0.1),
            div_factor=training_cfg.get("onecycle_div_factor", 25.0),
            final_div_factor=training_cfg.get("onecycle_final_div_factor", 1e4),
            anneal_strategy=training_cfg.get("onecycle_anneal_strategy", "cos"),
        )

    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_cfg["epochs"],
        eta_min=training_cfg.get("min_learning_rate", 1e-6),
    )

def main(args):
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    train_loader, val_loader = get_dataloaders(config)
    model = create_model(config["model"])
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        scheduler=scheduler,
    )
    trainer.fit(train_loader, val_loader)

    logger.info(
        "Training finished. Outputs saved to %s",
        Path(config["training"].get("save_dir", "experiments/results")) / config["experiment_name"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN baseline for food recognition")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)
