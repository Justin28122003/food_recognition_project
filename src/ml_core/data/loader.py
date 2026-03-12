from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    def create_loader(split: str, use_sampler: bool = False):
        x_p = str(base_path / f"camelyonpatch_level_2_split_{split}_x.h5")
        y_p = str(base_path / f"camelyonpatch_level_2_split_{split}_y.h5")

        # Using ToTensor handles the (C, H, W) conversion and scaling to [0, 1]
        ds = PCAMDataset(x_p, y_p, transform=transforms.ToTensor())

        sampler = None
        if use_sampler:
            # Flatten labels for weight calculation
            labels = ds.y_data[:].flatten()
            class_counts = np.bincount(labels)
            weights = 1.0 / class_counts[labels]
            sampler = WeightedRandomSampler(weights, len(weights))

        return DataLoader(
            ds,
            batch_size=data_cfg["batch_size"],
            sampler=sampler,
            num_workers=data_cfg.get("num_workers", 0),
            shuffle=(sampler is None),  # Shuffle only if not using sampler
        )

    train_loader = create_loader("train", use_sampler=True)
    val_loader = create_loader("valid", use_sampler=False)

    return train_loader, val_loader
