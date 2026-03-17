from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the project root (three levels up from this file)."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DataPaths:
    train_dir: Path
    test_dir: Path
    train_labels_csv: Path
    sample_csv: Path
    class_list: Path


def get_data_paths() -> DataPaths:
    """Return paths that match the actual on-disk layout."""
    raw = get_project_root() / "data" / "raw"
    return DataPaths(
        train_dir=raw / "train_set" / "train_set" / "train_set",
        test_dir=raw / "test_set" / "test_set" / "test_set",
        train_labels_csv=raw / "train_labels.csv",
        sample_csv=raw / "sample.csv",
        class_list=raw / "class_list_food.txt",
    )


def load_class_names(path: Path) -> List[str]:
    """Load the class-name list (1-indexed lines: ``id name``)."""
    names: List[str] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                names.append(parts[1])
    return names


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

NUM_CLASSES = 80  # labels in CSV are 1-80


class FoodDataset(Dataset):
    """PyTorch dataset for the food-recognition image set.

    Parameters
    ----------
    image_dir : Path
        Folder containing the JPG images.
    labels_df : DataFrame, optional
        DataFrame with columns ``img_name`` and ``label`` (1-indexed).
        When *None* the dataset is treated as an unlabelled test set.
    transform : torchvision transform, optional
        Applied to every PIL image before returning.
    """

    def __init__(
        self,
        image_dir: Path,
        labels_df: Optional[pd.DataFrame] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        if labels_df is not None:
            self.img_names: List[str] = labels_df["img_name"].tolist()
            # Shift 1-80 → 0-79 so labels match cross-entropy expectations
            self.labels: Optional[List[int]] = (labels_df["label"] - 1).tolist()
        else:
            self.img_names = sorted(f.name for f in image_dir.glob("*.jpg"))
            self.labels = None

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int):
        img_path = self.image_dir / self.img_names[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[idx]
        # For the test set return the filename so predictions can be mapped back
        return image, self.img_names[idx]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    resize_base = int(max(image_size) * 1.1)
    return transforms.Compose([
        transforms.Resize((resize_base, resize_base)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
    ])


def get_eval_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    resize_base = int(max(image_size) * 1.1)
    return transforms.Compose([
        transforms.Resize((resize_base, resize_base)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def get_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation ``DataLoader``s from *config*.

    Expected config keys (under ``data``)::

        batch_size:  32
        num_workers: 2
        val_split:   0.2
        image_size:  [224, 224]

    The split is seeded with ``config["seed"]`` (default 42) for
    reproducibility.
    """
    paths = get_data_paths()

    data_cfg = config["data"]
    batch_size: int = data_cfg.get("batch_size", 32)
    num_workers: int = data_cfg.get("num_workers", 2)
    val_split: float = data_cfg.get("val_split", 0.2)
    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    seed: int = config.get("seed", 42)

    # Read labels and perform a reproducible stratified train / val split
    labels_df = pd.read_csv(paths.train_labels_csv)

    train_df, val_df = train_test_split(
        labels_df,
        test_size=val_split,
        random_state=seed,
        stratify=labels_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = FoodDataset(
        paths.train_dir, train_df, transform=get_train_transform(image_size),
    )
    val_dataset = FoodDataset(
        paths.train_dir, val_df, transform=get_eval_transform(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
