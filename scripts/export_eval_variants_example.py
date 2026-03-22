from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from ml_core.data.loader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_data_paths,
    get_eval_transform,
)
from ml_core.utils import load_config, seed_everything


VALID_VARIANTS = ["clean", "blur", "grayscale", "crop", "occlusion", "lowres"]


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image_tensor = image_tensor.cpu() * std + mean
    return image_tensor.clamp(0.0, 1.0)


def tensor_to_numpy_image(image_tensor: torch.Tensor):
    """Convert CHW tensor in [0,1] to HWC numpy image."""
    return denormalize_image(image_tensor).permute(1, 2, 0).numpy()


def get_validation_dataframe(config: dict) -> pd.DataFrame:
    """Recreate the exact train/validation split used by the repo."""
    paths = get_data_paths()
    labels_df = pd.read_csv(paths.train_labels_csv)

    val_split = config["data"].get("val_split", 0.2)
    seed = config.get("seed", 42)

    _, val_df = train_test_split(
        labels_df,
        test_size=val_split,
        random_state=seed,
        stratify=labels_df["label"],
    )
    return val_df.reset_index(drop=True)


def load_validation_image(config: dict, image_name: str | None = None, image_index: int = 0):
    """Load one raw validation image by filename or index."""
    paths = get_data_paths()
    val_df = get_validation_dataframe(config)

    if image_name is not None:
        matches = val_df[val_df["img_name"] == image_name]
        if matches.empty:
            raise ValueError(f"Image '{image_name}' not found in validation split.")
        row = matches.iloc[0]
    else:
        if image_index < 0 or image_index >= len(val_df):
            raise IndexError(f"image_index {image_index} out of range for val set of size {len(val_df)}.")
        row = val_df.iloc[image_index]

    img_name = row["img_name"]
    label = int(row["label"]) - 1  # convert to 0-based label to match repo convention
    img_path = paths.train_dir / img_name

    image = Image.open(img_path).convert("RGB")
    return image, img_name, label


def save_single_variant(image_np, save_path: Path, title: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(image_np)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_combined_figure(images: dict[str, Any], save_path: Path, img_name: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    flat_axes = axes.flatten()

    for ax, (variant, image_np) in zip(flat_axes, images.items()):
        ax.imshow(image_np)
        ax.set_title(variant)
        ax.axis("off")

    for ax in flat_axes[len(images):]:
        ax.axis("off")

    fig.suptitle(f"Evaluation variants for {img_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export one validation image under all deterministic evaluation variants."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default=None,
        help="Specific validation image filename, e.g. train_12345.jpg",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Validation index to use if --image-name is not given",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug_eval_variants",
        help="Directory to save exported images",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    image_size = tuple(config["data"].get("image_size", [224, 224]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_image, img_name, label = load_validation_image(
        config=config,
        image_name=args.image_name,
        image_index=args.image_index,
    )

    rendered_images: dict[str, Any] = {}

    for variant in VALID_VARIANTS:
        transform = get_eval_transform(image_size=image_size, variant=variant)
        transformed_tensor = transform(raw_image)
        rendered_images[variant] = tensor_to_numpy_image(transformed_tensor)

        single_path = output_dir / f"{Path(img_name).stem}_{variant}.png"
        save_single_variant(
            image_np=rendered_images[variant],
            save_path=single_path,
            title=variant,
        )

    combined_path = output_dir / f"{Path(img_name).stem}_all_variants.png"
    save_combined_figure(
        images=rendered_images,
        save_path=combined_path,
        img_name=img_name,
    )

    print(f"Saved variant images to: {output_dir}")
    print(f"Image used: {img_name}")
    print(f"0-based label: {label}")
    print(f"Combined figure: {combined_path}")


if __name__ == "__main__":
    main()