from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml_core.data import FoodDataset, get_data_paths, get_eval_transform
from ml_core.models import create_model
from ml_core.utils import load_config, seed_everything, setup_logger


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV from a trained checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logger = setup_logger("Submission")

    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = create_model(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    paths = get_data_paths()
    data_cfg = config["data"]
    image_size = tuple(data_cfg.get("image_size", [224, 224]))

    test_dataset = FoodDataset(
        paths.test_dir,
        labels_df=None,
        transform=get_eval_transform(image_size=image_size, variant="clean"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
    )

    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Number of test images: %d", len(test_dataset))

    submission_rows: list[dict[str, int | str]] = []

    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            pred_indices = logits.argmax(dim=1)
            pred_labels = (pred_indices + 1).cpu().tolist()  # Convert 0..79 to 1..80

            for img_name, label in zip(img_names, pred_labels):
                submission_rows.append({"img_name": img_name, "label": int(label)})

    submission_df = pd.DataFrame(submission_rows, columns=["img_name", "label"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    logger.info("Saved submission to: %s", output_path)
    logger.info("Submission generated successfully.")


if __name__ == "__main__":
    main()
