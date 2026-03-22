from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from ml_core.data import get_data_paths, get_validation_dataloader, load_class_names
from ml_core.models import create_model
from ml_core.utils import load_config, seed_everything, setup_logger

VALID_VARIANTS = ["clean", "blur", "grayscale", "crop", "occlusion"]


def _evaluate_variant(
    model: nn.Module,
    dataloader,
    device: str,
    variant: str,
    criterion: nn.Module,
    num_classes: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], torch.Tensor]:
    total_examples = 0
    total_correct = 0
    total_top5_correct = 0
    total_loss = 0.0
    prediction_rows: List[Dict[str, Any]] = []
    per_class_examples = torch.zeros(num_classes, dtype=torch.long)
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for images, targets, img_names in tqdm(dataloader, desc=f"Eval {variant}", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            top_k = min(5, num_classes)
            topk_predictions = logits.topk(k=top_k, dim=1).indices
            top5_correct = topk_predictions.eq(targets.unsqueeze(1)).any(dim=1)

            batch_size = targets.size(0)
            total_examples += batch_size
            batch_correct = (predictions == targets)
            total_correct += batch_correct.sum().item()
            total_top5_correct += top5_correct.sum().item()
            total_loss += loss.item() * batch_size

            pred_cpu = predictions.cpu().tolist()
            target_cpu = targets.cpu().tolist()
            conf_cpu = confidences.cpu().tolist()

            for true_label, pred_label, is_correct in zip(target_cpu, pred_cpu, batch_correct.cpu().tolist()):
                per_class_examples[true_label] += 1
                if is_correct:
                    per_class_correct[true_label] += 1
                confusion_matrix[true_label, pred_label] += 1

            for img_name, true_label, pred_label, confidence in zip(
                img_names,
                target_cpu,
                pred_cpu,
                conf_cpu,
            ):
                prediction_rows.append(
                    {
                        "variant": variant,
                        "img_name": img_name,
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                        "correct": int(pred_label == true_label),
                        "confidence": float(confidence),
                    }
                )

    metrics = {
        "variant": variant,
        "num_examples": total_examples,
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "top5_accuracy": total_top5_correct / max(total_examples, 1),
    }

    per_class_rows: List[Dict[str, Any]] = []
    for class_idx in range(num_classes):
        class_examples = int(per_class_examples[class_idx].item())
        class_correct = int(per_class_correct[class_idx].item())
        class_accuracy = class_correct / class_examples if class_examples > 0 else 0.0
        per_class_rows.append(
            {
                "class_idx": class_idx,
                "num_examples": class_examples,
                "num_correct": class_correct,
                "accuracy": class_accuracy,
            }
        )

    return metrics, prediction_rows, per_class_rows, confusion_matrix


def main(args) -> None:
    logger = setup_logger("Evaluator")

    # 3.2 Load config
    config = load_config(args.config)
    # 3.3 Set seed
    seed_everything(config.get("seed", 42))

    # 3.4 Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 3.5 Build model
    model = create_model(config["model"])
    # 3.6 Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 3.7 Build validation loader
    val_loader = get_validation_dataloader(
        config,
        variant=args.eval_variant,
        return_img_names=True,
    )

    # 3.8 Build criterion
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / "evaluation" / args.eval_variant
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(get_data_paths().class_list)

    metrics, prediction_rows, per_class_rows, confusion_matrix = _evaluate_variant(
        model=model,
        dataloader=val_loader,
        device=device,
        variant=args.eval_variant,
        criterion=criterion,
        num_classes=int(config["model"]["num_classes"]),
    )

    for row in prediction_rows:
        true_idx = row["true_label"]
        pred_idx = row["pred_label"]
        row["true_class"] = class_names[true_idx] if 0 <= true_idx < len(class_names) else ""
        row["pred_class"] = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else ""

    prediction_path = output_dir / "predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False)

    summary_path = output_dir / "metrics_summary.csv"
    summary_df = pd.DataFrame([metrics])
    summary_df.to_csv(summary_path, index=False)

    for row in per_class_rows:
        class_idx = row["class_idx"]
        row["class_name"] = class_names[class_idx] if 0 <= class_idx < len(class_names) else ""
    per_class_path = output_dir / "per_class_metrics.csv"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

    confusion_path = output_dir / "confusion_matrix.csv"
    pd.DataFrame(confusion_matrix.numpy()).to_csv(confusion_path, index=False, header=False)

    summary_yaml_path = output_dir / "summary.yaml"
    summary_payload = {
        "eval_variant": args.eval_variant,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(args.config),
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "top5_accuracy": float(metrics["top5_accuracy"]),
        "num_examples": int(metrics["num_examples"]),
    }
    with open(summary_yaml_path, "w") as f:
        yaml.safe_dump(summary_payload, f, sort_keys=False)

    logger.info(
        "Eval finished | device=%s | checkpoint=%s | variant=%s | loss=%.4f | acc=%.4f | top5_acc=%.4f | output_dir=%s",
        device,
        checkpoint_path,
        args.eval_variant,
        metrics["loss"],
        metrics["accuracy"],
        metrics["top5_accuracy"],
        output_dir,
    )
    logger.info("Saved predictions to %s", prediction_path)
    logger.info("Saved summary metrics to %s", summary_path)
    logger.info("Saved per-class metrics to %s", per_class_path)
    logger.info("Saved confusion matrix to %s", confusion_path)
    logger.info("Saved summary yaml to %s", summary_yaml_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on deterministic validation variants"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--eval-variant",
        type=str,
        required=True,
        choices=VALID_VARIANTS,
        help="Evaluation variant",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation outputs",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
