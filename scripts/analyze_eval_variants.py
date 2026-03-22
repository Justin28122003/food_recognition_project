from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

VARIANT_ORDER = ["clean", "blur", "lowres", "grayscale", "crop", "occlusion"]
DROP_VARIANTS = ["blur", "lowres", "grayscale", "crop", "occlusion"]
DISPLAY_LABELS = {
    "clean": "Clean",
    "blur": "Blur",
    "lowres": "Low-res",
    "grayscale": "Grayscale",
    "crop": "Crop",
    "occlusion": "Occlusion",
}
ROBUST_CLEAN_ACC_THRESHOLD = 0.30
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed evaluation runs across all variants and create CSVs/plots."
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default="experiments/results/analysis_outputs/evaluation",
        help="Root folder containing one subfolder per evaluation variant",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/analysis_outputs/evaluation_analysis",
        help="Directory where combined analysis artifacts will be written",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        },
    )


def _annotate_vertical_bars(ax: plt.Axes, fmt: str = "{:.2f}", pad_frac: float = 0.01) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * pad_frac
    for patch in ax.patches:
        height = patch.get_height()
        x = patch.get_x() + patch.get_width() / 2
        ax.text(x, height + offset, fmt.format(height), ha="center", va="bottom", fontsize=10)


def _annotate_horizontal_bars(ax: plt.Axes, fmt: str = "{:.2f}", pad_frac: float = 0.01) -> None:
    xmin, xmax = ax.get_xlim()
    offset = (xmax - xmin) * pad_frac
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(width + offset, y, fmt.format(width), ha="left", va="center", fontsize=10)


def _format_class_display_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("_", " ", regex=False)


def _read_yaml(path: Path) -> Dict:
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} did not contain a mapping/dict.")
    return data


def load_overall_metrics(eval_root: Path) -> pd.DataFrame:
    records: List[Dict] = []

    for variant in VARIANT_ORDER:
        summary_path = eval_root / variant / "summary.yaml"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file for variant '{variant}': {summary_path}")

        summary = _read_yaml(summary_path)
        records.append(
            {
                "eval_variant": summary.get("eval_variant", variant),
                "loss": summary["loss"],
                "accuracy": summary["accuracy"],
                "top5_accuracy": summary["top5_accuracy"],
                "num_examples": summary["num_examples"],
            }
        )

    overall_df = pd.DataFrame(records)
    overall_df["eval_variant"] = pd.Categorical(
        overall_df["eval_variant"],
        categories=VARIANT_ORDER,
        ordered=True,
    )
    overall_df = overall_df.sort_values("eval_variant").reset_index(drop=True)

    clean_row = overall_df[overall_df["eval_variant"] == "clean"]
    if clean_row.empty:
        raise ValueError("No clean variant found in summary files; cannot compute drop-vs-clean metrics.")

    clean_acc = float(clean_row.iloc[0]["accuracy"])
    clean_top5 = float(clean_row.iloc[0]["top5_accuracy"])

    overall_df["top1_drop_vs_clean"] = clean_acc - overall_df["accuracy"]
    overall_df["top5_drop_vs_clean"] = clean_top5 - overall_df["top5_accuracy"]

    if clean_acc == 0:
        overall_df["robustness_ratio"] = float("nan")
    else:
        overall_df["robustness_ratio"] = overall_df["accuracy"] / clean_acc

    clean_mask = overall_df["eval_variant"] == "clean"
    overall_df.loc[clean_mask, "top1_drop_vs_clean"] = 0.0
    overall_df.loc[clean_mask, "top5_drop_vs_clean"] = 0.0
    overall_df.loc[clean_mask, "robustness_ratio"] = 1.0

    return overall_df


def _load_single_per_class(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing per-class metrics file: {path}")

    df = pd.read_csv(path)

    if "class_index" not in df.columns and "class_idx" in df.columns:
        df = df.rename(columns={"class_idx": "class_index"})

    required_cols = {"class_index", "class_name", "accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df = df[["class_index", "class_name", "accuracy"]].copy()
    df["class_index"] = df["class_index"].astype(int)
    return df


def build_per_class_tables(eval_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc_series_list = []
    class_name_series_list = []

    for variant in VARIANT_ORDER:
        per_class_path = eval_root / variant / "per_class_metrics.csv"
        per_class_df = _load_single_per_class(per_class_path)

        acc_series = per_class_df.set_index("class_index")["accuracy"].rename(f"{variant}_acc")
        class_series = per_class_df.set_index("class_index")["class_name"].rename(variant)

        acc_series_list.append(acc_series)
        class_name_series_list.append(class_series)

    per_class_wide = pd.concat(acc_series_list, axis=1).reset_index()

    class_name_matrix = pd.concat(class_name_series_list, axis=1)
    class_name_filled = class_name_matrix.bfill(axis=1).ffill(axis=1).iloc[:, 0]
    per_class_wide["class_name"] = per_class_wide["class_index"].map(class_name_filled)

    per_class_wide = per_class_wide[
        [
            "class_index",
            "class_name",
            "clean_acc",
            "blur_acc",
            "lowres_acc",
            "grayscale_acc",
            "crop_acc",
            "occlusion_acc",
        ]
    ].sort_values("class_index").reset_index(drop=True)

    drop_df = per_class_wide.copy()
    drop_df["blur_drop"] = drop_df["clean_acc"] - drop_df["blur_acc"]
    drop_df["lowres_drop"] = drop_df["clean_acc"] - drop_df["lowres_acc"]
    drop_df["grayscale_drop"] = drop_df["clean_acc"] - drop_df["grayscale_acc"]
    drop_df["crop_drop"] = drop_df["clean_acc"] - drop_df["crop_acc"]
    drop_df["occlusion_drop"] = drop_df["clean_acc"] - drop_df["occlusion_acc"]

    drop_df["local_detail_score"] = drop_df["blur_drop"] + drop_df["lowres_drop"]
    drop_df["structure_score"] = drop_df["crop_drop"] + drop_df["occlusion_drop"]
    drop_df["color_score"] = drop_df["grayscale_drop"]

    return per_class_wide, drop_df


def plot_overall_top1_accuracy(overall_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = overall_df.copy()
    plot_df["variant_code"] = plot_df["eval_variant"].astype(str)
    plot_df["variant_label"] = plot_df["variant_code"].map(DISPLAY_LABELS)

    plt.figure(figsize=(8.5, 5.2))
    ax = sns.barplot(data=plot_df, x="variant_label", y="accuracy")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Accuracy by Evaluation Variant")
    _annotate_vertical_bars(ax)
    plt.tight_layout()
    plt.savefig(output_dir / "overall_top1_accuracy.png", dpi=250)
    plt.close()


def plot_overall_top1_drop(overall_df: pd.DataFrame, output_dir: Path) -> None:
    drop_df = overall_df[overall_df["eval_variant"].isin(DROP_VARIANTS)].copy()
    drop_df["eval_variant"] = pd.Categorical(
        drop_df["eval_variant"].astype(str),
        categories=DROP_VARIANTS,
        ordered=True,
    )
    drop_df = drop_df.sort_values("eval_variant")
    drop_df["variant_code"] = drop_df["eval_variant"].astype(str)
    drop_df["variant_label"] = drop_df["variant_code"].map(DISPLAY_LABELS)

    plt.figure(figsize=(8.5, 5.2))
    ax = sns.barplot(data=drop_df, x="variant_label", y="top1_drop_vs_clean")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Top-1 accuracy drop")
    ax.set_title("Accuracy Drop vs Clean")
    _annotate_vertical_bars(ax)
    plt.tight_layout()
    plt.savefig(output_dir / "overall_top1_drop.png", dpi=250)
    plt.close()


def plot_per_class_drop_heatmap(drop_df: pd.DataFrame, output_dir: Path) -> None:
    heat_cols = ["blur_drop", "lowres_drop", "grayscale_drop", "crop_drop", "occlusion_drop"]
    ordered = drop_df.sort_values("local_detail_score", ascending=False).reset_index(drop=True)

    matrix = ordered[heat_cols].to_numpy()
    row_labels = _format_class_display_label(
        ordered["class_name"].fillna(ordered["class_index"].astype(str))
    ).tolist()
    col_labels = [DISPLAY_LABELS[col.replace("_drop", "")] for col in heat_cols]

    plt.figure(figsize=(11.5, 18))
    ax = sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Accuracy drop vs clean"},
    )
    ax.tick_params(axis="y", labelsize=7)

    ax.set_xlabel("Variant")
    ax.set_ylabel("Class")
    ax.set_title("Per-Class Accuracy Drop by Evaluation Variant")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_drop_heatmap.png", dpi=250)
    plt.close()


def _plot_sensitive_vs_robust(
    drop_df: pd.DataFrame,
    score_col: str,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    sensitive = drop_df.nlargest(TOP_K, score_col).sort_values(score_col, ascending=False).copy()

    robust_pool = drop_df[
        (drop_df["clean_acc"] >= ROBUST_CLEAN_ACC_THRESHOLD) & (drop_df[score_col] >= 0)
    ].copy()
    robust = robust_pool.nsmallest(TOP_K, score_col).sort_values(score_col, ascending=True).copy()

    sensitive["group"] = "Most sensitive"
    robust["group"] = "Most robust"

    plot_df = pd.concat([sensitive, robust], axis=0, ignore_index=True)
    plot_df["group"] = pd.Categorical(
        plot_df["group"],
        categories=["Most sensitive", "Most robust"],
        ordered=True,
    )
    plot_df["label"] = _format_class_display_label(
        plot_df["class_name"].fillna(plot_df["class_index"].astype(str))
    )

    y_order = plot_df["label"].tolist()

    plt.figure(figsize=(10.2, 7.2))
    ax = sns.barplot(
        data=plot_df,
        y="label",
        x=score_col,
        hue="group",
        orient="h",
        dodge=False,
        order=y_order,
    )
    ax.invert_yaxis()

    separator_y = len(sensitive) - 0.5
    ax.axhline(separator_y, color="gray", linewidth=1.0, alpha=0.7)

    xmin, xmax = ax.get_xlim()
    x_text = xmin + 0.02 * (xmax - xmin)
    if len(sensitive) > 0:
        y_sensitive = (len(sensitive) - 1) / 2
        ax.text(
            x_text,
            y_sensitive,
            "Most sensitive",
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
        )
    if len(robust) > 0:
        y_robust = len(sensitive) + (len(robust) - 1) / 2
        ax.text(
            x_text,
            y_robust,
            "Most robust",
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.0},
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Class")
    ax.set_title(title)
    ax.legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    _annotate_horizontal_bars(ax)
    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_local_detail_sensitive_vs_robust(drop_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_sensitive_vs_robust(
        drop_df=drop_df,
        score_col="local_detail_score",
        title="Local-Detail-Sensitive vs Robust Classes",
        x_label="Local-detail sensitivity score",
        output_path=output_dir / "local_detail_sensitive_vs_robust.png",
    )


def plot_structure_sensitive_vs_robust(drop_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_sensitive_vs_robust(
        drop_df=drop_df,
        score_col="structure_score",
        title="Structure-Sensitive vs Robust Classes",
        x_label="Structure sensitivity score",
        output_path=output_dir / "structure_sensitive_vs_robust.png",
    )


def plot_color_sensitive_vs_robust(drop_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_sensitive_vs_robust(
        drop_df=drop_df,
        score_col="color_score",
        title="Color-Sensitive vs Robust Classes",
        x_label="Color sensitivity score",
        output_path=output_dir / "color_sensitive_vs_robust.png",
    )


def main() -> None:
    args = parse_args()
    configure_plot_style()

    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_df = load_overall_metrics(eval_root=eval_root)
    overall_df.to_csv(output_dir / "overall_comparison.csv", index=False)

    per_class_wide, drop_df = build_per_class_tables(eval_root=eval_root)
    per_class_wide.to_csv(output_dir / "per_class_comparison.csv", index=False)
    drop_df.to_csv(output_dir / "per_class_drop_summary.csv", index=False)

    plot_overall_top1_accuracy(overall_df=overall_df, output_dir=output_dir)
    plot_overall_top1_drop(overall_df=overall_df, output_dir=output_dir)
    plot_per_class_drop_heatmap(drop_df=drop_df, output_dir=output_dir)
    plot_local_detail_sensitive_vs_robust(drop_df=drop_df, output_dir=output_dir)
    plot_structure_sensitive_vs_robust(drop_df=drop_df, output_dir=output_dir)
    plot_color_sensitive_vs_robust(drop_df=drop_df, output_dir=output_dir)

    print(f"Saved analysis CSVs and plots to: {output_dir}")


if __name__ == "__main__":
    main()
