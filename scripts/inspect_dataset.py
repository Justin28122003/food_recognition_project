"""Quick sanity-check / visualisation script for the food-recognition dataset."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from ml_core.data.loader import (
    get_data_paths,
    load_class_names,
    NUM_CLASSES,
)


def print_summary() -> None:
    paths = get_data_paths()
    print("\n=== DATA PATHS ===")
    for field in paths.__dataclass_fields__:
        p = getattr(paths, field)
        status = "OK" if p.exists() else "MISSING"
        print(f"  {field:20s}: {p}  [{status}]")

    if paths.class_list.exists():
        names = load_class_names(paths.class_list)
        print(f"\nClass names loaded: {len(names)} (expected {NUM_CLASSES})")

    if paths.train_labels_csv.exists():
        df = pd.read_csv(paths.train_labels_csv)
        print(f"\n=== TRAIN LABELS ({len(df)} rows) ===")
        print(f"  Label range    : {df['label'].min()} – {df['label'].max()}")
        print(f"  Unique labels  : {df['label'].nunique()}")
        print(f"  Duplicated names: {df['img_name'].duplicated().sum()}")

    for name, d in [("Train", paths.train_dir), ("Test", paths.test_dir)]:
        if d.exists():
            n = sum(1 for _ in d.glob("*.jpg"))
            print(f"  {name} images : {n}")


def plot_class_distribution() -> None:
    paths = get_data_paths()
    df = pd.read_csv(paths.train_labels_csv)

    class_counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(16, 5))
    class_counts.plot(kind="bar")
    plt.title("Training examples per class")
    plt.xlabel("Class label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def show_random_samples(num_samples: int = 9, random_state: int = 42) -> None:
    paths = get_data_paths()
    df = pd.read_csv(paths.train_labels_csv)
    sample_rows = df.sample(num_samples, random_state=random_state)

    ncols = 3
    nrows = (num_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (_, row) in zip(axes, sample_rows.iterrows()):
        image_path = paths.train_dir / row["img_name"]

        with Image.open(image_path).convert("RGB") as img:
            ax.imshow(img)

        ax.set_title(f"{row['img_name']}\nlabel={row['label']}")
        ax.axis("off")

    for ax in axes[len(sample_rows):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def print_image_size_stats(sample_size: int = 500) -> None:
    paths = get_data_paths()
    df = pd.read_csv(paths.train_labels_csv)
    sample = df.sample(min(sample_size, len(df)), random_state=42)

    widths, heights, modes = [], [], []
    for _, row in sample.iterrows():
        with Image.open(paths.train_dir / row["img_name"]) as img:
            widths.append(img.width)
            heights.append(img.height)
            modes.append(img.mode)

    size_df = pd.DataFrame({"width": widths, "height": heights, "mode": modes})

    print("\n=== IMAGE SIZE STATS ===")
    print(size_df[["width", "height"]].describe())
    print("\nImage modes:")
    print(size_df["mode"].value_counts())


def main() -> None:
    print_summary()
    plot_class_distribution()
    show_random_samples(num_samples=9)
    print_image_size_stats(sample_size=500)


if __name__ == "__main__":
    main()