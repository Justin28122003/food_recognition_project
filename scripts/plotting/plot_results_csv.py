import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()

def load_data(file_path: Path) -> pd.DataFrame:
    # TODO: Load CSV into Pandas DataFrame
    pass

def setup_style():
    # TODO: Set seaborn theme
    pass

def plot_metrics(df: pd.DataFrame, output_path: Optional[Path]):
    """
    Generate and save plots for Loss, Accuracy, and F1.
    """
    if df is None: return

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # TODO: Plot Train/Val Loss
    
    # TODO: Plot Train/Val Accuracy
    
    # TODO: Plot Learning Rate
    
    plt.tight_layout()
    plt.show() # or save to output_path

def main():
    args = parse_args()
    setup_style()
    df = load_data(args.input_csv)
    plot_metrics(df, args.output_dir)

if __name__ == "__main__":
    main()
