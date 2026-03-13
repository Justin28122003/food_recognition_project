import csv
from pathlib import Path
from typing import Any, Dict

import yaml
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)

        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = None
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        row = {"epoch": epoch, **metrics}

        if self.csv_writer is None:
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(row.keys()))
            self.csv_writer.writeheader()

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
        self.writer.close()
