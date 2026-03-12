from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
        filter_data: bool = False,
    ):
        self.x_data = h5py.File(x_path, "r")["x"]
        self.y_data = h5py.File(y_path, "r")["y"]
        self.transform = transform

        # Initialize indices for filtering
        self.indices = np.arange(len(self.x_data))

        if filter_data:
            valid_indices = []
            for i in range(len(self.x_data)):
                # Heuristic: Drop blackouts (0) and washouts (255)
                mean_val = np.mean(self.x_data[i])
                if 0 < mean_val < 255:
                    valid_indices.append(i)
            self.indices = np.array(valid_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        img = self.x_data[real_idx]
        label = self.y_data[real_idx].item()

        # Handle NaNs explicitly before clipping/casting
        # This replaces NaNs with 0.0 (black)
        img = np.nan_to_num(img, nan=0.0)

        # Numerical Stability: Clip before uint8 cast
        img = np.clip(img, 0, 255).astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            # Basic conversion if no transform provided
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, torch.tensor(label, dtype=torch.long)
