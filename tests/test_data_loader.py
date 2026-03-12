import h5py
import numpy as np
import pytest
import torch
from ml_core.data.loader import get_dataloaders
from ml_core.data.pcam import PCAMDataset


class TestPCAMPipeline:
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Creates dummy H5 files with names expected by loader.py"""
        for split in ["train", "valid"]:
            # Names must match loader.py ground truth
            x_path = tmp_path / f"camelyonpatch_level_2_split_{split}_x.h5"
            y_path = tmp_path / f"camelyonpatch_level_2_split_{split}_y.h5"

            with h5py.File(x_path, "w") as f_x, h5py.File(y_path, "w") as f_y:
                # 100 samples: 97 normal, 1 outlier (1e5), 1 black (0), 1 white (255)
                x_ds = f_x.create_dataset("x", (100, 96, 96, 3), dtype="float32")
                y_ds = f_y.create_dataset("y", (100, 1, 1, 1), dtype="int64")

                data = np.full((100, 96, 96, 3), 128.0, dtype="float32")
                data[0, 0, 0, :] = 1e5  # Numerical outlier
                data[1, :, :, :] = 0.0  # Visual outlier (Blackout)
                data[2, :, :, :] = 255.0  # Visual outlier (Washout)

                # 80/20 class imbalance for WeightedSampler test
                labels = np.zeros((100, 1, 1, 1), dtype="int64")
                labels[80:] = 1

                x_ds[:] = data
                y_ds[:] = labels
        return tmp_path

    def test_numerical_stability(self, mock_data_dir):
        """Checks if 1e5 values are clipped to 255 before becoming uint8."""
        x_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_x.h5")
        y_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_y.h5")
        ds = PCAMDataset(x_p, y_p, filter_data=False)
        img, _ = ds[0]
        assert (
            img.max() <= 255
        ), "Image values > 255 found. Did you forget to clip before uint8 cast?"

    def test_heuristic_filtering(self, mock_data_dir):
        """Checks if mean-based filtering drops the black/white outlier samples."""
        x_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_x.h5")
        y_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_y.h5")
        ds = PCAMDataset(x_p, y_p, filter_data=True)
        # Expected 98 because index 1 (mean 0) and 2 (mean 255) should be dropped
        assert (
            len(ds.indices) == 98
        ), f"Filtering failed. Expected 98 samples, got {len(ds.indices)}"

    def test_dataloader_output_logic(self, mock_data_dir):
        """Verifies shapes, types, and label squeezing."""
        config = {
            "data": {"data_path": str(mock_data_dir), "batch_size": 4, "num_workers": 0}
        }
        train_loader, _ = get_dataloaders(config)
        images, labels = next(iter(train_loader))

        assert images.shape == (4, 3, 96, 96)
        assert labels.dtype == torch.long
        assert labels.dim() == 1, "Labels should be squeezed to 1D (Batch size,)"

    def test_weighted_sampling(self, mock_data_dir):
        """Verifies WeightedRandomSampler balances the 80/20 split."""
        config = {
            "data": {
                "data_path": str(mock_data_dir),
                "batch_size": 40,
                "num_workers": 0,
            }
        }
        train_loader, _ = get_dataloaders(config)
        _, labels = next(iter(train_loader))

        positives = (labels == 1).sum().item()
        # In a batch of 40 without sampling, we'd expect ~8 positives.
        # With balancing, we expect closer to 20.
        assert (
            positives > 12
        ), f"WeightedSampler might not be working. Only {positives}/40 are class 1."
