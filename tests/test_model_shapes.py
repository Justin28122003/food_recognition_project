import pytest
import torch
from ml_core.models import MLP


class TestMLPImplementation:
    @pytest.fixture
    def sample_config(self):
        return {"input_shape": [3, 96, 96], "hidden_units": [64, 32], "num_classes": 2}

    def test_forward_pass(self, sample_config):
        """Verifies the model flattens input and outputs correct logit shapes."""
        model = MLP(**sample_config)
        x = torch.randn(8, *sample_config["input_shape"])
        output = model(x)
        assert output.shape == (
            8,
            2,
        ), f"Expected (8, 2), got {output.shape}. Did you flatten?"

    def test_backprop(self, sample_config):
        """Ensures weights update, verifying the computational graph isn't broken."""
        model = MLP(**sample_config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Capture weight state
        param = next(model.parameters())
        initial_val = param.clone()

        # Step
        x = torch.randn(2, *sample_config["input_shape"])
        y = torch.tensor([0, 1], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

        assert not torch.equal(
            initial_val, param
        ), "Weights did not update. Is the graph broken?"
