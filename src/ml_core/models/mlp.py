from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        layers = []
        in_features = self.input_dim

        for hidden in hidden_units:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden

        # map to num_classes (e.g., 2 for binary because we are using CrossEntropyLoss)
        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.network(x)
