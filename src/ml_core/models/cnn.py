from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn
from torchvision import models


@dataclass(frozen=True)
class BackboneSpec:
    factory: Any
    weights: Any
    classifier_attr: str
    in_features: int


def _get_backbone_spec(backbone_name: str) -> BackboneSpec:
    specs = {
        "resnet18": BackboneSpec(
            factory=models.resnet18,
            weights=models.ResNet18_Weights.DEFAULT,
            classifier_attr="fc",
            in_features=512,
        ),
        "resnet34": BackboneSpec(
            factory=models.resnet34,
            weights=models.ResNet34_Weights.DEFAULT,
            classifier_attr="fc",
            in_features=512,
        ),
        "densenet121": BackboneSpec(
            factory=models.densenet121,
            weights=models.DenseNet121_Weights.DEFAULT,
            classifier_attr="classifier",
            in_features=1024,
        ),
        "efficientnet_b0": BackboneSpec(
            factory=models.efficientnet_b0,
            weights=models.EfficientNet_B0_Weights.DEFAULT,
            classifier_attr="classifier",
            in_features=1280,
        ),
    }

    if backbone_name not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Available: {available}")

    return specs[backbone_name]


class TorchvisionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        spec = _get_backbone_spec(backbone_name)
        weights = spec.weights if pretrained else None
        self.model = spec.factory(weights=weights)

        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        classifier = _build_classifier(
            in_features=spec.in_features,
            num_classes=num_classes,
            dropout=dropout,
        )
        setattr(self.model, spec.classifier_attr, classifier)

    def forward(self, inputs):
        return self.model(inputs)


def _build_classifier(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    if dropout > 0:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    return nn.Linear(in_features, num_classes)


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    return TorchvisionClassifier(
        backbone_name=model_config.get("name", "resnet18"),
        num_classes=model_config["num_classes"],
        pretrained=model_config.get("pretrained", False),
        dropout=model_config.get("dropout", 0.2),
        freeze_backbone=model_config.get("freeze_backbone", False),
    )