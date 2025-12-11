# models/hud_model.py
"""
Definición del modelo local de HUD basado en ResNet18.
"""

import torch.nn as nn
from torchvision import models


class HUDModel(nn.Module):
    def __init__(
        self,
        num_round_classes: int,
        num_level_classes: int = 10,   # 0–9
        num_gold_classes: int = 101,   # 0–100
        num_hp_classes: int = 101,     # 0–100
    ) -> None:
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.round_head = nn.Linear(in_features, num_round_classes)
        self.level_head = nn.Linear(in_features, num_level_classes)
        self.gold_head = nn.Linear(in_features, num_gold_classes)
        self.hp_head = nn.Linear(in_features, num_hp_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "round": self.round_head(feat),
            "level": self.level_head(feat),
            "gold": self.gold_head(feat),
            "hp_self": self.hp_head(feat),
        }
