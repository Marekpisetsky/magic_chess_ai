# core/hud_local_reader.py
"""
Lector de HUD que usa el modelo local entrenado (hud_model.pt).
Se integra fácilmente con state.py y con tu bucle de juego.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
from torchvision import transforms

from models.hud_model import HUDModel


class HUDLocalReader:
    def __init__(self, weights_path: str = "hud_model.pt", device: str | None = None):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de pesos {self.weights_path}. "
                "Entrena el modelo con train_hud_model.py primero."
            )

        ckpt = torch.load(self.weights_path, map_location="cpu")
        self.round_vocab = ckpt["round_vocab"]
        self.inv_round_vocab = {v: k for k, v in self.round_vocab.items()}

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HUDModel(
            num_round_classes=len(self.round_vocab),
            num_level_classes=10,
            num_gold_classes=101,
            num_hp_classes=101,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _predict_tensor(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        x = img_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)

        def _argmax(logits: torch.Tensor) -> int:
            return int(logits.argmax(dim=1).item())

        round_idx = _argmax(outputs["round"])
        level_idx = _argmax(outputs["level"])
        gold_idx = _argmax(outputs["gold"])
        hp_idx = _argmax(outputs["hp_self"])

        round_str = self.inv_round_vocab.get(round_idx, None)

        return {
            "round": round_str,
            "level": level_idx,
            "gold": gold_idx,
            "hp_self": hp_idx,
        }

    def predict_from_image_path(self, image_path: str) -> Dict[str, Any]:
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img)
        return self._predict_tensor(x)
