# datasets/hud_dataset.py
"""
Dataset de entrenamiento para el modelo HUD local.
Lee data/labels.jsonl y las imágenes de data/raw_frames/.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HUDDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data",
        labels_file: str = "labels.jsonl",
        round_vocab: Dict[str, int] | None = None,
        transform: Any | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "raw_frames"
        self.labels_path = self.data_dir / labels_file

        if not self.labels_path.exists():
            raise FileNotFoundError(f"No se encontró {self.labels_path}")

        self.records: List[Dict[str, Any]] = []
        with self.labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        # Construir vocabulario de rondas si no viene de fuera
        if round_vocab is None:
            rounds = sorted(
                {r["round"] for r in self.records if r.get("round") is not None}
            )
            self.round_vocab: Dict[str, int] = {r: i for i, r in enumerate(rounds)}
        else:
            self.round_vocab = round_vocab

        # Transformaciones por defecto
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Índice utilizado por CrossEntropy para ignorar targets
        self.ignore_index = -100

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rec = self.records[idx]
        img_field = rec["image"]
        p = Path(img_field)
        if p.is_absolute():
            img_path = p
        else:
            norm = str(p).replace("\\", "/")
            if norm.startswith("data/"):
                img_path = Path(norm)
            else:
                img_path = self.frames_dir / p

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        # ROUND
        r = rec.get("round")
        if r is None or r not in self.round_vocab:
            round_idx = self.ignore_index
        else:
            round_idx = self.round_vocab[r]

        # LEVEL, GOLD, HP
        def _num_or_ignore(value: Any) -> int:
            if value is None:
                return self.ignore_index
            try:
                v = int(value)
            except (TypeError, ValueError):
                return self.ignore_index
            return max(0, v)

        level = _num_or_ignore(rec.get("level"))
        gold = _num_or_ignore(rec.get("gold"))
        hp_self = _num_or_ignore(rec.get("hp_self"))

        targets = {
            "round": torch.tensor(round_idx, dtype=torch.long),
            "level": torch.tensor(level, dtype=torch.long),
            "gold": torch.tensor(gold, dtype=torch.long),
            "hp_self": torch.tensor(hp_self, dtype=torch.long),
        }

        return x, targets
