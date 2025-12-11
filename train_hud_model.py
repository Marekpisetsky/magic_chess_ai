# train_hud_model.py
"""
Entrena el modelo local de HUD usando data/labels.jsonl y data/raw_frames/.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets.hud_dataset import HUDDataset
from models.hud_model import HUDModel


def train(num_epochs: int = 15, batch_size: int = 16, lr: float = 1e-4) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # Dataset completo para construir vocabulario de rondas
    full_ds = HUDDataset()
    round_vocab = full_ds.round_vocab
    num_round_classes = len(round_vocab)

    # Split train/val (80/20)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = HUDModel(
        num_round_classes=num_round_classes,
        num_level_classes=10,
        num_gold_classes=101,
        num_hp_classes=101,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, targets in train_loader:
            x = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            outputs = model(x)

            loss_round = criterion(outputs["round"], targets["round"])
            loss_level = criterion(outputs["level"], targets["level"])
            loss_gold = criterion(outputs["gold"], targets["gold"])
            loss_hp = criterion(outputs["hp_self"], targets["hp_self"])

            loss = loss_round + loss_level + loss_gold + loss_hp
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                outputs = model(x)
                loss_round = criterion(outputs["round"], targets["round"])
                loss_level = criterion(outputs["level"], targets["level"])
                loss_gold = criterion(outputs["gold"], targets["gold"])
                loss_hp = criterion(outputs["hp_self"], targets["hp_self"])
                loss = loss_round + loss_level + loss_gold + loss_hp
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss = {avg_train_loss:.4f} | "
            f"val_loss = {avg_val_loss:.4f}"
        )

    # Guardar modelo + vocabulario de rondas
    ckpt = {
        "model_state_dict": model.state_dict(),
        "round_vocab": round_vocab,
    }
    torch.save(ckpt, "hud_model.pt")
    print("Modelo guardado en hud_model.pt")


if __name__ == "__main__":
    train()
