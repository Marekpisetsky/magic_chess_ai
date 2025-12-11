# train_policy_supervised.py
"""
Entrena una política neuronal a partir de episodios guardados en data/episodes/.

Supervised learning:
  - Input: state_vector
  - Target: acción (índice)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from core.policy_network import PolicyNetwork, get_action_index_map, ACTIONS


EPISODES_DIR = Path("data/episodes")
MODEL_PATH = Path("policy_model.pt")


class ExperienceDataset(Dataset):
    def __init__(self, episodes_dir: Path) -> None:
        self.samples: List[Dict[str, Any]] = []
        action2idx = get_action_index_map()

        state_dim: int | None = None

        for ep_file in sorted(episodes_dir.glob("episode_*.jsonl")):
            with ep_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            # Primera línea = meta
            for line in lines[1:]:
                rec = json.loads(line)
                state = rec.get("state")
                action = rec.get("action")

                if state is None or action is None:
                    continue
                if action not in action2idx:
                    # acción desconocida con respecto a ACTIONS
                    continue

                if state_dim is None:
                    state_dim = len(state)
                else:
                    if len(state) != state_dim:
                        raise ValueError(
                            f"Inconsistencia en state_dim: se esperaba {state_dim} "
                            f"pero se encontró {len(state)} en {ep_file}"
                        )

                self.samples.append(
                    {
                        "state": state,
                        "action_idx": action2idx[action],
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No se encontraron muestras en {episodes_dir}")

        self.state_dim = state_dim or 0
        if self.state_dim <= 0:
            raise RuntimeError("state_dim inferido inválido.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        state = torch.tensor(s["state"], dtype=torch.float32)
        action = torch.tensor(s["action_idx"], dtype=torch.long)
        return state, action


def train(num_epochs: int = 10, batch_size: int = 64, lr: float = 1e-3, hidden_dim: int = 256) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Entrenando policy en dispositivo:", device)

    ds = ExperienceDataset(EPISODES_DIR)
    state_dim = ds.state_dim
    num_actions = len(ACTIONS)

    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = PolicyNetwork(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for state, action_idx in train_loader:
            state = state.to(device)
            action_idx = action_idx.to(device)

            optimizer.zero_grad()
            logits = model(state)
            loss = criterion(logits, action_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # Validación
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for state, action_idx in val_loader:
                state = state.to(device)
                action_idx = action_idx.to(device)

                logits = model(state)
                loss = criterion(logits, action_idx)
                total_val_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += (pred == action_idx).sum().item()
                total += action_idx.size(0)

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        acc = correct / max(1, total)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={acc:.3f}"
        )

    # Guardar modelo + metadatos
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
            "actions": list(ACTIONS),
        },
    }
    torch.save(ckpt, MODEL_PATH)
    print("Modelo de política guardado en", MODEL_PATH)


if __name__ == "__main__":
    train()
