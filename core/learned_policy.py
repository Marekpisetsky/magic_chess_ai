# core/learned_policy.py
"""
Envoltorio para usar la PolicyNetwork entrenada (policy_model.pt)
dentro del loop de juego, usando los metadatos guardados.
"""

from __future__ import annotations

from typing import List, Any

import torch

from core.policy_network import PolicyNetwork, ACTIONS as DEFAULT_ACTIONS


class LearnedPolicy:
    def __init__(self, model_path: str = "policy_model.pt", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.actions: List[str] = list(DEFAULT_ACTIONS)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> PolicyNetwork:
        ckpt = torch.load(model_path, map_location="cpu")

        # Nueva versión con config
        if "config" in ckpt:
            config = ckpt["config"]
            state_dim = int(config["state_dim"])
            hidden_dim = int(config.get("hidden_dim", 256))
            actions = config.get("actions", list(DEFAULT_ACTIONS))
            # nos aseguramos de que es una lista de strings
            self.actions = [str(a) for a in actions]
            num_actions = len(self.actions)
            state_dict = ckpt["model_state_dict"]
        else:
            # Compatibilidad con checkpoints antiguos
            state_dict = ckpt
            # inferimos state_dim y num_actions desde pesos
            fc1_w = state_dict["fc1.weight"]
            fc_out_w = state_dict["fc_out.weight"]
            state_dim = fc1_w.shape[1]
            num_actions = fc_out_w.shape[0]
            hidden_dim = fc1_w.shape[0]
            self.actions = list(DEFAULT_ACTIONS)[:num_actions]

        model = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def choose_action(self, state_vector: List[float]) -> str:
        state = torch.tensor(
            state_vector, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(state)
            idx = int(logits.argmax(dim=1).item())

        # seguridad por si idx está fuera de rango por algún bug
        if idx < 0 or idx >= len(self.actions):
            idx = 0  # fallback a "noop" o similar

        return self.actions[idx]
