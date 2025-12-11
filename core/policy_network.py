# core/policy_network.py
"""
Red neuronal para la política del agente de Magic Chess.

Entrada: vector de estado (floats/ints normalizados).
Salida: logits sobre un conjunto discreto de acciones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTIONS = [
    "noop",
    "level_up",
    "reroll",
    "buy_unit",
    "sell_unit",
]


def get_action_index_map():
    return {name: i for i, name in enumerate(ACTIONS)}


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_actions: int | None = None,
    ):
        super().__init__()
        num_actions = num_actions or len(ACTIONS)

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        return logits
