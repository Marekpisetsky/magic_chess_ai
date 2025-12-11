# core/reward.py
"""
Reward shaping para el agente de Magic Chess.

La recompensa combina:
  - resultado de la partida (win/lose),
  - cambios de vida,
  - progreso de ronda,
  - (opcional) estado económico.
"""

from __future__ import annotations

from typing import Optional

from core.state import GameState


def compute_reward(
    prev_state: GameState,
    new_state: GameState,
    done: bool,
    result: Optional[str] = None,
) -> float:
    """
    Calcula una recompensa escalar entre dos estados sucesivos.

    Parámetros:
      prev_state: estado antes de la acción
      new_state: estado después de la acción
      done: si la partida ha terminado
      result: 'win', 'lose' o None (si aún no se sabe)

    Idea:
      - castigo por perder vida
      - recompensa pequeña por avanzar de ronda
      - recompensa grande por ganar
      - castigo grande por perder
    """
    reward = 0.0

    # Cambio de vida (usa 'vida' del GameState; fallback a hp_self si existiera)
    prev_hp = getattr(prev_state, "vida", getattr(prev_state, "hp_self", 100))
    new_hp = getattr(new_state, "vida", getattr(new_state, "hp_self", 100))
    hp_delta = new_hp - prev_hp
    if hp_delta < 0:
        reward += hp_delta * 0.05  # perder 20 de vida -> -1.0

    # Avance de ronda (ej. 1-1 -> 1-2)
    prev_r1, prev_r2 = prev_state._round_to_nums()
    new_r1, new_r2 = new_state._round_to_nums()
    if (new_r1, new_r2) > (prev_r1, prev_r2):
        reward += 0.1  # pequeño premio por avanzar

    # Resultado final
    if done:
        if result == "win":
            reward += 5.0
        elif result == "lose":
            reward -= 5.0

    # TODO: puedes añadir aquí economía avanzada:
    # - premio por tener oro >= X antes de ciertas rondas
    # - castigo por quedar pobre demasiado pronto, etc.

    return reward
