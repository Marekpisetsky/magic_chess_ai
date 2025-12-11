# core/rule_based_policy.py
"""
Política basada en reglas simples para Magic Chess.

Sirve como "teacher" inicial para:
  - generar episodios razonables,
  - entrenar luego la política neuronal (imitation learning).

Acciones soportadas (coherentes con PolicyNetwork):
  - "noop"
  - "level_up"
  - "reroll"
  - "buy_unit"
  - "sell_unit"
"""

from __future__ import annotations

from typing import List

from core.policy_network import ACTIONS
from core.state import GameState


class RuleBasedPolicy:
    """
    Política por reglas muy simples. No es buena, pero da estructura.
    Mejórala poco a poco con tu conocimiento del juego.
    """

    def __init__(self) -> None:
        self.actions = ACTIONS

    def choose_action(self, state: GameState, valid_actions: List[str] | None = None) -> str:
        """
        Decide una acción a partir del GameState.

        valid_actions: si lo pasas, restringe la acción a ese subconjunto.
        """
        if valid_actions is None:
            valid_actions = self.actions

        # Acceso rápido (ajustado a GameState actual)
        gold = getattr(state, "oro", getattr(state, "gold", 0))
        level = getattr(state, "nivel_tablero", getattr(state, "level", 1))
        hp = getattr(state, "vida", getattr(state, "hp_self", 100))
        round_label = getattr(state, "round_label", "") or "0-0"
        r1, r2 = state._round_to_nums() if hasattr(state, "_round_to_nums") else (0, 0)

        # Ejemplo de reglas simplonas:

        # 1) Si estás muy bajo de vida, prioriza hacer algo agresivo (subir nivel)
        if hp <= 15 and "level_up" in valid_actions and gold >= 4:
            return "level_up"

        # 2) Primeras rondas: compra unidades, no guardes tanto oro
        if r1 == 1:
            if gold >= 2 and "buy_unit" in valid_actions:
                return "buy_unit"

        # 3) Mitad de partida: subir nivel en rondas clave
        # (esto es muy aproximado, puedes refinarlo)
        if r1 in (2, 3, 4) and "level_up" in valid_actions and gold >= 4 and level < 7:
            return "level_up"

        # 4) Si tienes mucho oro (50+), gasta un poco en reroll
        if gold >= 50 and "reroll" in valid_actions:
            return "reroll"

        # 5) Si tienes oro medio (20-40), compra algo de vez en cuando
        if 20 <= gold < 50 and "buy_unit" in valid_actions:
            return "buy_unit"

        # 6) Si estás lleno de unidades malas, puedes decidir vender (placeholder)
        # Aquí necesitaríamos más info del tablero, pero dejamos un stub
        # if some_condition and "sell_unit" in valid_actions:
        #     return "sell_unit"

        # 7) Fallback: no hacer nada
        if "noop" in valid_actions:
            return "noop"

        # Si llega aquí, devolvemos la primera acción válida
        return valid_actions[0]
