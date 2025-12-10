from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.state import ScreenState, PlayerStatus


@dataclass
class Advice:
    """Consejo de alto nivel para la ronda actual."""
    summary: str
    actions: List[str]


def _parse_stage(round_label: Optional[str]) -> str:
    """
    Devuelve 'early', 'mid', 'late' según la ronda.
    Espera formatos tipo 'I-3', 'II-3', 'III-2', etc.
    """
    if not round_label or "-" not in round_label:
        return "unknown"

    stage_part = round_label.split("-")[0].strip().upper()

    if stage_part in ("I", "I-1", "1"):
        return "early"
    if stage_part in ("II", "2"):
        return "mid"
    # III, IV, V... -> late
    return "late"


def _relative_hp_position(you: PlayerStatus, others: List[PlayerStatus]) -> str:
    """
    Devuelve una etiqueta simple de tu situación de vida:
    'high', 'average', 'low'.
    """
    if you.hp is None:
        return "unknown"

    hp_values = [p.hp for p in others if p.hp is not None]
    if not hp_values:
        return "unknown"

    max_hp = max(hp_values + [you.hp])
    min_hp = min(hp_values + [you.hp])
    avg_hp = sum(hp_values + [you.hp]) / (len(hp_values) + 1)

    if you.hp >= avg_hp + 10:
        return "high"
    if you.hp <= avg_hp - 10 or you.hp == min_hp:
        return "low"
    return "average"


def make_advice(state: ScreenState) -> Advice:
    """
    Genera un consejo simple basado en:
    - tu vida
    - el oro
    - la ronda (early/mid/late)
    - comparación de vida con el lobby
    """
    you = state.you
    others = state.others

    stage = _parse_stage(state.round_label)
    hp_pos = _relative_hp_position(you, others)

    lines: List[str] = []
    actions: List[str] = []

    # --- economía básica ---
    gold = you.gold or 0

    if gold >= 50:
        lines.append("Tienes economía máxima (50+ oro).")
        actions.append("Mantén al menos 50 oro para el máximo interés.")
    elif gold >= 30:
        lines.append("Tienes una economía decente (30–49 oro).")
        actions.append("Prioriza ahorrar hasta 50 oro salvo que estés muy débil.")
    elif gold >= 10:
        lines.append("Tu economía es media-baja (10–29 oro).")
        actions.append("Intenta no rolear demasiado y subir poco a poco hacia 50 oro.")
    else:
        lines.append("Tienes muy poco oro (<10).")
        actions.append("Evita gastar en rerolls innecesarios y estabiliza primero tu tablero.")

    # --- vida / presión ---
    if you.hp is not None:
        if you.hp >= 70:
            lines.append(f"Tu vida es alta ({you.hp}). Puedes permitirte jugar más greed/economía.")
            actions.append("No entres en pánico; puedes aceptar algunas derrotas para escalar economía.")
        elif you.hp >= 40:
            lines.append(f"Tu vida es media ({you.hp}). Debes equilibrar economía y fuerza de tablero.")
            actions.append("Asegúrate de no perder demasiada vida consecutiva; considera rolear un poco para estabilizar.")
        else:
            lines.append(f"Tu vida es baja ({you.hp}). Necesitas estabilizar YA.")
            actions.append("Prioriza comprar mejoras de tablero y subir unidades clave. La economía es secundaria ahora.")

    # --- posición relativa en lobby ---
    if hp_pos == "high":
        lines.append("Estás fuerte en comparación con el lobby (vida alta relativa).")
        actions.append("Puedes jugar greedy: subir nivel antes, buscar sinergias fuertes a largo plazo.")
    elif hp_pos == "low":
        lines.append("Estás débil en comparación con el lobby (vida baja relativa).")
        actions.append("Juega agresivo: fortalece tu tablero, no esperes demasiado para rolear.")
    elif hp_pos == "average":
        lines.append("Tu vida es similar a la media del lobby.")
        actions.append("Ajusta según la tienda: si encuentras mejoras fuertes, invierte; si no, ahorra.")

    # --- etapa de la partida ---
    if stage == "early":
        lines.append("Etapa temprana de la partida (early game).")
        actions.append("Concéntrate en formar sinergias básicas y no sacrificar demasiada vida por economía.")
    elif stage == "mid":
        lines.append("Etapa media de la partida (mid game).")
        actions.append("Debes decidir tu composición principal y estabilizar antes de entrar en late.")
    elif stage == "late":
        lines.append("Etapa tardía de la partida (late game).")
        actions.append("Optimiza tu comp principal, sube unidades a 2/3 estrellas y no guardes oro sin usar.")

    # Resumen
    summary = " ".join(lines) if lines else "Estado leído, pero sin suficiente información para un consejo sólido."
    return Advice(summary=summary, actions=actions)
