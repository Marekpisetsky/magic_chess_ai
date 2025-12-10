# core/decision.py
from typing import List, Dict, Any
import json

from .state import GameState, ShopHero


class DecisionEngine:
    """
    Versión 0: reglas sencillas pero razonables.
    Más adelante se enchufa aprendizaje real.
    """

    def recommend_actions(self, state: GameState) -> List[Dict[str, Any]]:
        recs = []

        # Regla 1: si hay oro >= 50, recomendar subir nivel de tablero
        if state.oro >= 50 and state.nivel_tablero < 9:
            recs.append({
                "tipo": "subir_nivel",
                "explicacion": "Tienes mucho oro, subir nivel aumenta chance de héroes fuertes."
            })

        # Regla 2: si hay héroes baratos que encajen en sinergias activas (mock)
        # Aquí todavía no conocemos bien sinergias, así que solo ejemplo:
        for hero in state.tienda:
            if hero.coste == 1 and hero.slot_index == 0:
                recs.append({
                    "tipo": "comprar_heroe",
                    "heroe": hero.nombre,
                    "slot": hero.slot_index,
                    "explicacion": f"Heroe barato {hero.nombre} en primer slot, buena apuesta temprana."
                })
                break

        # Regla 3: si vida muy baja, recomendar decisiones agresivas
        if state.vida < 20:
            recs.append({
                "tipo": "agresivo",
                "explicacion": "Vida baja: gasta oro para mejorar tablero y evitar eliminación."
            })

        # Regla de seguridad: si nada se dispara, al menos sugerir inspeccionar algo nuevo
        if not recs:
            recs.append({
                "tipo": "no_accion",
                "explicacion": "No hay jugada clara con alta confianza, mejor conservar oro."
            })

        return recs

    @staticmethod
    def recs_to_json(recs: List[Dict[str, Any]]) -> str:
        return json.dumps(recs, ensure_ascii=False)
