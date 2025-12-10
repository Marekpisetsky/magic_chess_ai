# core/overlay.py
from typing import List, Dict, Any


class OverlayRenderer:
    """
    Stub: de momento solo imprime por consola.
    Luego lo sustituimos por un overlay real sobre la ventana del juego.
    """

    def __init__(self):
        pass

    def show_recommendations(self, recs: List[Dict[str, Any]]):
        print("\n=== RECOMENDACIONES IA ===")
        for r in recs:
            tipo = r.get("tipo")
            expl = r.get("explicacion", "")
            print(f"- [{tipo}] {expl}")
        print("==========================")
