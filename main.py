# main.py
import os
import time

from config import (
    VLM_API_BASE_URL,
    VLM_API_KEY,
    VLM_API_MODEL,
    VLM_BACKEND,
)
from core.capture import WindowCapture
from core.decision import DecisionEngine
from core.knowledge import KnowledgeBase
from core.overlay import OverlayRenderer
from core.vlm import get_vlm
from core.vlm_api import ApiVLM


def build_vlm():
    """
    Selecciona backend segun config:
    - mock: VLM de prueba que genera un estado ficticio.
    - api : llama a un endpoint estilo OpenAI/NIM.
    """
    if VLM_BACKEND == "api":
        api_key = VLM_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        return ApiVLM(
            base_url=VLM_API_BASE_URL,
            model=VLM_API_MODEL,
            api_key=api_key,
        )
    return get_vlm(VLM_BACKEND)


def main():
    kb = KnowledgeBase()
    vlm = build_vlm()
    capture = WindowCapture()
    decision_engine = DecisionEngine()
    overlay = OverlayRenderer()

    match_id = kb.start_match()
    ronda = 1

    print(f"Partida iniciada, match_id={match_id}")

    for frame in capture.capture_loop():
        if frame is None:
            print("No encuentro la ventana del juego. Abre Magic Chess: Go Go.")
            time.sleep(1)
            continue

        # En version real deberiamos detectar eventos como nueva ronda o reroll.
        # Por ahora: 1 lectura de GameState cada X segundos = nueva ronda simulada.
        game_state = vlm.analyze_frame(frame, ronda)

        recs = decision_engine.recommend_actions(game_state)
        overlay.show_recommendations(recs)

        kb.add_round(
            match_id=match_id,
            ronda=ronda,
            fase=game_state.fase,
            game_state_json=game_state.to_json(),
            recomendaciones_json=DecisionEngine.recs_to_json(recs),
            acciones_realizadas_json=None,
        )

        ronda += 1
        if ronda > 10:
            # Solo para demo: paramos despues de 10 rondas simuladas
            break

        time.sleep(5)  # simula duracion entre rondas

    kb.end_match(match_id, posicion=4, vida=30)
    print("Partida terminada (demo).")


if __name__ == "__main__":
    main()
