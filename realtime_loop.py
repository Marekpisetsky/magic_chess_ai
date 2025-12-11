# realtime_loop.py
# Loop continuo: captura la ventana del juego, pasa por VLM + HUD/OCR recortado,
# genera GameState, recomienda acciones y persiste en la base de conocimiento.

import os
import time
from pathlib import Path
import copy

import cv2
import numpy as np
from PIL import Image

from core.experience_logger import EpisodeLogger
from core.hud_local_reader import HUDLocalReader
from core.reward import compute_reward
from core.rule_based_policy import RuleBasedPolicy
from core.state import GameState
from core.capture import WindowCapture
from config import GAME_WINDOW_TITLE

# Directorio para frames y control de episodio fake
RAW_FRAMES_DIR = Path("data/raw_frames")
RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
MAX_STEPS = 50  # terminar episodio tras este numero de pasos
_step_counter = 0
_frame_idx = 0
window_capture = WindowCapture(window_title=GAME_WINDOW_TITLE)


def capture_current_frame() -> str:
    """
    Captura la ventana del juego y guarda un PNG en data/raw_frames/.
    Si falla la captura, genera un frame negro para no romper el pipeline.
    """
    global _step_counter
    global _frame_idx
    _step_counter += 1
    _frame_idx += 1

    out_path = RAW_FRAMES_DIR / f"frame_{_frame_idx:05d}.png"

    frame_bgr = window_capture.capture_once()
    if frame_bgr is None:
        # Fallback: imagen negra 1080p
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), img)
        return str(out_path)

    cv2.imwrite(str(out_path), frame_bgr)
    return str(out_path)


def read_hud(frame_path: str, gs: GameState, hud_reader: HUDLocalReader | None) -> None:
    """
    Actualiza el estado a partir del HUD local si existe; si no, aplica defaults.
    """
    if hud_reader is not None:
        try:
            hud = hud_reader.predict_from_image_path(frame_path)
            gs.update_from_hud(hud)
            return
        except Exception as hud_err:
            print(f"[hud] Error leyendo HUD local: {hud_err}")
    # Fallback: valores por defecto para que el pipeline no se rompa
    gs.round_label = gs.round_label or "1-1"
    if getattr(gs, "oro", None) is None:
        gs.oro = 0
    if getattr(gs, "nivel_tablero", None) is None:
        gs.nivel_tablero = 1
    if getattr(gs, "vida", None) is None:
        gs.vida = 100


def env_step(action: str, gs: GameState) -> tuple[bool, dict]:
    """
    Stub: no ejecuta clicks. Cierra tras MAX_STEPS.
    """
    done = _step_counter >= MAX_STEPS
    info = {}
    if done:
        info["result"] = "lose"
    time.sleep(0.05)
    return done, info


def main() -> None:
    print("Generando episodio de prueba (stub).")

    hud_reader = None
    hud_model_path = Path("hud_model.pt")
    if hud_model_path.exists():
        try:
            print("Cargando modelo HUD local desde hud_model.pt")
            hud_reader = HUDLocalReader(str(hud_model_path))
        except Exception as hud_err:
            print(f"[realtime] No se pudo cargar hud_model.pt: {hud_err}")
            hud_reader = None
    else:
        print("AVISO: no hay hud_model.pt, usando HUD dummy.")

    gs = GameState(
        fase="early",
        ronda=1,
        round_label="1-1",
        oro=0,
        vida=100,
        nivel_tablero=1,
        xp_actual=0,
        tienda=[],
        tablero=[],
        banco=[],
        sinergias_activas={},
        sinergias_potenciales={},
        comandante="",
        emblema="",
        confianza_lectura=0.0,
        tienda_abierta=False,
    )

    logger = EpisodeLogger()
    logger.start_episode()
    policy = RuleBasedPolicy()

    done = False
    step_idx = 0
    prev_state = copy.deepcopy(gs)
    env_info = {}

    try:
        while not done:
            step_idx += 1

            frame_path = capture_current_frame()

            # Leer HUD (real si hay modelo, dummy si no)
            read_hud(frame_path, gs, hud_reader)

            state_vec = gs.to_vector()
            action = policy.choose_action(gs)

            done, env_info = env_step(action, gs)

            reward = compute_reward(prev_state, gs, done, env_info.get("result") if done else None)

            info = {
                "round": gs.round_label,
                "gold": getattr(gs, "oro", None),
                "level": getattr(gs, "nivel_tablero", None),
                "hp": getattr(gs, "vida", None),
                "step_idx": step_idx,
            }
            info.update(env_info)

            logger.log_step(
                state_vector=state_vec,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )

            prev_state = copy.deepcopy(gs)

    finally:
        meta = {
            "result": env_info.get("result", "unknown"),
            "final_round": gs.round_label,
            "max_steps": MAX_STEPS,
        }
        path = logger.end_episode(meta)
        print("Episodio guardado en:", path)


if __name__ == "__main__":
    main()
