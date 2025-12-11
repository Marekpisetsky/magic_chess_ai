# realtime_loop.py
# Loop continuo: captura la ventana del juego, pasa por VLM + HUD/OCR recortado,
# genera GameState, recomienda acciones y persiste en la base de conocimiento.

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

from config import (
    VLM_API_BASE_URL,
    VLM_API_KEY,
    VLM_API_MODEL,
    VLM_BACKEND,
)
from core.capture import WindowCapture
from core.decision import DecisionEngine
from core.hud_reader import HUDReader
from core.knowledge import KnowledgeBase
from core.overlay import OverlayRenderer
from core.vlm_nemotron import NemotronVLVLM
from core.vlm import get_vlm
from core.vision import analyze_frame as analyze_players

HUD_DEBUG_OVERLAY = False  # pon True solo para depurar cajas (escribe hud_debug.png en cada frame)
# Factor para reducir la imagen enviada al VLM (1.0 = sin cambio). Baja a 0.5-0.7 para acelerar.
VLM_DOWNSCALE = 0.6
# Cada cuántos frames recalcular la lista de jugadores (para HP); baja la carga si >1.
PLAYERS_EVERY_N = 2


def build_vlm():
    if VLM_BACKEND == "api":
        api_key = VLM_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        return NemotronVLVLM(
            base_url=VLM_API_BASE_URL,
            model=VLM_API_MODEL,
            api_key=api_key,
        )
    return get_vlm(VLM_BACKEND)


def main() -> None:
    print("Iniciando loop en tiempo real. Ctrl+C para parar.\n")

    kb = KnowledgeBase()
    vlm = build_vlm()
    capture = WindowCapture()
    decision_engine = DecisionEngine()
    overlay = OverlayRenderer()
    hud_reader = HUDReader(
        base_url=f"{VLM_API_BASE_URL}/v1",
        model=VLM_API_MODEL,
        api_key=VLM_API_KEY or "lm-studio",
        debug_overlay=HUD_DEBUG_OVERLAY,
    )

    match_id = kb.start_match()
    ronda = 1

    last_state = None

    delay = 1.0 / capture.fps if hasattr(capture, "fps") else 1.0

    def _save_temp_frame(arr_rgb: np.ndarray) -> str:
        """Guarda frame (RGB) para reuse en vision.py."""
        frames_dir = Path("frames")
        frames_dir.mkdir(exist_ok=True)
        tmp_path = frames_dir / "tmp_realtime.png"
        cv2.imwrite(str(tmp_path), cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
        return str(tmp_path)

    try:
        while True:
            frame_bgr = capture.capture_once()
            if frame_bgr is None:
                print("[realtime] No encuentro la ventana del juego. Abrela y enfoca la partida.")
                time.sleep(delay)
                continue
            # Convertimos a RGB para evitar colores invertidos en el VLM y HUD
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            try:
                hud = hud_reader.read(frame)
            except Exception as hud_err:
                print(f"[realtime] HUD error: {hud_err}")
                hud = None

            # Reducimos tamaño solo para el VLM (HUD y jugadores usan full-res)
            vlm_input = frame
            if 0 < VLM_DOWNSCALE < 1.0:
                h, w, _ = frame.shape
                vlm_input = cv2.resize(frame, (int(w * VLM_DOWNSCALE), int(h * VLM_DOWNSCALE)), interpolation=cv2.INTER_AREA)

            try:
                gs = vlm.analyze_frame(vlm_input, ronda)
            except Exception as vision_err:
                print(f"[realtime] Error analizando frame: {vision_err}")
                time.sleep(delay)
                continue

            # Lee lista de jugadores para obtener HP del local (mas fiable que VLM global)
            local_hp = None
            # Solo analiza jugadores cada PLAYERS_EVERY_N frames para bajar carga
            if ronda % max(1, PLAYERS_EVERY_N) == 0:
                try:
                    screen_state = analyze_players(_save_temp_frame(frame))
                except Exception:
                    screen_state = None
                if screen_state and screen_state.you and screen_state.you.hp is not None:
                    local_hp = screen_state.you.hp

            # Fusion de datos: preferimos HUD/players sobre VLM global
            if hud:
                if hud.oro is not None:
                    gs.oro = int(hud.oro)
                if hud.nivel_tablero is not None:
                    gs.nivel_tablero = int(hud.nivel_tablero)
                gs.tienda_abierta = bool(hud.tienda_abierta)
                # guarda round_label textual si existe
                if hud.round_label:
                    gs.ronda = gs.ronda  # conservamos contador; podríamos guardar aparte
                if hud.debug_path:
                    print(f"[realtime] Debug HUD guardado en {hud.debug_path}")
            if local_hp is not None:
                gs.vida = local_hp

            def _smooth_state(prev, curr):
                if prev is None:
                    return curr
                if getattr(curr, "confianza_lectura", 0.0) < 0.2:
                    return prev
                if getattr(curr, "tienda_abierta", False):
                    if curr.vida in (0, None):
                        curr.vida = prev.vida
                    if curr.nivel_tablero in (0, None):
                        curr.nivel_tablero = prev.nivel_tablero
                    if curr.oro in (0, None):
                        curr.oro = prev.oro
                if abs(curr.vida - prev.vida) > 30 and getattr(curr, "confianza_lectura", 0.0) < 0.6:
                    curr.vida = prev.vida
                if abs(curr.oro - prev.oro) > 80 and getattr(curr, "confianza_lectura", 0.0) < 0.6:
                    curr.oro = prev.oro
                return curr

            gs = _smooth_state(last_state, gs)
            last_state = gs

            recs = decision_engine.recommend_actions(gs)
            overlay.show_recommendations(recs)

            kb.add_round(
                match_id=match_id,
                ronda=ronda,
                fase=gs.fase,
                game_state_json=gs.to_json(),
                recomendaciones_json=DecisionEngine.recs_to_json(recs),
                acciones_realizadas_json=None,
            )

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ronda={ronda} "
                f"fase={gs.fase} oro={gs.oro} vida={gs.vida} "
                f"nivel={gs.nivel_tablero} conf={getattr(gs, 'confianza_lectura', 0):.2f} tienda={gs.tienda_abierta}"
            )

            ronda += 1
            time.sleep(delay)

    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    except Exception as e:
        print("\nError:", e)
    finally:
        try:
            kb.end_match(match_id, posicion=None, vida=None)
        except Exception:
            pass


if __name__ == "__main__":
    main()
