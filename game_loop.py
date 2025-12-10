import time
from datetime import datetime

from core.vision import analyze_frame
from core.state import ScreenState

IMAGE_PATH = "board.jpg"  # más adelante aquí pondremos una captura en tiempo real


def handle_state(state: ScreenState) -> None:
    """
    Aquí en el futuro irá la lógica de decisiones:
    - economía
    - subir nivel
    - reorganizar tablero
    etc.

    De momento solo mostramos un resumen.
    """
    print("--------------------------------------------------")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Frame analizado")
    print(f"  Pantalla: {state.screen_type}")
    print(f"  Ronda   : {state.round_label}")
    print(f"  Tú      : {state.you.name} | HP={state.you.hp} | Oro={state.you.gold} | Nivel={state.you.level}")
    print("  Otros   :")
    for p in state.others:
        print(f"    - {p.name}: HP={p.hp}")


def main() -> None:
    # Por ahora, 5 iteraciones para que lo veas claro.
    # Más adelante esto puede ser while True con captura en tiempo real.
    for i in range(5):
        state = analyze_frame(IMAGE_PATH)
        handle_state(state)
        time.sleep(2)  # espera 2 segundos entre frames


if __name__ == "__main__":
    main()
