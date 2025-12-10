from core.vision import analyze_frame
from core.brain import make_advice


IMAGE_PATH = "board.jpg"  # usa la misma que probaste en vision_test


def main() -> None:
    state = analyze_frame(IMAGE_PATH)
    advice = make_advice(state)

    print("=== ESTADO DETECTADO ===")
    print(f"Pantalla : {state.screen_type}")
    print(f"Ronda    : {state.round_label}")
    print(f"Tú       : {state.you.name}")
    print(f"  HP     : {state.you.hp}")
    print(f"  Oro    : {state.you.gold}")
    print(f"  Nivel  : {state.you.level}")
    print("Otros jugadores:")
    for p in state.others:
        print(f"  - {p.name}: HP={p.hp}")

    print("\n=== CONSEJO DEL CEREBRO ===")
    print(advice.summary)
    print("\nAcciones sugeridas:")
    for i, action in enumerate(advice.actions, start=1):
        print(f"  {i}. {action}")


if __name__ == "__main__":
    main()
