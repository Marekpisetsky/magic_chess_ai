from core.vision import analyze_frame

IMAGE_PATH = "board.jpg"  # o el nombre que estés usando


def main() -> None:
    state = analyze_frame(IMAGE_PATH)

    print("=== ScreenState ===")
    print(f"Tipo de pantalla: {state.screen_type}")
    print(f"Ronda: {state.round_label}")
    print(f"Jugador local: {state.you.name}")
    print(f"  HP: {state.you.hp}")
    print(f"  Oro: {state.you.gold}")
    print(f"  Nivel: {state.you.level}")
    print("Otros jugadores:")
    for p in state.others:
        print(f"  - {p.name}: HP={p.hp}")


if __name__ == "__main__":
    main()
