import cv2
from core.vlm_nemotron import NemotronVLVLM

def main():
    # Ruta a tu captura de pantalla
    image_path = "samples/magic_chess_1.png"

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"No pude cargar la imagen: {image_path}")
        return

    vlm = NemotronVLVLM()
    ronda = 1  # por ahora un número fijo

    game_state = vlm.analyze_frame(frame, ronda)

    print("=== GameState recibido ===")
    print(f"fase: {game_state.fase}")
    print(f"ronda: {game_state.ronda}")
    print(f"oro: {game_state.oro}")
    print(f"vida: {game_state.vida}")
    print(f"nivel_tablero: {game_state.nivel_tablero}")
    print(f"xp_actual: {game_state.xp_actual}")
    print("tienda:")
    for h in game_state.tienda:
        print(f"  slot {h.slot_index}: {h.nombre} (coste {h.coste})")
    print("sinergias_activas:", game_state.sinergias_activas)
    print("comandante:", game_state.comandante)
    print("emblema:", game_state.emblema)

if __name__ == "__main__":
    main()
