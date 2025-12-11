# tools/hud_debug_view.py
"""
Visualizador del HUD con recuadros y textos, para debug.
"""

from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt


# Coordenadas de ejemplo para 1920x1080. Ajusta a tu HUD real.
HUD_REGIONS: Dict[str, Tuple[int, int, int, int]] = {
    "round": (800, 10, 1120, 80),
    "level": (200, 920, 360, 1060),
    "gold": (1480, 920, 1760, 1060),
    "hp_self": (1480, 80, 1760, 160),
}


def draw_hud_debug(
    image_path: str,
    hud_values: Dict[str, str | int] | None = None,
    regions: Dict[str, Tuple[int, int, int, int]] = HUD_REGIONS,
) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"No pude abrir la imagen: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hud_values = hud_values or {}

    for key, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"{key}: {hud_values.get(key, '?')}"
        cv2.putText(
            img,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    plt.imshow(img)
    plt.axis("off")
    plt.title("HUD debug view")
    plt.show()


if __name__ == "__main__":
    dummy = {"round": "1-1", "level": 3, "gold": 2, "hp_self": 100}
    draw_hud_debug("frames/current_frame.png", dummy)
