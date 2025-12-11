# tools/generate_labels_with_teacher.py
"""
Recorre data/raw_frames/, envía cada imagen al 'teacher' (API de visión)
y genera data/labels.jsonl con las etiquetas para el modelo local.
"""

import json
from pathlib import Path

from vlm_api import analyze_hud_image


DATA_DIR = Path("data")
FRAMES_DIR = DATA_DIR / "raw_frames"
OUT_PATH = DATA_DIR / "labels.jsonl"


def main() -> None:
    if not FRAMES_DIR.exists():
        raise SystemExit(f"No existe el directorio de frames: {FRAMES_DIR}")

    images = sorted(p for p in FRAMES_DIR.glob("*.png"))

    if not images:
        raise SystemExit(f"No se encontraron PNG en {FRAMES_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for img_path in images:
            hud = analyze_hud_image(str(img_path))
            record = {
                "image": img_path.name,
                "round": hud.get("round"),
                "level": hud.get("level"),
                "gold": hud.get("gold"),
                "hp_self": hud.get("hp_self"),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("Etiquetado:", record)

    print(f"\nListo. Dataset guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
