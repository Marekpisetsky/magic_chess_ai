# tools/manual_hud_labeler.py
"""
Etiquetador manual de HUD para Magic Chess.

Recorre las imágenes de un directorio (por defecto data/frames),
las muestra una por una y te pide que introduzcas:

  - round_label (ej. 1-1, 1-2, 2-3)
  - gold (oro actual)
  - level (nivel del jugador)
  - hp (vida del jugador)

Guarda las etiquetas en data/labels.jsonl en formato JSONL,
una línea por imagen:

  {"image": "data/frames/frame_0001.png",
   "round_label": "1-1",
   "gold": 2,
   "level": 3,
   "hp": 100}
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import argparse


def open_image(path: Path) -> None:
    """
    Abre la imagen con el visor por defecto del sistema.
    Usamos 'start' en Windows, 'xdg-open' en Linux, 'open' en macOS.
    """
    path = path.resolve()
    if not path.exists():
        print(f"[AVISO] Imagen no existe: {path}")
        return

    try:
        if os.name == "nt":  # Windows
            # shell=True para usar 'start'
            subprocess.Popen(['start', str(path)], shell=True)
        elif sys.platform == "darwin":  # macOS
            subprocess.Popen(["open", str(path)])
        else:  # Linux / otros
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"[AVISO] No se pudo abrir la imagen automáticamente: {e}")
        print(f"Abre esta ruta a mano si quieres verla: {path}")


def load_existing_labels(labels_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Carga etiquetas ya existentes para no repetir trabajo.
    Devuelve un dict { image_path_str: labels_dict }.
    """
    labels: Dict[str, Dict[str, Any]] = {}
    if not labels_path.exists():
        return labels

    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_path = rec.get("image")
            if img_path:
                labels[img_path] = rec
    return labels


def ask_label(prompt: str, default: Any = None, cast_type=None) -> Any:
    """
    Pregunta en consola con valor por defecto opcional.
    Si el usuario pulsa ENTER, se queda con el default.
    Si cast_type se indica (int, float, etc.), castea el valor.
    """
    if default is not None:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    while True:
        val = input(full_prompt).strip()
        if not val:
            return default
        if cast_type is None:
            return val
        try:
            return cast_type(val)
        except ValueError:
            print(f"Valor inválido, esperaba {cast_type.__name__}. Intenta de nuevo.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/frames",
        help="Directorio donde están las capturas del HUD (PNG/JPG).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/labels.jsonl",
        help="Ruta del archivo de salida JSONL con las etiquetas.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_path = Path(args.out)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise SystemExit(f"No existe el directorio de imágenes: {images_dir}")

    # Cargar etiquetas existentes (para continuar donde lo dejaste)
    existing = load_existing_labels(labels_path)
    print(f"Ya hay {len(existing)} imágenes etiquetadas en {labels_path}")

    # Listar imágenes
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    all_images: List[Path] = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in exts
    )

    if not all_images:
        raise SystemExit(f"No se encontraron imágenes en {images_dir}")

    print(f"Encontradas {len(all_images)} imágenes en {images_dir}")
    print("Comenzando etiquetado manual. Comandos especiales:")
    print("  - Escribe 'skip' en cualquier campo para saltar esta imagen.")
    print("  - Escribe 'quit' en round_label para salir del programa.\n")

    # Abrir archivo de salida en modo append
    out_file = labels_path.open("a", encoding="utf-8")

    try:
        for idx, img_path in enumerate(all_images, start=1):
            img_str = str(img_path).replace("\\", "/")  # normalizar

            # Si ya está etiquetada, saltar
            if img_str in existing:
                continue

            print(f"\n[{idx}/{len(all_images)}] Imagen: {img_str}")
            open_image(img_path)

            # Valores por defecto (puedes cambiarlos si quieres)
            default_round = "1-1"
            default_gold = 0
            default_level = 1
            default_hp = 100

            # Pedir datos al usuario
            round_label = ask_label("Ronda (ej. 1-1, 1-2, 2-3)", default_round, str)
            if isinstance(round_label, str) and round_label.lower() == "quit":
                print("Saliendo por petición del usuario.")
                break
            if isinstance(round_label, str) and round_label.lower() == "skip":
                print("Saltando imagen.")
                continue

            gold = ask_label("Oro actual", default_gold, int)
            if isinstance(gold, str) and gold.lower() == "skip":
                print("Saltando imagen.")
                continue

            level = ask_label("Nivel del jugador", default_level, int)
            if isinstance(level, str) and level.lower() == "skip":
                print("Saltando imagen.")
                continue

            hp = ask_label("Vida (HP)", default_hp, int)
            if isinstance(hp, str) and hp.lower() == "skip":
                print("Saltando imagen.")
                continue

            rec = {
                "image": img_str,
                "round_label": round_label,
                "gold": gold,
                "level": level,
                "hp": hp,
            }

            out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_file.flush()

            print("Guardado:", rec)

    finally:
        out_file.close()
        print(f"\nEtiquetado terminado. Labels guardadas en {labels_path}")


if __name__ == "__main__":
    import os, sys
    main()
