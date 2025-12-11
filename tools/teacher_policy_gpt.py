# tools/teacher_policy_gpt.py
"""
Script para etiquetar estados con una acción sugerida por GPT (teacher).

Entrada:
  data/states_for_teacher.jsonl, con líneas tipo:
    {"state": [...], "info": {"round": "1-2", "gold": 6, "level": 3, "hp": 100}}

Salida:
  data/teacher_labels.jsonl, con líneas tipo:
    {"state": [...], "info": {...}, "best_action": "level_up"}

Este dataset se puede usar para entrenar la PolicyNetwork (imitation learning).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


INPUT_PATH = Path("data/states_for_teacher.jsonl")
OUT_PATH = Path("data/teacher_labels.jsonl")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ACTIONS = [
    "noop",
    "level_up",
    "reroll",
    "buy_unit",
    "sell_unit",
]


def build_prompt(state: Dict[str, Any], info: Dict[str, Any]) -> str:
    """
    Construye el prompt textual para GPT.

    state: vector numérico, pero aquí nos interesa más info semántica de 'info'.
    info: diccionario con campos tipo round, gold, level, hp, etc.
    """
    round_label = info.get("round", "1-1")
    gold = info.get("gold", 0)
    level = info.get("level", 1)
    hp = info.get("hp", 100)

    return (
        "Eres un experto jugando Magic Chess. Voy a darte el estado actual de la partida y "
        "quiero que elijas la MEJOR acción entre este conjunto:\n"
        f"{ACTIONS}\n\n"
        "Estado relevante:\n"
        f"- Ronda: {round_label}\n"
        f"- Oro: {gold}\n"
        f"- Nivel: {level}\n"
        f"- Vida (HP): {hp}\n\n"
        "Devuélveme SOLO un JSON con este formato:\n"
        "{ \"best_action\": \"una_de_las_acciones\" }\n"
        "sin comentarios extra, sin texto adicional.\n"
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"No existe el archivo de entrada: {INPUT_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec.get("state", [])
            info = rec.get("info", {})

            prompt = build_prompt(state, info)

            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Eres un jugador profesional de Magic Chess."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = resp.choices[0].message.content
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        text_parts.append(c["text"])
                content = "".join(text_parts)

            try:
                label = json.loads(content)
            except json.JSONDecodeError:
                # intentar recortar la parte JSON
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    label = json.loads(content[start : end + 1])
                else:
                    print("No se pudo parsear JSON en respuesta:", content)
                    continue

            best_action = label.get("best_action")
            if best_action not in ACTIONS:
                print("Acción inválida recibida:", best_action, "respuesta=", label)
                continue

            out_rec = {
                "state": state,
                "info": info,
                "best_action": best_action,
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            print("Etiquetado con teacher:", out_rec)

    print(f"Etiquetas de teacher guardadas en {OUT_PATH}")


if __name__ == "__main__":
    main()
