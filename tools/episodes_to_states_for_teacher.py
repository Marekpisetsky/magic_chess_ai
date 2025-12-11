# tools/episodes_to_states_for_teacher.py
"""
Convierte episodios (episode_*.jsonl) en estados sueltos para el teacher GPT.

Salida:
  data/states_for_teacher.jsonl con:
    {"state": [...], "info": {"round": "...", "gold": X, "level": Y, "hp": Z}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


EPISODES_DIR = Path("data/episodes")
OUT_PATH = Path("data/states_for_teacher.jsonl")


def main() -> None:
    if not EPISODES_DIR.exists():
        raise SystemExit(f"No existe {EPISODES_DIR}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for ep_file in sorted(EPISODES_DIR.glob("episode_*.jsonl")):
            with ep_file.open("r", encoding="utf-8") as f_in:
                lines = f_in.readlines()

            # meta
            meta = {}
            if lines:
                try:
                    meta = json.loads(lines[0]).get("meta", {})
                except Exception:
                    meta = {}

            # ejemplo: submuestreo simple, coger solo 1 de cada N pasos
            N = 5
            for idx, line in enumerate(lines[1:]):  # skip meta
                if idx % N != 0:
                    continue

                rec = json.loads(line)
                state = rec.get("state")
                info = rec.get("info", {})
                if state is None:
                    continue

                round_label = info.get("round")
                gold = info.get("gold")
                level = info.get("level")
                hp = info.get("hp")

                out = {
                    "state": state,
                    "info": {
                        "round": round_label,
                        "gold": gold,
                        "level": level,
                        "hp": hp,
                        "episode_id": meta.get("episode_id"),
                    },
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Estados para teacher guardados en {OUT_PATH}")


if __name__ == "__main__":
    main()
