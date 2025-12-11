# tools/gold_stats_from_episodes.py
"""
Analiza episodios en data/episodes/ y calcula estadísticas de oro por ronda.

Salida:
  data/gold_stats.json con estructura:
  {
    "1-1": {"count": N, "avg_gold": X, "min_gold": A, "max_gold": B},
    "1-2": {...},
    ...
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


EPISODES_DIR = Path("data/episodes")
OUT_PATH = Path("data/gold_stats.json")


def main() -> None:
    if not EPISODES_DIR.exists():
        raise SystemExit(f"No existe el directorio de episodios: {EPISODES_DIR}")

    stats: Dict[str, Dict[str, Any]] = {}

    for ep_file in sorted(EPISODES_DIR.glob("episode_*.jsonl")):
        with ep_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        # saltamos la primera línea (meta)
        for line in lines[1:]:
            rec = json.loads(line)
            info = rec.get("info", {})
            round_label = info.get("round")
            gold = info.get("gold")

            if round_label is None or gold is None:
                continue

            try:
                gold_val = int(gold)
            except (TypeError, ValueError):
                continue

            if round_label not in stats:
                stats[round_label] = {
                    "count": 0,
                    "sum_gold": 0,
                    "min_gold": gold_val,
                    "max_gold": gold_val,
                }

            s = stats[round_label]
            s["count"] += 1
            s["sum_gold"] += gold_val
            s["min_gold"] = min(s["min_gold"], gold_val)
            s["max_gold"] = max(s["max_gold"], gold_val)

    # calcular promedio
    out: Dict[str, Any] = {}
    for round_label, s in stats.items():
        count = s["count"]
        avg_gold = s["sum_gold"] / count if count > 0 else 0.0
        out[round_label] = {
            "count": count,
            "avg_gold": avg_gold,
            "min_gold": s["min_gold"],
            "max_gold": s["max_gold"],
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Estadísticas de oro por ronda guardadas en {OUT_PATH}")


if __name__ == "__main__":
    main()
