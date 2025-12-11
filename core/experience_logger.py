# core/experience_logger.py
"""
Logger de experiencia para el agente de Magic Chess.

Guarda transiciones (state, action, reward, done, info) en formato JSONL,
con una línea inicial de metadatos para trazabilidad.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class EpisodeLogger:
    """
    Registra una partida (episodio) como una lista de transiciones.

    Cada transición:
        {
          "state": [...],       # vector numérico (lista de floats/ints)
          "action": "level_up", # string de acción
          "reward": 0.0,        # float
          "done": false,        # bool
          "info": { ... }       # opcional, depuración/contexto
        }

    Primera línea del archivo:
        {
          "meta": {
            "episode_id": "...",
            "created_utc": "...",
            "schema_version": 1,
            ...extra_meta
          }
        }
    """

    def __init__(self, base_dir: str = "data/episodes") -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._transitions: List[Dict[str, Any]] = []
        self._episode_id: Optional[str] = None
        self._start_time: Optional[datetime] = None

    def start_episode(self) -> None:
        """
        Inicia un episodio nuevo. Si ya había uno en curso, se descarta.
        """
        self._episode_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self._start_time = datetime.utcnow()
        self._transitions = []

    def log_step(
        self,
        state_vector: list[float] | list[int],
        action: str,
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Registra un paso del episodio.
        """
        if self._episode_id is None:
            # Si se olvidó llamar a start_episode, lo hacemos por ti.
            self.start_episode()

        tr: Dict[str, Any] = {
            "state": list(state_vector),
            "action": str(action),
            "reward": float(reward),
            "done": bool(done),
        }
        if info:
            tr["info"] = info

        self._transitions.append(tr)

    def end_episode(self, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
        """
        Cierra el episodio actual y lo guarda en disco.
        Devuelve la ruta del archivo JSONL.
        """
        if self._episode_id is None or self._start_time is None:
            raise RuntimeError("No hay episodio en curso.")

        filename = f"episode_{self._episode_id}.jsonl"
        out_path = self.base_path / filename

        meta: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "created_utc": self._start_time.isoformat() + "Z",
            "schema_version": 1,
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_record = {"meta": meta}

        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
            for tr in self._transitions:
                f.write(json.dumps(tr, ensure_ascii=False) + "\n")

        # reset interno
        self._episode_id = None
        self._start_time = None
        self._transitions = []

        return out_path
