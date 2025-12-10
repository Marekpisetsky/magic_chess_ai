import base64
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import requests

from .state import GameState, ShopHero, BoardUnit

class ApiVLM:
    """
    Envia la captura a un modelo de vision via API y recibe un GameState.
    Implementacion generica; luego concretamos adaptadores para OpenAI, Gemini o NIM.
    """

    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _encode_image(self, frame: np.ndarray) -> str:
        # Convierte el frame en PNG base64
        _, buf = cv2.imencode(".png", frame)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _build_prompt() -> str:
        # Prompt para que el modelo devuelva EXACTAMENTE nuestro GameState
        return (
            "Eres un extractor estricto de estado para el juego Magic Chess: Go Go.\n"
            "Devuelve SOLO un JSON valido (sin texto extra) con esta estructura exacta:\n"
            "{\n"
            '  "fase": "pre_game" | "early" | "mid" | "late",\n'
            '  "ronda": <int>,\n'
            '  "oro": <int>,\n'
            '  "vida": <int>,\n'
            '  "nivel_tablero": <int>,\n'
            '  "xp_actual": <int>,\n'
            '  "tienda": [\n'
            '    {"slot_index": 0, "nombre": "NombreHeroe", "coste": <int>},\n'
            '    {"slot_index": 1, ...},\n'
            '    {"slot_index": 2, ...},\n'
            '    {"slot_index": 3, ...},\n'
            '    {"slot_index": 4, ...}\n'
            "  ],\n"
            '  "sinergias_activas": {"NombreSinergia": <int>, ...},\n'
            '  "tienda_abierta": true | false\n'
            "}\n"
            "Reglas: vida es el corazon verde (0-100), nivel_tablero es el nivel del orbe/board (1-10), oro es la moneda (0-100).\n"
            "Si no ves un valor o la tienda lo tapa, devuelve null o 0 pero no inventes.\n"
        )

    @staticmethod
    def _strip_json_fences(raw: str) -> str:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    @classmethod
    def _safe_json_loads(cls, raw: str) -> Dict:
        clean = cls._strip_json_fences(raw)
        if not clean:
            return {}
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(clean[start : end + 1])
                except Exception:
                    return {}
            return {}

    def _call_openai_style(self, frame: np.ndarray) -> Tuple[Dict, str]:
        """Ejemplo con API tipo OpenAI / NIM. Devuelve (json_dict, raw_content)."""
        img_b64 = self._encode_image(frame)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",  # OpenAI / NIM compatible
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return self._safe_json_loads(content), content

    def analyze_frame(self, frame: Any, ronda: int) -> GameState:
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        raw_text: str = ""
        try:
            data, raw_text = self._call_openai_style(frame)
        except Exception:
            data = {}

        tienda = []
        for item in data.get("tienda", []):
            if not isinstance(item, dict):
                continue
            try:
                tienda.append(
                    ShopHero(
                        slot_index=item.get("slot_index", 0),
                        nombre=item.get("nombre", ""),
                        coste=item.get("coste", 0),
                    )
                )
            except Exception:
                continue

        gs = GameState(
            fase=data.get("fase", "early") or "early",
            ronda=ronda,
            oro=data.get("oro", 0) or 0,
            vida=data.get("vida", 100) or 100,
            nivel_tablero=data.get("nivel_tablero", 1) or 1,
            xp_actual=data.get("xp_actual", 0) or 0,
            tienda=tienda,
            tablero=[],          # se puede ampliar mas adelante
            banco=[],
            sinergias_activas=data.get("sinergias_activas", {}) or {},
            comandante=data.get("comandante", "") or "",
            emblema=data.get("emblema", "") or "",
            confianza_lectura=0.7 if data else 0.0,
            tienda_abierta=bool(data.get("tienda_abierta", False)),
        )

        # Si no pudimos parsear nada, deja el bruto en disco para depurar.
        if not data and raw_text:
            frames_dir = Path("frames")
            frames_dir.mkdir(exist_ok=True)
            (frames_dir / "vlm_last_raw.txt").write_text(raw_text, encoding="utf-8")

        return gs
