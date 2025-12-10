import base64
import json
from typing import Any, Dict, List

import cv2
import numpy as np
import requests

from config import VLM_API_BASE_URL, VLM_API_MODEL, VLM_API_KEY
from .state import GameState, ShopHero


class NemotronVLVLM:
    """
    Cliente para NVIDIA Llama 3.1 Nemotron Nano VL 8B v1 via NIM.
    Envia una captura de Magic Chess: Go Go y devuelve un GameState.
    """

    def __init__(
        self,
        base_url: str = VLM_API_BASE_URL,
        model: str = VLM_API_MODEL,
        api_key: str = VLM_API_KEY,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    @staticmethod
    def _encode_image(frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", frame)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _build_prompt(ronda: int) -> str:
        return (
            "Eres un asistente experto en el juego \"Magic Chess: Go Go\".\n"
            "Analiza la captura de pantalla y devuelve EXCLUSIVAMENTE un JSON "
            "con el estado actual del juego.\n\n"
            "Usa EXACTAMENTE esta estructura:\n"
            "{\n"
            '  "fase": "pre_game" | "early" | "mid" | "late",\n'
            f"  \"ronda\": {ronda},\n"
            '  "oro": int,\n'
            '  "vida": int,\n'
            '  "nivel_tablero": int,\n'
            '  "xp_actual": int,\n'
            '  "tienda": [\n'
            '    {"slot_index": 0, "nombre": "HeroeA", "coste": int},\n'
            '    {"slot_index": 1, "nombre": "HeroeB", "coste": int},\n'
            '    {"slot_index": 2, "nombre": "HeroeC", "coste": int},\n'
            '    {"slot_index": 3, "nombre": "HeroeD", "coste": int},\n'
            '    {"slot_index": 4, "nombre": "HeroeE", "coste": int}\n'
            "  ],\n"
            '  "sinergias_activas": {"sinergiaA": int, "sinergiaB": int},\n'
            '  "comandante": "NombreComandante" | "",\n'
            '  "emblema": "NombreEmblema" | "",\n'
            '  "tienda_abierta": true | false\n'
            "}\n\n"
            "NO incluyas texto fuera del JSON.\n"
            "Si algun valor no se puede leer bien, estima el valor mas probable."
        )

    def _call_api(self, frame: np.ndarray, ronda: int) -> Dict:
        img_b64 = self._encode_image(frame)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": self._build_prompt(ronda),
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        url = f"{self.base_url}/v1/chat/completions"
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    def analyze_frame(self, frame: Any, ronda: int) -> GameState:
        if frame is None:
            return GameState(
                fase="unknown",
                ronda=ronda,
                oro=0,
                vida=100,
                nivel_tablero=1,
                xp_actual=0,
                tienda=[],
                tablero=[],
                banco=[],
                sinergias_activas={},
                sinergias_potenciales={},
                comandante="",
                emblema="",
                confianza_lectura=0.0,
                tienda_abierta=False,
            )

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        result = self._call_api(frame, ronda)

        tienda_raw: List[Dict[str, Any]] = result.get("tienda", [])
        tienda: List[ShopHero] = []
        for item in tienda_raw:
            try:
                tienda.append(
                    ShopHero(
                        slot_index=int(item.get("slot_index", 0)),
                        nombre=str(item.get("nombre", "")),
                        coste=int(item.get("coste", 1)),
                    )
                )
            except Exception:
                continue

        return GameState(
            fase=result.get("fase", "early"),
            ronda=ronda,
            oro=int(result.get("oro", 0)),
            vida=int(result.get("vida", 100)),
            nivel_tablero=int(result.get("nivel_tablero", 1)),
            xp_actual=int(result.get("xp_actual", 0)),
            tienda=tienda,
            tablero=[],
            banco=[],
            sinergias_activas=result.get("sinergias_activas", {}),
            sinergias_potenciales={},
            comandante=result.get("comandante", ""),
            emblema=result.get("emblema", ""),
            confianza_lectura=0.9,
            tienda_abierta=bool(result.get("tienda_abierta", False)),
        )
