import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI

# OCR opcional (si pytesseract esta instalado y el binario disponible)
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Coordenadas normalizadas (x1, y1, x2, y2) para 1920x1080.
# Ajustadas con las capturas de referencia.
ROUND_BOX = (0.32, 0.01, 0.36, 0.04)        # barra superior I-1/I-2 (a la izq. del timer verde)
LEVEL_BOX = (0.01, 0.86, 0.07, 0.91)        # orbe azul abajo-izquierda (nivel)
GOLD_BOX = (0.90, 0.83, 0.99, 0.90)         # moneda grande abajo-derecha (recortado izq/alto)
SHOP_DETECT_BOX = (0.03, 0.15, 0.97, 0.60)  # banda donde aparece la tienda
PLAYERS_BOX = (0.80, 0.07, 0.95, 0.75)      # lista de jugadores/vidas a la derecha
HP_ONLY_BOX = (0.87, 0.07, 0.95, 0.75)      # columna de corazones


@dataclass
class HUDReadout:
    round_label: Optional[str]
    nivel_tablero: Optional[int]
    oro: Optional[int]
    tienda_abierta: bool
    debug_path: Optional[str] = None


class HUDReader:
    """Lee HUD (oro, nivel, ronda, tienda abierta) usando recortes y el VLM local."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        debug_overlay: bool = False,
        use_ocr_numbers: bool = True,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key or "lm-studio")
        self.model = model
        self.debug_overlay = debug_overlay
        self.use_ocr_numbers = use_ocr_numbers
        self.ocr_available = pytesseract is not None
        if self.use_ocr_numbers and not self.ocr_available:
            print("[hud] Aviso: pytesseract no disponible; usando solo VLM para numeros.")

    @staticmethod
    def _crop(frame: np.ndarray, box: Tuple[float, float, float, float]) -> Image.Image:
        h, w, _ = frame.shape
        x1, y1, x2, y2 = box
        left = int(x1 * w)
        top = int(y1 * h)
        right = int(x2 * w)
        bottom = int(y2 * h)
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        return Image.fromarray(frame[top:bottom, left:right, :])

    def _encode_image(self, img: Image.Image) -> Tuple[str, str]:
        import io

        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return "image/png", base64.b64encode(bio.getvalue()).decode("utf-8")

    def _ocr_digits(self, img: Image.Image) -> Optional[int]:
        if pytesseract is None:
            return None
        # Preprocesado ligero para numeros blancos/amarillos sobre fondos claros
        gray = img.convert("L")
        # Invertir si fondo claro
        arr = np.array(gray)
        if arr.mean() > 128:
            arr = 255 - arr
        # Binariza
        arr = (arr > 120).astype(np.uint8) * 255
        try:
            text = pytesseract.image_to_string(arr, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return int(digits)
        except Exception:
            return None
        return None

    def _ask_multi(self, crops: Dict[str, Image.Image]) -> Dict:
        """Una sola llamada con 4 recortes."""
        content = [
            {
                "type": "text",
                "text": (
                    "Recibes 4 imagenes: round_box, level_box, gold_box, shop_box. "
                    "Devuelve SOLO un JSON valido con las claves: "
                    "{\"round_label\": string|null, \"nivel_tablero\": int|null, "
                    "\"oro\": int|null, \"tienda_abierta\": true|false}. "
                    "round_label es texto tipo I-1/I-2; nivel_tablero es el numero blanco del orbe azul; "
                    "oro es el numero grande en la moneda amarilla; tienda_abierta es true si la tienda con 5 cartas y botones de oro esta visible."
                ),
            }
        ]

        for name, img in crops.items():
            mime, b64 = self._encode_image(img)
            content.append({"type": "text", "text": f"Imagen: {name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.choices[0].message.content or ""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    return {}
            return {}

    def _save_debug(self, frame: np.ndarray, result: Dict) -> str:
        if not self.debug_overlay:
            return ""
        img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(img, "RGBA")
        boxes = {
            "round": ROUND_BOX,
            "nivel": LEVEL_BOX,
            "oro": GOLD_BOX,
            "players": PLAYERS_BOX,
            "hp": HP_ONLY_BOX,
        }
        colors = {
            "round": (0, 255, 0, 90),     # verde
            "nivel": (0, 128, 255, 110),  # azul claro
            "oro": (255, 215, 0, 110),    # amarillo
            "shop": (255, 0, 0, 70),      # rojo
            "players": (128, 0, 255, 60), # morado
            "hp": (0, 255, 255, 50),      # cian
        }
        w, h = img.size
        for name, (x1, y1, x2, y2) in boxes.items():
            draw.rectangle(
                (x1 * w, y1 * h, x2 * w, y2 * h),
                outline=colors[name][:3],
                width=3,
                fill=colors[name],
            )
        # pinta shop solo si se detecta abierta
        if result.get("tienda_abierta", False):
            x1, y1, x2, y2 = SHOP_DETECT_BOX
            draw.rectangle(
                (x1 * w, y1 * h, x2 * w, y2 * h),
                outline=colors["shop"][:3],
                width=3,
                fill=colors["shop"],
            )

        frames_dir = Path("frames")
        frames_dir.mkdir(exist_ok=True)
        out_path = frames_dir / "hud_debug.png"
        img.save(out_path)
        return str(out_path)

    def read(self, frame: np.ndarray) -> HUDReadout:
        crops = {
            "round_box": self._crop(frame, ROUND_BOX),
            "level_box": self._crop(frame, LEVEL_BOX),
            "gold_box": self._crop(frame, GOLD_BOX),
            "shop_box": self._crop(frame, SHOP_DETECT_BOX),
        }
        result = self._ask_multi(crops)

        debug_path = self._save_debug(frame, result)

        # OCR preferente para oro y nivel si disponible
        ocr_oro = self._ocr_digits(crops["gold_box"]) if self.use_ocr_numbers else None
        ocr_nivel = self._ocr_digits(crops["level_box"]) if self.use_ocr_numbers else None

        return HUDReadout(
            round_label=result.get("round_label"),
            nivel_tablero=ocr_nivel if ocr_nivel is not None else result.get("nivel_tablero"),
            oro=ocr_oro if ocr_oro is not None else result.get("oro"),
            tienda_abierta=bool(result.get("tienda_abierta", False)),
            debug_path=debug_path or None,
        )
