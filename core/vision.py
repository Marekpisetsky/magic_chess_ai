import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from openai import OpenAI

from core.state import PlayerStatus, ScreenState

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen2-vl-7b-instruct"

# Ajusta esto a tu nombre en el juego
LOCAL_PLAYER_NAME = "CodevaMP"


# ---------------------------------------------------------------------
# Helpers de imagen
# ---------------------------------------------------------------------


def _content_to_text(content) -> str:
    """
    Normaliza message.content de OpenAI (puede venir como str o lista de bloques).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
        return " ".join(str(x) for x in content)
    return str(content)


def _strip_json_fences(raw: str) -> str:
    """
    El modelo a veces envuelve la respuesta en ```json ... ```. Quitamos esos fences.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _encode_full_image(path: str) -> tuple[str, str]:
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return mime, _b64_encode(data)


def _encode_players_strip(path: str) -> tuple[str, str]:
    """
    Recorta solo la columna derecha donde está la lista de jugadores
    y la devuelve en base64.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Recortamos aprox. el 28% derecho y 4%-96% vertical para asegurar capturar HUD
    left = int(w * 0.72)
    top = int(h * 0.04)
    right = w
    bottom = int(h * 0.96)

    crop = img.crop((left, top, right, bottom))

    buf = BytesIO()
    crop.save(buf, format="PNG")
    return "image/png", _b64_encode(buf.getvalue())


# ---------------------------------------------------------------------
# Parseo del JSON a ScreenState
# ---------------------------------------------------------------------


def build_screen_state_from_json(data: dict, local_name: str) -> ScreenState:
    """
    Convierte el JSON combinado (estado global + lista de jugadores)
    en un ScreenState usable por el resto del código.
    """
    screen_type = data.get("screen_type") or "other"
    round_label = data.get("round_label")

    players_raw = data.get("players") or []
    own_raw = data.get("own") or {}

    players: list[PlayerStatus] = []

    def is_plausible_player_name(name: str) -> bool:
        """
        Filtra basura evidente:
        - cadenas vacías
        - cosas tipo 'Nv.4'
        """
        name = (name or "").strip()
        if len(name) < 2:
            return False
        if name.lower().startswith("nv."):
            return False
        return True

    # Crear PlayerStatus para cada entrada de "players"
    for p in players_raw:
        name = (p.get("name") or "").strip()
        if not is_plausible_player_name(name):
            continue

        hp = p.get("hp")
        if isinstance(hp, (int, float)):
            # Vida de jugador siempre 0–100; si se va, ignoramos
            if hp < 0 or hp > 100:
                hp = None

        players.append(
            PlayerStatus(
                name=name,
                hp=int(hp) if isinstance(hp, (int, float)) else None,
                is_local=False,
            )
        )

    # Buscar jugador local por nombre
    local_player = None
    for pl in players:
        if pl.name == local_name:
            local_player = pl
            pl.is_local = True
            break

    # Si no lo encontramos, creamos uno vacío
    if local_player is None:
        local_player = PlayerStatus(name=local_name, is_local=True)
        players.append(local_player)
    else:
        # nos aseguramos de que el nombre sea exactamente el tuyo
        local_player.name = local_name

    # Oro / nivel desde "own"
    own_gold = own_raw.get("gold")
    own_level = own_raw.get("level")
    if isinstance(own_level, str):
        digits = "".join(ch for ch in own_level if ch.isdigit())
        own_level = int(digits) if digits else None

    if own_gold is not None:
        local_player.gold = own_gold
    if own_level is not None:
        local_player.level = own_level

    # Separar otros jugadores
    others = [p for p in players if not p.is_local]

    return ScreenState(
        screen_type=screen_type,
        round_label=round_label,
        you=local_player,
        others=others,
    )


# ---------------------------------------------------------------------
# Helpers de JSON robusto
# ---------------------------------------------------------------------


def _safe_json_loads(raw: str, tag: str) -> dict:
    """
    Intenta parsear JSON. Si falla, loguea y devuelve dict vac??o.
    """
    raw_clean = _strip_json_fences(_content_to_text(raw)).strip()
    if not raw_clean:
        print(f"[vision] ?s? Respuesta vac??a ({tag})")
        return {}
    try:
        return json.loads(raw_clean)
    except json.JSONDecodeError as e:
        start_json = raw_clean.find("{")
        end_json = raw_clean.rfind("}")
        if start_json != -1 and end_json != -1 and end_json > start_json:
            candidate = raw_clean[start_json : end_json + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
        print(f"[vision] ?s? JSON inv??lido en {tag}: {e}")
        print("[vision] Contenido crudo:")
        print(raw_clean[:400])
        return {}
# ---------------------------------------------------------------------
# Llamada principal a LM Studio
# ---------------------------------------------------------------------


def analyze_frame(image_path: str) -> ScreenState:
    """
    Toma un screenshot completo y hace DOS llamadas al modelo:

    1) full frame -> tipo de pantalla, ronda, oro, nivel.
    2) strip derecha -> SOLO la lista de jugadores.

    Luego combina ambas cosas en un ScreenState.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_file}")
    image_path = str(image_file)

    client = OpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
    )

    # --- 1) Imagen completa ---
    full_mime, full_b64 = _encode_full_image(image_path)

    global_system = """
Eres un analizador de Magic Chess.

Esta llamada SOLO debe sacar:
- tipo de pantalla (screen_type)
- texto de ronda (round_label)
- oro y nivel del jugador local (own.gold, own.level)

NO intentes leer la lista de jugadores; eso se hace en otra llamada.
Si no puedes ver un dato, devuélvelo como null. No inventes valores.
Devuelve SIEMPRE un JSON con esta forma exacta:

{
  "screen_type": "own_board" | "enemy_board" | "shop_open" | "carousel" | "lobby_or_menu" | "other",
  "round_label": string | null,
  "own": {
    "gold": int | null,
    "level": int | null
  }
}
"""

    global_user = """
Analiza la captura y devuelve SOLO el JSON con:

{
  "screen_type": "...",
  "round_label": ...,
  "own": { "gold": ..., "level": ... }
}

Si no se ve, pon null. No añadas texto fuera del JSON.
"""

    resp_global = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": global_system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": global_user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{full_mime};base64,{full_b64}"
                        },
                    },
                ],
            },
        ],
    )

    raw_global = _content_to_text(resp_global.choices[0].message.content)
    data_global = _safe_json_loads(raw_global, "global")

    # Defaults si vino vacío / roto
    if not data_global:
        data_global = {
            "screen_type": "other",
            "round_label": None,
            "own": {"gold": None, "level": None},
        }

    # --- 2) Strip derecha con solo la lista de jugadores ---
    strip_mime, strip_b64 = _encode_players_strip(image_path)

    players_system = """
Eres un OCR preciso del HUD DERECHO de Magic Chess.

La imagen que ves está recortada: SOLO contiene la columna vertical de
la derecha donde están los retratos de los jugadores con barras verdes
de vida y números 0–100.

Tu trabajo:
- Leer CADA fila visible (hasta 8).
- Para cada fila:
    * "name": texto del nombre junto al retrato.
    * "hp": número del corazón verde (0–100). Si no se ve claro, null.
    * "is_highlighted": true si la fila está resaltada, false si no, null si dudas.

SI NO ves esa columna de jugadores, devuelve "players": [].
Si un nombre no se lee, usa null y no inventes.

FORMATO EXACTO:

{
  "players": [
    {
      "name": string,
      "hp": int | null,
      "is_highlighted": bool | null
    }
  ]
}
"""

    players_user = """
Lee la lista de jugadores del HUD derecho y devuelve SOLO:

{
  "players": [
    { "name": "...", "hp": ..., "is_highlighted": ... }
  ]
}
"""

    resp_players = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        messages=[
            {"role": "system", "content": players_system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": players_user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{strip_mime};base64,{strip_b64}"
                        },
                    },
                ],
            },
        ],
    )

    raw_players = _content_to_text(resp_players.choices[0].message.content)
    data_players = _safe_json_loads(raw_players, "players")

    if not data_players:
        data_players = {"players": []}

    # --- Combinamos ambas respuestas ---
    combined = {
        "screen_type": data_global.get("screen_type"),
        "round_label": data_global.get("round_label"),
        "own": data_global.get("own") or {},
        "players": data_players.get("players") or [],
    }

    return build_screen_state_from_json(combined, LOCAL_PLAYER_NAME)
