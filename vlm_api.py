import os
import base64
import json
from typing import Any, Dict

from openai import OpenAI


# Usa la variable de entorno OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modelo de visión. Ajusta según tengas acceso:
# - "gpt-4.1"
# - "gpt-4o"
# - "gpt-4o-mini"
OPENAI_VLM_MODEL = "gpt-4.1"


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_hud_prompt() -> str:
    return (
        "Estás analizando una captura del HUD del modo Magic Chess de Mobile Legends.\n"
        "La imagen contiene:\n"
        "- La ronda, mostrada como algo tipo '1-1', '1-2', '2-3' en la parte superior.\n"
        "- El nivel del jugador, p.ej. 'Nv.3', normalmente en la parte inferior izquierda.\n"
        "- El oro (gold) en la parte inferior derecha, número dentro de un círculo dorado.\n"
        "- La vida (HP) del jugador principal (self), normalmente como número al lado de la barra de vida.\n\n"
        "Devuelve SIEMPRE un JSON con este formato EXACTO (sin texto extra):\n"
        "{\n"
        '  \"round\": \"1-1\",        // string o null si no se puede leer\n'
        '  \"level\": 3,             // entero o null\n'
        '  \"gold\": 2,              // entero o null\n'
        '  \"hp_self\": 100          // entero o null\n'
        "}\n"
        "Si no puedes leer algún valor con seguridad, usa null. "
        "No devuelvas comentarios, ni explicaciones, ni nada fuera del JSON."
    )


def analyze_hud_image(image_path: str) -> Dict[str, Any]:
    """
    Envía una captura de pantalla del juego a la API de OpenAI para que
    extraiga round, level, gold y hp_self. Devuelve un dict con esos campos.
    """
    image_b64 = _encode_image_to_base64(image_path)
    prompt = _build_hud_prompt()

    response = client.chat.completions.create(
        model=OPENAI_VLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content

    if isinstance(content, list):
        # SDK a veces devuelve lista de segmentos
        text_parts = []
        for c in content:
            if isinstance(c, dict) and "text" in c:
                text_parts.append(c["text"])
        content = "".join(text_parts)

    if not isinstance(content, str):
        raise ValueError(f"Respuesta inesperada de la API: {content!r}")

    # Intentamos parsear JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start : end + 1])
        else:
            raise ValueError(f"No se pudo extraer JSON de la respuesta: {content!r}")

    # Normalizar campos
    for key in ("round", "level", "gold", "hp_self"):
        data.setdefault(key, None)

    return data
