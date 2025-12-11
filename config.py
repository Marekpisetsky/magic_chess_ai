# Configuracion de modelos y captura

# mock -> usa el VLM de pruebas que inventa un estado.
# api  -> llama al VLM de inferencia via NIM/API (Nemotron, etc).
VLM_BACKEND = "api"

VLM_API_BASE_URL = "http://localhost:1234"
VLM_API_MODEL = "qwen2-vl-7b-instruct"
VLM_API_KEY = ""  # en local no se usa

# Titulo de la ventana del juego (ajustalo si es distinto)
GAME_WINDOW_TITLE = "MagicChessGoGo"
CAPTURE_FPS = 5

DB_PATH = "magic_chess.db"
