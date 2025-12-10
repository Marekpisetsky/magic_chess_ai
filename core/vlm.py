# core/vlm.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import random

from .state import GameState, ShopHero, BoardUnit


class VisionLanguageModel(ABC):
    @abstractmethod
    def analyze_frame(self, frame: Any, ronda: int) -> GameState:
        """
        Recibe una imagen de la pantalla y devuelve un GameState aproximado.
        'frame' sera un array tipo numpy (captura de pantalla).
        """
        ...


class MockVLM(VisionLanguageModel):
    """
    VLM de prueba que inventa un estado coherente para testear el sistema.
    Mas adelante se reemplaza por uno real (local/cloud).
    """
    def analyze_frame(self, frame: Any, ronda: int) -> GameState:
        fase = "early" if ronda < 8 else "mid" if ronda < 16 else "late"

        # Simulamos datos
        oro = random.randint(0, 50)
        vida = random.randint(20, 100)
        nivel_tablero = random.randint(3, 9)
        xp_actual = random.randint(0, 8)

        tienda = [
            ShopHero(slot_index=i,
                     nombre=f"Hero_{random.randint(1,20)}",
                     coste=random.choice([1,2,3,4,5]))
            for i in range(5)
        ]

        tablero = [
            BoardUnit(fila=random.randint(0,3),
                      columna=random.randint(0,7),
                      nombre=f"Hero_{random.randint(1,20)}",
                      estrellas=random.choice([1,2,3]))
            for _ in range(random.randint(3,8))
        ]

        sinergias_activas = {
            "Mago": random.choice([0,3,6]),
            "Guerrero": random.choice([0,2,4])
        }

        return GameState(
            fase=fase,
            ronda=ronda,
            oro=oro,
            vida=vida,
            nivel_tablero=nivel_tablero,
            xp_actual=xp_actual,
            tienda=tienda,
            tablero=tablero,
            banco=[],
            sinergias_activas=sinergias_activas,
            sinergias_potenciales={},
            comandante="Cmd_mock",
            emblema="Emblema_mock",
            confianza_lectura=0.8,
            tienda_abierta=False,
        )


def get_vlm(backend: str) -> VisionLanguageModel:
    if backend == "mock":
        return MockVLM()
    # Aqui luego anadimos implementaciones reales:
    # elif backend == "local":
    #   return LocalLlavaVLM(...)
    # elif backend == "api":
    #   return OpenAIOrGeminiVLM(...)
    else:
        raise ValueError(f"Backend VLM desconocido: {backend}")
