from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _clamp_int(val: Any, minimum: int, maximum: int) -> int:
    try:
        n = int(val)
    except Exception:
        return minimum
    return max(minimum, min(maximum, n))


@dataclass
class ShopHero:
    """Heroe que aparece en la tienda (slot_index 0-4)."""

    slot_index: int
    nombre: str
    coste: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_index": int(self.slot_index),
            "nombre": self.nombre,
            "coste": int(self.coste),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ShopHero":
        return ShopHero(
            slot_index=int(data.get("slot_index", 0)),
            nombre=str(data.get("nombre", "")),
            coste=int(data.get("coste", 0)),
        )


@dataclass
class BoardUnit:
    """Unidad colocada en tablero/banco."""

    fila: int
    columna: int
    nombre: str
    estrellas: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fila": int(self.fila),
            "columna": int(self.columna),
            "nombre": self.nombre,
            "estrellas": int(self.estrellas),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BoardUnit":
        return BoardUnit(
            fila=int(data.get("fila", 0)),
            columna=int(data.get("columna", 0)),
            nombre=str(data.get("nombre", "")),
            estrellas=int(data.get("estrellas", 1)),
        )


@dataclass
class GameState:
    """
    Estado detallado de una ronda de partida.
    Este modelo lo comparten el VLM, el motor de decisiones y la base de conocimiento.
    """

    fase: str
    ronda: int
    round_label: str = ""
    oro: int
    vida: int
    nivel_tablero: int
    xp_actual: int
    tienda: List[ShopHero] = field(default_factory=list)
    tablero: List[BoardUnit] = field(default_factory=list)
    banco: List[BoardUnit] = field(default_factory=list)
    sinergias_activas: Dict[str, int] = field(default_factory=dict)
    sinergias_potenciales: Dict[str, int] = field(default_factory=dict)
    comandante: str = ""
    emblema: str = ""
    confianza_lectura: float = 0.0
    tienda_abierta: bool = False

    def __post_init__(self) -> None:
        # Sanitiza valores numericos basicos para evitar rangos absurdos.
        self.fase = self.fase or "unknown"
        self.ronda = _clamp_int(self.ronda, 0, 200)
        self.round_label = (self.round_label or "").strip()
        self.oro = _clamp_int(self.oro, 0, 200)
        self.vida = _clamp_int(self.vida, 0, 100)
        self.nivel_tablero = _clamp_int(self.nivel_tablero, 1, 15)
        self.xp_actual = _clamp_int(self.xp_actual, 0, 100)
        self.confianza_lectura = max(0.0, min(1.0, float(self.confianza_lectura)))
        self.tienda_abierta = bool(self.tienda_abierta)

        # Sanea lista de tienda y unidades
        sane_shop: List[ShopHero] = []
        for h in self.tienda:
            try:
                sane_shop.append(
                    ShopHero(
                        slot_index=_clamp_int(getattr(h, "slot_index", 0), 0, 4),
                        nombre=str(getattr(h, "nombre", "")),
                        coste=_clamp_int(getattr(h, "coste", 0), 0, 10),
                    )
                )
            except Exception:
                continue
        self.tienda = sane_shop

        def _sanitize_units(units: List[BoardUnit]) -> List[BoardUnit]:
            out: List[BoardUnit] = []
            for u in units:
                try:
                    out.append(
                        BoardUnit(
                            fila=_clamp_int(getattr(u, "fila", 0), 0, 7),
                            columna=_clamp_int(getattr(u, "columna", 0), 0, 9),
                            nombre=str(getattr(u, "nombre", "")),
                            estrellas=_clamp_int(getattr(u, "estrellas", 1), 1, 3),
                        )
                    )
                except Exception:
                    continue
            return out

        self.tablero = _sanitize_units(self.tablero)
        self.banco = _sanitize_units(self.banco)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fase": self.fase,
            "ronda": int(self.ronda),
            "round_label": self.round_label,
            "oro": int(self.oro),
            "vida": int(self.vida),
            "nivel_tablero": int(self.nivel_tablero),
            "xp_actual": int(self.xp_actual),
            "tienda": [h.to_dict() for h in self.tienda],
            "tablero": [u.to_dict() for u in self.tablero],
            "banco": [u.to_dict() for u in self.banco],
            "sinergias_activas": dict(self.sinergias_activas),
            "sinergias_potenciales": dict(self.sinergias_potenciales),
            "comandante": self.comandante,
            "emblema": self.emblema,
            "confianza_lectura": float(self.confianza_lectura),
            "tienda_abierta": bool(self.tienda_abierta),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        tienda_raw = data.get("tienda") or []
        tablero_raw = data.get("tablero") or []
        banco_raw = data.get("banco") or []

        return cls(
            fase=str(data.get("fase", "early")),
            ronda=int(data.get("ronda", 0)),
            round_label=str(data.get("round_label") or ""),
            oro=int(data.get("oro", 0)),
            vida=int(data.get("vida", 100)),
            nivel_tablero=int(data.get("nivel_tablero", 1)),
            xp_actual=int(data.get("xp_actual", 0)),
            tienda=[ShopHero.from_dict(h) for h in tienda_raw if isinstance(h, dict)],
            tablero=[BoardUnit.from_dict(u) for u in tablero_raw if isinstance(u, dict)],
            banco=[BoardUnit.from_dict(u) for u in banco_raw if isinstance(u, dict)],
            sinergias_activas=dict(data.get("sinergias_activas") or {}),
            sinergias_potenciales=dict(data.get("sinergias_potenciales") or {}),
            comandante=str(data.get("comandante") or ""),
            emblema=str(data.get("emblema") or ""),
            confianza_lectura=float(data.get("confianza_lectura", 0.0)),
            tienda_abierta=bool(data.get("tienda_abierta", False)),
        )

    @classmethod
    def from_json(cls, raw: str) -> "GameState":
        return cls.from_dict(json.loads(raw))

    # --------- Actualizacion y featurizacion ---------

    def update_from_hud(self, hud: Dict[str, Any]) -> None:
        """
        Actualiza campos basicos con datos leidos por OCR/modelo local.
        Espera claves: round, level, gold, hp_self.
        """
        if hud.get("round") is not None:
            self.round_label = str(hud["round"])
        if hud.get("level") is not None:
            self.nivel_tablero = _clamp_int(hud["level"], 1, 15)
        if hud.get("gold") is not None:
            self.oro = _clamp_int(hud["gold"], 0, 200)
        if hud.get("hp_self") is not None:
            self.vida = _clamp_int(hud["hp_self"], 0, 100)

    def _round_to_nums(self) -> tuple[int, int]:
        """
        Convierte round_label tipo '2-3' -> (2, 3).
        Si no hay round_label, devuelve (0, 0).
        """
        label = (self.round_label or "").strip()
        if "-" in label:
            try:
                a, b = label.split("-")
                return int(a), int(b)
            except Exception:
                return (0, 0)
        return (0, 0)

    def to_vector(self) -> List[float]:
        """
        Representacion numerica simple y normalizada del estado.
        Se puede extender con mas features (sinergias, tamano de tablero, etc.).
        """
        r1, r2 = self._round_to_nums()
        level_norm = self.nivel_tablero / 9.0
        gold_norm = min(self.oro, 100) / 100.0
        hp_norm = self.vida / 100.0
        xp_norm = min(self.xp_actual, 20) / 20.0
        r1_norm = r1 / 10.0
        r2_norm = r2 / 10.0
        return [
            level_norm,
            gold_norm,
            hp_norm,
            xp_norm,
            r1_norm,
            r2_norm,
        ]


@dataclass
class PlayerStatus:
    """Estado de un jugador en la partida."""
    name: str
    hp: Optional[int] = None
    gold: Optional[int] = None
    level: Optional[int] = None
    is_local: bool = False


@dataclass
class ScreenState:
    """
    Estado de una captura de pantalla:
    - tipo de pantalla (tablero propio, enemigo, tienda, etc.)
    - ronda
    - jugador local
    - resto de jugadores
    """
    screen_type: str
    round_label: Optional[str]
    you: PlayerStatus
    others: List[PlayerStatus]
