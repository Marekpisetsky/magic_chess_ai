# core/knowledge.py
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import DB_PATH


class KnowledgeBase:
    """
    Memoria persistente:
    - game_entities: héroes, sinergias, comandantes, emblemas, cartas, eventos
    - matches: partidas
    - rounds: rondas con GameState serializado
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()

        # Entidades del juego (memoria "eterna")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS game_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tipo TEXT NOT NULL,        -- heroe, sinergia, comandante, emblema, carta, evento
            nombre TEXT NOT NULL UNIQUE,
            descripcion_larga TEXT,
            metadatos_json TEXT,
            estado_conocimiento TEXT DEFAULT 'desconocido', -- desconocido, parcial, completo
            primera_vez_visto TEXT,
            ultima_vez_visto TEXT
        )
        """)

        # Partidas
        cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_inicio TEXT,
            timestamp_fin TEXT,
            comandante TEXT,
            emblema TEXT,
            resultado_posicion INTEGER,
            resultado_vida INTEGER
        )
        """)

        # Rondas por partida
        cur.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            ronda INTEGER,
            fase TEXT,
            game_state_json TEXT,
            recomendaciones_json TEXT,
            acciones_realizadas_json TEXT,
            FOREIGN KEY(match_id) REFERENCES matches(id)
        )
        """)

        self.conn.commit()

    # ---------- ENTIDADES ----------

    def upsert_entity(self, nombre: str, tipo: str,
                      descripcion_larga: Optional[str] = None,
                      metadatos_json: Optional[str] = None,
                      estado_conocimiento: Optional[str] = None):
        cur = self.conn.cursor()

        cur.execute("""
        INSERT INTO game_entities (nombre, tipo, descripcion_larga, metadatos_json, estado_conocimiento,
                                   primera_vez_visto, ultima_vez_visto)
        VALUES (?, ?, ?, ?, COALESCE(?, 'parcial'), datetime('now'), datetime('now'))
        ON CONFLICT(nombre) DO UPDATE SET
            tipo = excluded.tipo,
            descripcion_larga = COALESCE(excluded.descripcion_larga, game_entities.descripcion_larga),
            metadatos_json = COALESCE(excluded.metadatos_json, game_entities.metadatos_json),
            estado_conocimiento = COALESCE(excluded.estado_conocimiento, game_entities.estado_conocimiento),
            ultima_vez_visto = datetime('now')
        """, (nombre, tipo, descripcion_larga, metadatos_json, estado_conocimiento))
        self.conn.commit()

    def get_entity(self, nombre: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM game_entities WHERE nombre = ?", (nombre,))
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    # ---------- PARTIDAS / RONDAS ----------

    def start_match(self) -> int:
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO matches (timestamp_inicio) VALUES (datetime('now'))
        """)
        self.conn.commit()
        return cur.lastrowid

    def end_match(self, match_id: int, posicion: int, vida: int):
        cur = self.conn.cursor()
        cur.execute("""
        UPDATE matches
        SET timestamp_fin = datetime('now'),
            resultado_posicion = ?,
            resultado_vida = ?
        WHERE id = ?
        """, (posicion, vida, match_id))
        self.conn.commit()

    def add_round(self, match_id: int, ronda: int, fase: str,
                  game_state_json: str,
                  recomendaciones_json: str,
                  acciones_realizadas_json: Optional[str] = None):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO rounds (match_id, ronda, fase, game_state_json, recomendaciones_json, acciones_realizadas_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (match_id, ronda, fase, game_state_json, recomendaciones_json, acciones_realizadas_json))
        self.conn.commit()
