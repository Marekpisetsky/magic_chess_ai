# Agente Autónomo de Magic Chess (Estilo Exocomp)
## Documento Técnico — Versión 1.0

---

## 1. Objetivo General

Construir un **agente autónomo** capaz de jugar Magic Chess de forma profesional, utilizando:

- Un **modelo local** de visión especializado en el HUD del juego.
- Una arquitectura modular inspirada en los **Exocomps** de Star Trek:
  - *Percibir → Entender → Decidir → Actuar → Mejorar*.

El sistema debe ser:

- **Autónomo** (sin depender de una API durante la partida).
- **Preciso** (lecturas del HUD con ≥ 98 % de exactitud en test).
- **Extensible** (HUD hoy, tablero y sinergias mañana).
- **Mantenible** (arquitectura limpia y versionada).

---

## 2. Arquitectura General del Agente

```text
                   +-----------------+
                   |  Captura Frame  |
                   +--------+--------+
                            |
                            v
                 +----------+-----------+
                 |   Percepción Visual  |  <-- Modelo Local (HUD Engine)
                 +----------+-----------+
                            |
                            v
                 +----------+-----------+
                 |   Estado del Juego   |  <-- core/state.py
                 +----------+-----------+
                            |
                            v
                 +----------+-----------+
                 |  Política de Decisión|  <-- advisor.py / decision.py
                 +----------+-----------+
                            |
                            v
                 +----------+-----------+
                 |   Motor de Acciones  |
                 +----------------------+


Módulos:

Captura

core/capture.py obtiene frames del juego.

Percepción Visual

vlm_api.py: interfaz con la API de visión (teacher).

tools/generate_labels_with_teacher.py: genera labels.jsonl.

models/hud_model.py: modelo local (student).

datasets/hud_dataset.py: dataset de entrenamiento.

train_hud_model.py: entrenamiento.

core/hud_local_reader.py: inferencia local en tiempo real.

Estado del Juego

core/state.py mantiene oro, vida, ronda, nivel y más.

Política de Decisión

advisor.py, core/decision.py definen la acción óptima según el estado.

Motor de Acciones

Lógica que traduce acciones a clicks/inputs dentro del juego.

3. Percepción Visual — “Visual HUD Engine v1”
3.1 Objetivo

Leer, a partir de la captura de pantalla, los valores:

round (ej. "1-1", "2-3")

level (entero 1–9)

gold (entero 0–100)

hp_self (entero 0–100)

con un modelo local rápido y robusto.

3.2 Dataset

Guardar capturas en data/raw_frames/ (frame_0001.png, frame_0002.png, …).

Ejecutar tools/generate_labels_with_teacher.py:

Cada imagen se envía a la API (teacher).

Se genera data/labels.jsonl con:

{"image": "frame_0001.png", "round": "1-1", "level": 3, "gold": 2, "hp_self": 100}

Este fichero es la semilla de entrenamiento del modelo local.

3.3 Modelo Local

Backbone: ResNet18 preentrenada (torchvision).

Cuatro “cabezas” de clasificación:

round → clases de rondas vistas en el dataset.

level → clases 0–9 (0 = desconocido).

gold → clases 0–100.

hp_self → clases 0–100.

Loss total: suma de CrossEntropyLoss por cada cabeza.

Entrenamiento:

10–30 epochs.

LR inicial 1e-4.

Optimizer Adam.

Data augmentation ligera (resize, normalize, pequeñas variaciones).

El modelo entrenado se guarda como:

hud_model.pt

junto con el vocabulario de rondas.

4. Estado del Juego (core/state.py)

Responsabilidades:

Mantener una representación consistente del juego:

round, level, gold, hp_self.

Historial de rondas.

Info de tablero (extensión futura).

API limpia, orientada a la política:

update_from_hud(hud_dict)

get_economy_state()

get_hp_state()

get_level_state()

No decide, solo entiende y expone.

5. Política de Decisión (advisor / decision)

Dos niveles:

Reglas Base (v1)

Política basada en heurísticas de auto-battlers:

Gestión de oro.

Subida de nivel en rondas clave.

Compra/venta básica de unidades.

Política Aprendida (v2)

A futuro: modelo entrenado con datos de partidas:

Imitación de jugadores humanos.

RL para maximizar winrate.

Entrada: estado del juego.
Salida: acción (buy, sell, level_up, reroll, do_nothing, …) + razón opcional.

6. Motor de Acciones

Traduce la acción de alto nivel en interacciones reales:

Click en tienda.

Click en subir nivel.

Drag & drop de unidades (futuro).

Debe ser:

Determinista.

“Seguro” (no clickear fuera de rango).

Loggable (para reproducir partidas).

7. Aprendizaje Continuo — Ciclo Estilo Exocomp

Ciclo:

El agente juega partidas usando el modelo local.

Se registran:

Frames,

Estado,

Acciones,

Resultado (victoria/derrota).

Se detectan fallos de visión (HUD mal leído) o decisiones débiles.

Se vuelve a etiquetar con el teacher solo donde falla.

Se reentrena el modelo local (hud_model_v2.pt, hud_model_v3.pt, …).

Con cada iteración el sistema se automejora.

8. Roadmap

HUD Engine v1

Dataset inicial.

Entrenamiento + evaluación.

Integración en realtime_loop.py.

HUD Engine v2

Más datos (incluyendo casos difíciles).

Data augmentation más rica.

Métricas y test más estrictos.

Tablero Engine v1

Detección de héroes y sinergias.

Decision Engine v1

Política por reglas.

Decision Engine v2

Política aprendida (supervisado / RL).

Agente Público

Demo,

Post técnico,

Repositorio limpio y documentado.

9. Visión Final

Lograr un agente que, igual que un Exocomp:

Observe el entorno,

Entienda el contexto,

Tome decisiones útiles,

Aprenda de sus propias experiencias,

hasta competir con jugadores humanos de alto nivel en Magic Chess.

Este documento define la base técnica y la dirección del proyecto.

