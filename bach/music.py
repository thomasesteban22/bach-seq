"""
Generacion musical, archivos MIDI, mapa de atencion y estadisticas de las continuaciones.
"""

from __future__ import annotations

import numpy as np
import pretty_midi
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bach import CONTEXT, VOICES

TEMPO = 80  # negras por minuto; cada paso (semicorchea) dura 60 / (4 * TEMPO) s


def generate(model, context, steps: int, temperature: float = 1.0, seed: int | None = None):
    """Envuelve model.generate fijando una semilla aleatoria de muestreo si se indica."""
    gen = None
    if seed is not None:
        gen = torch.Generator(device=next(model.parameters()).device)
        gen.manual_seed(seed)
    return model.generate(context, steps, temperature=temperature, generator=gen).numpy()


def to_midi(notes: np.ndarray, path: str, tempo: int = TEMPO, program: int = 52) -> None:
    """
    Convierte una matriz (T, 4) en MIDI con 4 instrumentos (uno por voz).
    0 = silencio. Notas consecutivas iguales dentro de una voz se fusionan
    (convencion de reproduccion: la matriz no distingue nota sostenida de repetida).
    """
    step = 60.0 / (4 * tempo)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for v, name in enumerate(VOICES):
        inst = pretty_midi.Instrument(program=program, name=name)
        col = notes[:, v]
        t = 0
        while t < len(col):
            pitch, start = int(col[t]), t
            while t < len(col) and col[t] == pitch:
                t += 1
            if pitch > 0:
                inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch,
                                                   start=start * step, end=t * step))
        pm.instruments.append(inst)
    pm.write(path)


def continuation_stats(generated: np.ndarray) -> dict:
    """
    Indicadores simples para describir repeticiones, variedad y silencios:
      - silencios_%: porcentaje de notas generadas que son 0.
      - repeticion_%: porcentaje de pasos identicos al paso anterior (las 4 voces).
      - notas_distintas_por_voz: cuantas alturas diferentes usa cada voz.
      - pasos_distintos: cuantos acordes (filas) distintos aparecen.
    """
    g = np.asarray(generated)
    same_as_prev = (g[1:] == g[:-1]).all(axis=1)
    return {
        "silencios_%": round(100 * float((g == 0).mean()), 1),
        "repeticion_%": round(100 * float(same_as_prev.mean()), 1),
        "notas_distintas_por_voz": {VOICES[v]: int(len(np.unique(g[:, v]))) for v in range(4)},
        "pasos_distintos": int(len({tuple(r) for r in g})),
        "rango_por_voz": {VOICES[v]: f"{int(g[:, v][g[:, v] > 0].min()) if (g[:, v] > 0).any() else 0}-"
                                     f"{int(g[:, v].max())}" for v in range(4)},
    }


def attention_map(model, context, temperature: float = 1.0, seed: int = 42):
    """Genera 16 pasos libres y devuelve los pesos de atencion (16, 32) del bloque."""
    gen = torch.Generator(device=next(model.parameters()).device)
    gen.manual_seed(seed)
    _, blocks = model.generate(context, 16, temperature=temperature, generator=gen,
                               return_attention=True)
    return blocks[0].numpy()


def plot_attention(weights: np.ndarray, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(weights, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xlabel("Paso del contexto (encoder), 1 a 32")
    ax.set_ylabel("Paso generado (decoder), 1 a 16")
    ax.set_xticks(range(0, CONTEXT, 2)); ax.set_xticklabels(range(1, CONTEXT + 1, 2))
    ax.set_yticks(range(weights.shape[0])); ax.set_yticklabels(range(1, weights.shape[0] + 1))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="peso de atencion")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
