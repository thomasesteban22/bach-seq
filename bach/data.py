"""
Carga de JSB Chorales y construccion de ventanas 32 -> 16.

Reglas de la guia que se cumplen aqui:
  - Se conservan las particiones train / valid / test del dataset.
  - Vocabulario fijo 0..127 sin recodificar; 0 (silencio) es una categoria mas.
  - Ventanas completas de 48 pasos dentro de cada coral: 32 de X y 16 de Y.
  - Desplazamiento 8 en train y 16 en valid y test; se descarta el resto final.
  - Ninguna ventana cruza el limite de un coral ni cambia de particion.
"""

from __future__ import annotations

import glob
import os
import tarfile
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from bach import CONTEXT, HORIZON, VOICES

DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/jsb_chorales/jsb_chorales.tgz"
)
STRIDES = {"train": 8, "valid": 16, "test": 16}


def download(data_dir: str = "data") -> str:
    """Descarga y descomprime el dataset si no existe. Devuelve la carpeta raiz."""
    root = os.path.join(data_dir, "jsb_chorales")
    if os.path.isdir(os.path.join(root, "train")):
        return root
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(data_dir, "jsb_chorales.tgz")
    if not os.path.exists(tgz):
        urllib.request.urlretrieve(DATA_URL, tgz)
    with tarfile.open(tgz) as tar:
        tar.extractall(root)
    return root


@dataclass
class Chorale:
    name: str
    notes: np.ndarray  # (T, 4) int64


def load_split(root: str, split: str) -> list[Chorale]:
    """Carga todos los corales de una particion, en orden por nombre de archivo."""
    files = sorted(glob.glob(os.path.join(root, split, "*.csv")))
    chorales = []
    for f in files:
        df = pd.read_csv(f)
        if list(df.columns) != VOICES:
            raise ValueError(f"{f}: columnas inesperadas {list(df.columns)}")
        notes = df[VOICES].to_numpy(dtype=np.int64)
        if notes.min() < 0 or notes.max() > 127:
            raise ValueError(f"{f}: valores fuera de 0..127")
        chorales.append(Chorale(os.path.splitext(os.path.basename(f))[0], notes))
    return chorales


def load_all(root: str) -> dict[str, list[Chorale]]:
    return {split: load_split(root, split) for split in ("train", "valid", "test")}


def describe(splits: dict[str, list[Chorale]]) -> pd.DataFrame:
    """Resumen para verificar el dataset: corales, longitudes y rango por voz."""
    rows = []
    for split, chorales in splits.items():
        lengths = [len(c.notes) for c in chorales]
        allnotes = np.concatenate([c.notes for c in chorales])
        row = {"particion": split, "corales": len(chorales),
               "long_min": min(lengths), "long_max": max(lengths),
               "silencios_%": round(100 * (allnotes == 0).mean(), 2)}
        for v, name in enumerate(VOICES):
            nz = allnotes[:, v][allnotes[:, v] > 0]
            row[f"{name}_rango"] = f"{nz.min()}-{nz.max()}"
        rows.append(row)
    return pd.DataFrame(rows)


def make_windows(chorales: list[Chorale], stride: int):
    """
    Corta ventanas de 48 pasos dentro de cada coral.

    Devuelve X (N, 32, 4), Y (N, 16, 4) y el origen de cada ventana
    (nombre del coral, paso inicial), util para rastrear ejemplos.
    """
    size = CONTEXT + HORIZON
    xs, ys, origin = [], [], []
    for c in chorales:
        for start in range(0, len(c.notes) - size + 1, stride):
            w = c.notes[start:start + size]
            xs.append(w[:CONTEXT])
            ys.append(w[CONTEXT:])
            origin.append((c.name, start))
    return np.stack(xs), np.stack(ys), origin


def build_windows(splits: dict[str, list[Chorale]]):
    """Ventanas de las tres particiones, construidas una sola vez y compartidas por E1-E4."""
    return {s: make_windows(splits[s], STRIDES[s]) for s in ("train", "valid", "test")}


def make_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool,
                seed: int | None = None) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(seed if seed is not None else 0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)
