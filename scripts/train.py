"""
Entrena los experimentos E1-E4 y los registra en MLflow.

Uso:
  python scripts/train.py                      # E1, E2, E3 y luego E4
  python scripts/train.py --runs E1 E2 E3      # solo los tres de la guia
  python scripts/train.py --runs E4 --e4-param hidden --e4-value 256

E4 toma la arquitectura con menor perdida de validacion entre E1-E3 (consultando
MLflow) y cambia UN solo hiperparametro. Por defecto: hidden 128 -> 256.
"""

import argparse
import os
import tempfile

from _common import load_data

import mlflow

from bach.config import setup_mlflow
from bach.music import attention_map, plot_attention
from bach.runs import best_run, experiment_runs
from bach.training import DEFAULTS, run_experiment

BASE = {
    "E1": ({"cell": "lstm", "attention": False}, "Encoder-decoder LSTM"),
    "E2": ({"cell": "gru", "attention": False}, "Encoder-decoder GRU"),
    "E3": ({"cell": "gru", "attention": True}, "Encoder-decoder GRU con atencion aditiva"),
}
CASTS = {"hidden": int, "emb_dim": int, "layers": int, "batch_size": int, "lr": float}


def make_attention_fn(splits):
    """Mapa de atencion del run E3 sobre los primeros 32 pasos del primer coral de valid."""
    chorale = splits["valid"][0]

    def fn(model, folder):
        w = attention_map(model, chorale.notes[:32], temperature=1.0, seed=42)
        path = os.path.join(folder, "attention_map_valid.png")
        plot_attention(w, path, f"E3: atencion en generacion libre ({chorale.name}, semilla 42, T=1.0)")
        return path
    return fn


def run_e4(windows, splits, param, value):
    df = experiment_runs()
    best = best_run(df, only=["E1", "E2", "E3"])
    base_id = best["tags.lab_id"]
    cfg = dict(BASE[base_id][0])
    old = DEFAULTS[param]
    cfg[param] = value
    desc = (f"Arquitectura de {base_id} (menor val_loss entre E1-E3: "
            f"{best['metrics.best_val_loss']:.4f}) reentrenada desde cero cambiando "
            f"{param}: {old} -> {value}.")
    print(desc)
    attn = make_attention_fn(splits) if cfg["attention"] else None
    return run_experiment("E4", cfg, windows, description=desc, attention_fn=attn,
                          extra_tags={"base_run": base_id, "changed_param": param,
                                      "changed_from": str(old), "changed_to": str(value)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["E1", "E2", "E3", "E4"])
    ap.add_argument("--e4-param", default="hidden", choices=list(CASTS))
    ap.add_argument("--e4-value", default="256")
    args = ap.parse_args()

    setup_mlflow()
    splits, windows = load_data()
    for name in args.runs:
        if name in BASE:
            cfg, desc = BASE[name]
            attn = make_attention_fn(splits) if cfg["attention"] else None
            run_experiment(name, cfg, windows, description=desc, attention_fn=attn)
        elif name == "E4":
            run_e4(windows, splits, args.e4_param, CASTS[args.e4_param](args.e4_value))
        else:
            raise SystemExit(f"Run desconocido: {name}")


if __name__ == "__main__":
    main()
