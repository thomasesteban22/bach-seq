"""Consultas a MLflow: tabla de experimentos, seleccion del modelo y carga de checkpoints."""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from bach.config import EXPERIMENT
from bach.training import load_checkpoint


def experiment_runs() -> pd.DataFrame:
    """Runs de entrenamiento terminados del experimento, con sus metricas clave."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"No existe el experimento {EXPERIMENT}")
    df = mlflow.search_runs([exp.experiment_id], filter_string="attributes.status = 'FINISHED'")
    df = df[df.get("metrics.best_val_loss").notna()] if "metrics.best_val_loss" in df else df.iloc[0:0]
    return df.sort_values("start_time")


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla para el README: configuracion, mejor epoca, perdida, exactitudes, parametros y tiempo."""
    t = pd.DataFrame({
        "Run": df["tags.lab_id"],
        "Celda": df["params.cell"].str.upper(),
        "Atencion": df["params.attention"],
        "Oculta": df["params.hidden_dim"],
        "Emb": df["params.embedding_dim"],
        "lr": df["params.learning_rate"],
        "Mejor epoca": df["metrics.best_epoch"].astype(int),
        "Val loss": df["metrics.best_val_loss"].round(4),
        "Val acc nota": df["metrics.val_note_acc"].round(4),
        "Val acc paso": df["metrics.val_step_acc"].round(4),
        "Parametros": df["metrics.trainable_params"].astype(int),
        "Tiempo (s)": df["metrics.train_time_s"].round(0).astype(int),
        "run_id": df["run_id"],
    })
    return t.reset_index(drop=True)


def best_run(df: pd.DataFrame, only: list[str] | None = None) -> pd.Series:
    """Run con menor perdida de validacion (opcionalmente restringido a ciertos lab_id)."""
    if only is not None:
        df = df[df["tags.lab_id"].isin(only)]
    if df.empty:
        raise RuntimeError("No hay runs terminados para seleccionar.")
    return df.loc[df["metrics.best_val_loss"].idxmin()]


def run_by_lab_id(lab_id: str) -> pd.Series:
    df = experiment_runs()
    df = df[df["tags.lab_id"] == lab_id]
    if df.empty:
        raise RuntimeError(f"No hay un run terminado con lab_id={lab_id}")
    return df.iloc[-1]


def load_run_checkpoint(run_id: str, device="cpu"):
    """Descarga el mejor checkpoint de un run y reconstruye el modelo."""
    path = MlflowClient().download_artifacts(run_id, "checkpoint/best_checkpoint.pt")
    return load_checkpoint(path, device)
