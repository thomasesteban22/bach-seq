"""
Configuracion por variables de entorno. Nada de credenciales en el codigo.

  MLFLOW_TRACKING_URI   URL del servidor MLflow (por ejemplo http://<ip>:5000)
  MLFLOW_EXPERIMENT     nombre del experimento propio del equipo
  MODEL_NAME            nombre unico en Model Registry
  DATA_DIR              carpeta donde se descarga el dataset
"""

import os

import mlflow

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "nlp-bach-grupoG")
MODEL_NAME = os.environ.get("MODEL_NAME", "bach-chorales-grupoG")
DATA_DIR = os.environ.get("DATA_DIR", "data")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


def setup_mlflow() -> str:
    """Conecta con el servidor (MLFLOW_TRACKING_URI) y selecciona el experimento."""
    mlflow.set_experiment(EXPERIMENT)
    print(f"MLflow: {mlflow.get_tracking_uri()}  |  experimento: {EXPERIMENT}")
    return EXPERIMENT
