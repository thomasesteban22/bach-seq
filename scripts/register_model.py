"""
Registra el modelo elegido en Model Registry con alias "champion".

- Se restauran los pesos del mejor checkpoint del run elegido.
- mlflow.pytorch.log_model se ejecuta dentro del MISMO run de entrenamiento, para
  que la version quede vinculada a ese run.
- Se guarda el objeto PyTorch completo (con su metodo generate). Si la version de
  MLflow ofrece serialization_format, se usa "pickle", como pide la guia.
- code_paths incluye el paquete bach/, asi otro proceso puede importar Seq2Seq.
"""

import inspect
import json
import os

from _common import ROOT

import mlflow
import mlflow.pytorch
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature

from bach.config import MODEL_NAME, RESULTS_DIR, setup_mlflow
from bach.runs import load_run_checkpoint


def main():
    setup_mlflow()
    with open(os.path.join(RESULTS_DIR, "selection.json")) as f:
        sel = json.load(f)
    model, _ = load_run_checkpoint(sel["run_id"])
    model.cpu().eval()

    # Firma de entrada (lote, 32, 4) enteros. No se pasa input_example porque forward()
    # necesita tambien Y (teacher forcing); la generacion se hace con model.generate().
    example = np.zeros((1, 32, 4), dtype=np.int64)
    extra = {}
    if "serialization_format" in inspect.signature(mlflow.pytorch.log_model).parameters:
        extra["serialization_format"] = "pickle"

    with mlflow.start_run(run_id=sel["run_id"]):
        info = mlflow.pytorch.log_model(
            model, artifact_path="model",
            code_paths=[os.path.join(ROOT, "bach")],
            registered_model_name=MODEL_NAME,
            signature=infer_signature(example),
            **extra,
        )
    client = MlflowClient()
    version = max(client.search_model_versions(f"name='{MODEL_NAME}'"),
                  key=lambda v: int(v.version))
    assert version.run_id == sel["run_id"], "La version no quedo vinculada al run elegido"
    client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
    client.set_model_version_tag(MODEL_NAME, version.version, "lab_id", sel["selected_run"])
    print(f"Registrado {MODEL_NAME} v{version.version} (run {sel['run_id']}), alias champion")
    print("model_uri:", info.model_uri)


if __name__ == "__main__":
    main()
