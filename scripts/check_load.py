"""
Verificacion en un proceso distinto: resuelve champion -> version, carga
models:/<MODEL_NAME>/<version> desde MLflow (no un archivo local) y genera 33 pasos.
Ejecutar desde una carpeta fuera del repositorio para comprobar que el modelo
trae su propio codigo (code_paths):
  cd /tmp && python ~/bach-seq/scripts/check_load.py
"""

import os

import mlflow.pytorch
from mlflow import MlflowClient

name = os.environ.get("MODEL_NAME", "bach-chorales-grupoG")
mv = MlflowClient().get_model_version_by_alias(name, "champion")
model = mlflow.pytorch.load_model(f"models:/{name}/{mv.version}", map_location="cpu")
model.eval()
out = model.generate([[60, 55, 52, 48]] * 32, 33)
print(f"Cargado {name} v{mv.version} (run {mv.run_id}) "
      f"[{type(model).__module__}.{type(model).__name__}]; generado {tuple(out.shape)}, "
      f"rango {int(out.min())}-{int(out.max())}")
