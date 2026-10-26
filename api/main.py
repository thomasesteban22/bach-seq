"""
API de generacion de corales.

Al iniciar, resuelve el alias "champion" del Model Registry a un numero de version
y carga ESA version concreta (models:/<MODEL_NAME>/<version>). Las solicitudes no
vuelven a consultar MLflow; si el alias cambia, se reinicia el servicio.

Variables de entorno (nunca en el repositorio):
  MLFLOW_TRACKING_URI       servidor MLflow, por ejemplo http://<ip>:5000
  MLFLOW_TRACKING_USERNAME  usuario de MLflow, si el servidor usa autenticacion
  MLFLOW_TRACKING_PASSWORD  contrasena de MLflow, si el servidor usa autenticacion
  MODEL_NAME                nombre registrado (por defecto bach-chorales-grupoG)

Endpoints:
  GET  /health    -> {"status": "ok", "model": ..., "version": ...}
  POST /generate  -> {"generated": [[...4 notas...] x steps], "model": ..., "version": "..."}
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field, conlist

MODEL_NAME = os.environ.get("MODEL_NAME", "bach-chorales-grupoG")
ALIAS = "champion"

state: dict = {}

Note = Annotated[int, Field(ge=0, le=127, strict=True)]
Step = conlist(Note, min_length=4, max_length=4)


class GenerateRequest(BaseModel):
    seed: conlist(Step, min_length=32, max_length=32) = Field(
        ..., description="32 pasos de contexto, cada uno con 4 notas MIDI enteras 0..127 (0 = silencio)")
    steps: int = Field(..., ge=1, le=64, strict=True, description="Pasos nuevos a generar (1..64)")
    temperature: float = Field(1.0, ge=0.5, le=1.5, description="Temperatura de muestreo (0.5..1.5)")


class GenerateResponse(BaseModel):
    generated: list[list[int]]
    model: str
    version: str


def load_champion():
    """Resuelve champion -> version y carga esa version concreta en CPU."""
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version}", map_location="cpu")
    model.eval()
    return model, str(version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"], state["version"] = load_champion()
    print(f"Modelo cargado: {MODEL_NAME} v{state['version']} (alias {ALIAS})", flush=True)
    yield
    state.clear()


app = FastAPI(title="Bach chorales API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"status": "ok", "model": MODEL_NAME, "version": state["version"]}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # La validacion de pydantic ya rechazo (422) cualquier entrada invalida:
    # forma distinta de 32x4, notas fuera de 0..127 o no enteras, steps fuera de 1..64,
    # temperatura fuera de 0.5..1.5. Aqui el modelo solo recibe entradas validas.
    with torch.no_grad():
        out = state["model"].generate(req.seed, req.steps, temperature=req.temperature)
    return GenerateResponse(generated=out.tolist(), model=MODEL_NAME, version=state["version"])
