"""
Pruebas de la API contra el modelo champion real (necesita MLFLOW_TRACKING_URI).
  pytest -v tests/test_api.py
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app  # noqa: E402

SEED = [[74, 70, 65, 58]] * 8 + [[75, 70, 67, 60]] * 8 + [[74, 70, 65, 58]] * 8 + [[72, 69, 65, 53]] * 8


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # ejecuta el lifespan: carga champion
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"] and isinstance(body["version"], str)


def test_generate_valid(client):
    r = client.post("/generate", json={"seed": SEED, "steps": 16, "temperature": 1.0})
    assert r.status_code == 200
    g = r.json()["generated"]
    assert len(g) == 16 and all(len(s) == 4 for s in g)
    assert all(isinstance(n, int) and 0 <= n <= 127 for s in g for n in s)
    assert isinstance(r.json()["version"], str)


def test_generate_33_steps(client):
    """33 pasos obliga a 3 bloques (16 + 16 + 1); no incluye el contexto."""
    r = client.post("/generate", json={"seed": SEED, "steps": 33})
    assert r.status_code == 200
    assert len(r.json()["generated"]) == 33


@pytest.mark.parametrize("payload", [
    {"seed": SEED[:31], "steps": 8},                                  # 31 pasos de contexto
    {"seed": [s + [60] for s in SEED], "steps": 8},                   # 5 voces
    {"seed": [[128, 70, 65, 58]] + SEED[1:], "steps": 8},             # nota > 127
    {"seed": [[-1, 70, 65, 58]] + SEED[1:], "steps": 8},              # nota negativa
    {"seed": [[60.5, 70, 65, 58]] + SEED[1:], "steps": 8},            # nota no entera
    {"seed": SEED, "steps": 0},                                       # steps < 1
    {"seed": SEED, "steps": 65},                                      # steps > 64
    {"seed": SEED, "steps": 8, "temperature": 2.0},                   # temperatura fuera de rango
    {"steps": 8},                                                     # falta seed
])
def test_generate_invalid(client, payload, monkeypatch):
    called = []
    from api import main
    monkeypatch.setattr(main.state["model"], "generate",
                        lambda *a, **k: called.append(1))
    r = client.post("/generate", json=payload)
    assert r.status_code == 422
    assert not called, "El modelo no debe llamarse con una entrada invalida"
