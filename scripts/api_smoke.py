"""
Prueba de la API desplegada (o local) por HTTP.
  python scripts/api_smoke.py http://<host>:8000
Imprime el codigo y un resumen de cada respuesta: valida, steps=33 e invalidas.
"""

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
SEED = [[74, 70, 65, 58]] * 8 + [[75, 70, 67, 60]] * 8 + [[74, 70, 65, 58]] * 8 + [[72, 69, 65, 53]] * 8


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


cases = [
    ("GET /health", "GET", "/health", None),
    ("valida, steps=16", "POST", "/generate", {"seed": SEED, "steps": 16, "temperature": 1.0}),
    ("steps=33", "POST", "/generate", {"seed": SEED, "steps": 33}),
    ("invalida: nota 200", "POST", "/generate", {"seed": [[200, 70, 65, 58]] + SEED[1:], "steps": 8}),
    ("invalida: 31 pasos", "POST", "/generate", {"seed": SEED[:31], "steps": 8}),
    ("invalida: steps=65", "POST", "/generate", {"seed": SEED, "steps": 65}),
    ("invalida: temperature=2", "POST", "/generate", {"seed": SEED, "steps": 8, "temperature": 2}),
]
for label, method, path, body in cases:
    code, resp = call(method, path, body)
    if "generated" in resp:
        g = resp["generated"]
        summary = f"generated {len(g)}x{len(g[0])}, primeros 2: {g[:2]}, model={resp['model']}, version={resp['version']!r}"
    elif "detail" in resp and isinstance(resp["detail"], list):
        d = resp["detail"][0]
        summary = f"{d['type']} en {d['loc']}: {d['msg']}"
    else:
        summary = resp
    print(f"{label:26s} -> {code}  {summary}")
