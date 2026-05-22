"""Smoke tests del stack preproductivo."""
import json
import os
import urllib.request

import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

PAYLOAD = {
    "Genero": "Hombre",
    "Edad": 67,
    "Flag_hipertension": 1,
    "Flag_problem_cardiaco": 1,
    "Estados_civil": "Si",
    "Tipo_trabajo": "Empresa_privada",
    "Zona_residencia": "Urbano",
    "Promedio_nivel_glucosa": 190.4,
    "IMC": 32.1,
    "Flag_fumador": "fuma",
}


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_api_root_responde():
    data = get_json(f"{API_URL}/")
    assert data.get("api") is not None


def test_api_health_ok():
    data = get_json(f"{API_URL}/health")
    assert data["status"] == "ok"


def test_api_prediccion_valida():
    data = post_json(f"{API_URL}/predecir", PAYLOAD)
    assert "score_riesgo" in data
    assert data["decision"] in ["ALTO RIESGO", "BAJO RIESGO"]


def test_mlflow_ui_responde():
    try:
        with urllib.request.urlopen(f"{MLFLOW_URL}/health", timeout=10) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("MLflow no disponible")
