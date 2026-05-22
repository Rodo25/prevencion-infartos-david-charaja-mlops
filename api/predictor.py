"""api/predictor.py — Carga el modelo y ejecuta predicciones."""
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/modelo.pkl"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "artifacts/metrics.json"))

FEATURES = [
    "Edad",
    "Flag_hipertension",
    "Flag_problem_cardiaco",
    "Promedio_nivel_glucosa",
    "IMC",
    "Genero",
    "Estados_civil",
    "Tipo_trabajo",
    "Zona_residencia",
    "Flag_fumador",
]
UMBRAL = 0.25


class Predictor:
    """Wrapper del modelo para uso en FastAPI."""

    def __init__(self):
        self.modelo = None
        self.metricas = {}

    def cargar(self):
        """Carga modelo y métricas desde artifacts/."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. Ejecuta el trainer.")
        with open(MODEL_PATH, "rb") as f:
            self.modelo = pickle.load(f)
        if METRICS_PATH.exists():
            self.metricas = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        log.info("Modelo cargado desde %s", MODEL_PATH)

    def predecir(self, datos: dict) -> dict:
        """Retorna score y decisión binaria."""
        if self.modelo is None:
            self.cargar()
        X = pd.DataFrame([datos])[FEATURES]
        score = float(self.modelo.predict_proba(X)[:, 1][0])
        pred = int(score >= UMBRAL)
        return {
            "score_riesgo": round(score, 4),
            "prediccion": pred,
            "decision": "ALTO RIESGO" if pred == 1 else "BAJO RIESGO",
            "modelo": type(self.modelo).__name__,
            "umbral": UMBRAL,
        }


predictor = Predictor()
