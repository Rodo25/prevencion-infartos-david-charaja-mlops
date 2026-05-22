"""Predicción rápida desde terminal."""
import json
import pickle

import pandas as pd

from config import MODEL_PATH, FEATURES

EJEMPLO = {
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


if __name__ == "__main__":
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    X = pd.DataFrame([EJEMPLO])[FEATURES]
    score = float(model.predict_proba(X)[:, 1][0])
    pred = int(score >= 0.50)
    print(json.dumps({"score_riesgo": round(score, 4), "prediccion": pred}, indent=2))
