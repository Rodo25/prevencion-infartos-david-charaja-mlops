"""Configuración central del proyecto."""
from pathlib import Path

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reportes")

RAW_DATA_PATH = DATA_DIR / "dataset_prevencion_infartos.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = ARTIFACTS_DIR / "modelo.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / "classification_report.txt"

TARGET = "Ataque_cardiaco"
ID_COL = "ID"

NUMERIC_FEATURES = [
    "Edad",
    "Flag_hipertension",
    "Flag_problem_cardiaco",
    "Promedio_nivel_glucosa",
    "IMC",
]

CATEGORICAL_FEATURES = [
    "Genero",
    "Estados_civil",
    "Tipo_trabajo",
    "Zona_residencia",
    "Flag_fumador",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20
