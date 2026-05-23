"""src/02_train_model.py — Entrena modelo, registra métricas y versiona con MLflow."""
import json
import logging
import os
import pickle
from pathlib import Path

try:
    import mlflow
    import mlflow.sklearn
except Exception:  # permite ejecutar tests aunque MLflow no esté instalado localmente
    mlflow = None
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    ARTIFACTS_DIR,
    CLASSIFICATION_REPORT_PATH,
    FEATURES,
    FEATURES_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TARGET,
    TRAIN_PATH,
    TEST_PATH,
)

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | TRAIN | %(message)s", 
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Crea pipeline sklearn con imputación, encoding y modelo balanceado."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcula métricas para clasificación binaria desbalanceada."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.25).astype(int)
    return {
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "roc_auc": round(roc_auc_score(y, y_prob), 4),
        "average_precision": round(average_precision_score(y, y_prob), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def train() -> dict:
    """Entrena el modelo final y guarda artefactos."""
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Faltan train.csv/test.csv. Ejecuta: python src/01_prepare_data.py")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = build_pipeline()

    def guardar_artifactos(metrics: dict) -> None:
        report = classification_report(y_test, 
                                       (model.predict_proba(X_test)[:, 1] >= 0.25).astype(int), 
                                       zero_division=0)
        CLASSIFICATION_REPORT_PATH.write_text(report, encoding="utf-8")
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        FEATURES_PATH.write_text(json.dumps({"features": FEATURES, "target": TARGET}, indent=2), 
                                 encoding="utf-8")

    if mlflow is None:
        log.warning("MLflow no está instalado; se guardan artefactos locales sin tracking.")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        guardar_artifactos(metrics)
    else:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "prevencion-infartos"))
        with mlflow.start_run(run_name="random_forest_balanceado"):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            mlflow.log_params({
                "modelo": "RandomForestClassifier",
                "n_estimators": 150,
                "max_depth": 10,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "features": len(FEATURES),
            })
            for k, v in metrics.items():
                if k != "confusion_matrix":
                    mlflow.log_metric(k, v)
            guardar_artifactos(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(CLASSIFICATION_REPORT_PATH))
            mlflow.log_artifact(str(METRICS_PATH))
            mlflow.log_artifact(str(FEATURES_PATH))

    log.info("Métricas test: %s", metrics)
    log.info("Modelo guardado en %s", MODEL_PATH)
    return metrics


if __name__ == "__main__":
    train()
