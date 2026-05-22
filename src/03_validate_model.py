"""src/03_validate_model.py — Quality gate del modelo."""
import json
import sys

from config import METRICS_PATH, MODEL_PATH

MIN_RECALL = 0.70
MIN_AVG_PRECISION = 0.04


def validate() -> None:
    """Valida que el modelo cumpla métricas mínimas para no desplegar basura."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe {MODEL_PATH}. Ejecuta entrenamiento.")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"No existe {METRICS_PATH}. Ejecuta entrenamiento.")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    recall = float(metrics.get("recall", 0))
    avg_precision = float(metrics.get("average_precision", 0))

    print("=== Quality Gate — Prevención de Infartos ===")
    print(f"Recall: {recall:.4f} | mínimo: {MIN_RECALL}")
    print(f"Average precision: {avg_precision:.4f} | mínimo: {MIN_AVG_PRECISION}")

    if recall < MIN_RECALL or avg_precision < MIN_AVG_PRECISION:
        print("FALLO: el modelo no cumple el umbral mínimo.")
        sys.exit(1)

    print("APROBADO: modelo apto para despliegue preproductivo.")


if __name__ == "__main__":
    validate()
