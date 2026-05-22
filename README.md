# MLOps — Prevención de Infartos

Proyecto académico de MLOps para entrenar, versionar, desplegar y monitorear un modelo que predice el riesgo de `Ataque_cardiaco` usando variables sencillas del cliente asegurado.

## Estructura

```text
.
├── api/                         # API FastAPI para inferencia
├── data/                        # Dataset base
├── src/                         # Pipeline de datos, entrenamiento y validación
├── tests/                       # Pruebas unitarias y smoke tests
├── artifacts/                   # Artefactos generados: modelo, métricas, features
├── mlruns/                      # Tracking local de MLflow
├── Dockerfile                   # Imagen de API
├── Dockerfile.trainer           # Imagen de entrenamiento
├── docker-compose.preprod.yml   # Stack preproductivo
└── Makefile                     # Comandos automatizados
```

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make all
```

## API local

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger: `http://localhost:8000/docs`

## Preproducción con Docker Compose

```bash
make preprod-up
make smoke
```

Servicios:
- API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`

## MLflow

El entrenamiento registra:
- métricas: F1, recall, precision, accuracy, ROC AUC, PR AUC.
- artefactos: matriz de confusión, reporte de clasificación, modelo serializado.
- parámetros del modelo y del split.

> Nota metodológica: el dataset está fuertemente desbalanceado; por eso el quality gate prioriza `recall` y `average_precision`.
