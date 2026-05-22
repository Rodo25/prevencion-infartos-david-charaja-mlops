"""api/app.py — API REST del modelo de prevención de infartos."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import predictor
from api.schemas import ClienteInput, HealthResponse, PrediccionOutput

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s | API | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la API."""
    log.info("Iniciando API en entorno: %s", os.getenv("ENV", "dev"))
    predictor.cargar()
    yield
    log.info("API finalizada")


app = FastAPI(
    title="API Prevención de Infartos — Preproducción",
    description="Servicio de inferencia para estimar probabilidad de ataque cardiaco.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["Info"])
def root():
    return {"api": "Prevención de Infartos", "version": "1.0.0", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if predictor.modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthResponse(
        status="ok",
        modelo=type(predictor.modelo).__name__,
        version="1.0.0",
        recall=float(predictor.metricas.get("recall", 0)),
        env=os.getenv("ENV", "dev"),
    )


@app.post("/predecir", response_model=PrediccionOutput, tags=["Predicción"])
def predecir(cliente: ClienteInput):
    try:
        return PrediccionOutput(**predictor.predecir(cliente.model_dump()))
    except Exception as e:
        log.exception("Error en predicción")
        raise HTTPException(status_code=500, detail=str(e))
