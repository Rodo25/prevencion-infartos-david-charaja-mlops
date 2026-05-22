"""api/schemas.py — Contratos de entrada y salida de la API."""
from typing import Optional

from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    Genero: str = Field(..., examples=["Hombre"])
    Edad: float = Field(..., ge=0, le=120, examples=[58])
    Flag_hipertension: int = Field(..., ge=0, le=1, examples=[1])
    Flag_problem_cardiaco: int = Field(..., ge=0, le=1, examples=[0])
    Estados_civil: str = Field(..., examples=["Si"])
    Tipo_trabajo: str = Field(..., examples=["Empresa_privada"])
    Zona_residencia: str = Field(..., examples=["Urbano"])
    Promedio_nivel_glucosa: float = Field(..., ge=0, examples=[165.2])
    IMC: Optional[float] = Field(None, ge=0, examples=[29.4])
    Flag_fumador: Optional[str] = Field(None, examples=["Nunca_fuma"])


class PrediccionOutput(BaseModel):
    score_riesgo: float
    prediccion: int
    decision: str
    modelo: str
    umbral: float


class HealthResponse(BaseModel):
    status: str
    modelo: str
    version: str
    recall: float
    env: str
