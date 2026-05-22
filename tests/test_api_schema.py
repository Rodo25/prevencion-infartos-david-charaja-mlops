"""Tests de contratos API."""
import sys

sys.path.insert(0, ".")
from api.schemas import ClienteInput


def test_cliente_input_valido():
    payload = {
        "Genero": "Hombre",
        "Edad": 58,
        "Flag_hipertension": 1,
        "Flag_problem_cardiaco": 0,
        "Estados_civil": "Si",
        "Tipo_trabajo": "Empresa_privada",
        "Zona_residencia": "Urbano",
        "Promedio_nivel_glucosa": 180.2,
        "IMC": 31.5,
        "Flag_fumador": "Nunca_fuma",
    }
    cliente = ClienteInput(**payload)
    assert cliente.Edad == 58
