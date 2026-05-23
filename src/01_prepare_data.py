"""src/01_prepare_data.py — Prepara el dataset de prevención de infartos.

Lee el CSV separado por punto y coma, normaliza nombres/valores, separa train/test
con estratificación y guarda data/train.csv y data/test.csv.
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_DATA_PATH, TRAIN_PATH, TEST_PATH, TARGET, RANDOM_STATE, TEST_SIZE, DATA_DIR

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s | DATA | %(message)s", 
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el CSV del proyecto. Soporta separador ';'."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {path}")
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia tipos y categorías básicas sin filtrar registros."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": None, "None": None, "": None})
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("Int64")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera train/test estratificados."""
    DATA_DIR.mkdir(exist_ok=True)
    df = clean_data(load_raw())
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    log.info("Dataset total: %d filas x %d columnas", df.shape[0], df.shape[1])
    log.info("Train: %d | Test: %d | Tasa positiva train: %.3f", 
             len(train_df), len(test_df), train_df[TARGET].mean())
    return train_df, test_df


if __name__ == "__main__":
    prepare()
