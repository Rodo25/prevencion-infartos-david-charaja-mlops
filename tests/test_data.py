"""Tests de preparación de datos."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, "src")
from config import FEATURES, TARGET
from importlib import import_module

prepare_mod = import_module("01_prepare_data")

def test_load_raw_returns_dataframe():
    df = prepare_mod.load_raw(Path("data/dataset_prevencion_infartos.csv"))
    assert isinstance(df, pd.DataFrame)


def test_clean_data_has_target_binary():
    df = prepare_mod.clean_data(prepare_mod.load_raw(Path("data/dataset_prevencion_infartos.csv")))
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_clean_data_has_features():
    df = prepare_mod.clean_data(prepare_mod.load_raw(Path("data/dataset_prevencion_infartos.csv")))
    for col in FEATURES:
        assert col in df.columns
