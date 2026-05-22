"""Tests del pipeline de modelo."""
import os
import sys

import pandas as pd

sys.path.insert(0, "src")
from config import FEATURES
from importlib import import_module

prepare_mod = import_module("01_prepare_data")
train_mod = import_module("02_train_model")


def test_build_pipeline_has_predict_proba():
    model = train_mod.build_pipeline()
    assert hasattr(model, "fit")


def test_train_saves_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "artifacts").mkdir()

    df = prepare_mod.clean_data(pd.read_csv(PathLikeDataset.path, sep=";"))
    train = df.sample(1000, random_state=42)
    test = df.drop(train.index).sample(300, random_state=42)
    train.to_csv(tmp_path / "data/train.csv", index=False)
    test.to_csv(tmp_path / "data/test.csv", index=False)

    metrics = train_mod.train()
    assert (tmp_path / "artifacts/modelo.pkl").exists()
    assert (tmp_path / "artifacts/metrics.json").exists()
    assert "recall" in metrics


class PathLikeDataset:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_prevencion_infartos.csv")


def test_pipeline_can_fit_small_sample():
    df = prepare_mod.clean_data(pd.read_csv(PathLikeDataset.path, sep=";")).sample(800, random_state=1)
    X, y = df[FEATURES], df["Ataque_cardiaco"]
    model = train_mod.build_pipeline()
    model.fit(X, y)
    proba = model.predict_proba(X.head(5))
    assert proba.shape == (5, 2)
