"""Load saved model artefacts and score the entrants of a historical race.

Features come from ``features_study.csv`` which is built by the leakage-safe
pipeline, so every value is what was knowable before that race started.
Entrants are taken from the race's own result sheet, so line-ups and driver
transfers are always the historical ones (no hard-coded driver list).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, resolve

MODEL_NAMES = ["decision_tree", "knn", "svm", "random_forest"]


class ModelUnavailable(FileNotFoundError):
    """Raised when a model artefact has not been trained yet."""


def artefact_path(name: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return resolve(cfg["models"]["dir"]) / f"{name}.joblib"


def available_models(cfg: dict | None = None) -> list[str]:
    return [n for n in MODEL_NAMES if artefact_path(n, cfg).exists()]


def load_artefact(name: str, cfg: dict | None = None) -> dict:
    path = artefact_path(name, cfg)
    if not path.exists():
        raise ModelUnavailable(f"Model '{name}' has not been trained ({path} missing). Run `python -m src.train`.")
    art = joblib.load(path)
    required = {"model", "numeric_features", "categorical_features", "feature_set"}
    if not required.issubset(art):
        raise ModelUnavailable(f"Artefact {path} is missing keys {required - set(art)}")
    return art


def race_entrants(features: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    """All drivers who started the given race (historical line-up)."""
    race = features[(features["season"] == int(season)) & (features["round"] == int(rnd))]
    return race.copy()


def predict_race(art: dict, race: pd.DataFrame) -> pd.DataFrame:
    """Score every entrant; returns a table ranked by podium score."""
    if race.empty:
        raise ValueError("No entrants found for this race in the processed dataset.")
    cols = art["numeric_features"] + art["categorical_features"]
    missing = [c for c in cols if c not in race.columns]
    if missing:
        raise KeyError(f"Feature columns missing from data: {missing}")
    model = art["model"]
    X = race[cols]
    pred = model.predict(X).astype(int)
    if art.get("score_kind") == "probability":
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
    out = race[["season", "round", "race_name", "driver_id", "driver_name", "constructor_name",
                "grid", "finish_position", "position_text", "podium_all_starters", "drv_career_starts"]].copy()
    out["podium_score"] = np.asarray(score, dtype=float)
    out["predicted_podium"] = pred
    out["model_rank"] = out["podium_score"].rank(ascending=False, method="first").astype(int)
    out["in_training_data"] = int(race["season"].iloc[0]) in set(art.get("trained_on_seasons", []))
    return out.sort_values("model_rank").reset_index(drop=True)


def describe_score(score: float, kind: str, name: str = "") -> str:
    if kind == "probability" and name == "knn":
        return (f"model score {score:.2f} (0-1): the distance-weighted share of similar past entries that "
                "reached the podium. Treat it as a rough estimate, not a calibrated probability.")
    if kind == "probability":
        return (f"model score {score:.2f} (0-1). Trained with balanced class weights, so this is a relative "
                "likelihood score, not a calibrated probability.")
    return (f"SVM margin {score:+.2f}. Positive values fall on the podium side of the decision boundary; "
            "this is a distance, not a probability.")
