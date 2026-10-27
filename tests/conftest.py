"""Shared fixtures: a small, fully synthetic results table with known structure.

Synthetic data is used ONLY to test code behaviour (transfers, rookies, leakage);
it never feeds the reported experiments.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data_cleaning import clean_results  # noqa: E402


def make_cfg(tmp_path: Path | None = None) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["data"].update({"history_start_season": 2012, "study_start_season": 2013, "study_end_season": 2015,
                        "qualifying_start_season": 2013})
    cfg["split"] = {"train_seasons": [2013, 2013], "validation_seasons": [2014, 2014], "test_seasons": [2015, 2015]}
    cfg["features"]["constructor_lineage"] = {"old_name": "lineage_x", "new_name": "lineage_x"}
    if tmp_path is not None:
        cfg["models"]["dir"] = str(tmp_path / "models")
        cfg["data"]["raw_dir"] = str(tmp_path / "raw")
        cfg["data"]["processed_dir"] = str(tmp_path / "processed")
        cfg["reports"]["results_dir"] = str(tmp_path / "results")
    return cfg


def lineup(season: int) -> dict[str, str]:
    """driver -> constructor. Encodes a transfer, a rookie, a returning driver and a rebrand."""
    team_x = "old_name" if season <= 2013 else "new_name"   # rebrand of the same lineage
    lu = {"alice": "red", "bob": "red", "carl": "blue", "dana": "blue", "eve": team_x, "fay": team_x}
    if season >= 2014:
        lu["dana"] = "red"          # dana transfers blue -> red
        lu["bob"] = "blue"          # bob transfers red -> blue
    if season == 2013:
        lu.pop("fay")               # fay misses 2013, returns 2014
        lu["gus"] = team_x          # gus races only in 2013
    if season >= 2015:
        lu["hal"] = "blue"          # hal is a 2015 rookie (third car, synthetic)
    return lu


def make_flat_results(seed: int = 0, rounds: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2012, 2016):
        lu = lineup(season)
        for rnd in range(1, rounds + 1):
            drivers = list(lu)
            order = rng.permutation(len(drivers))
            grid = rng.permutation(len(drivers)) + 1
            dnf = drivers[order.argmax()]           # last place retires
            for i, d in enumerate(drivers):
                pos = int(order[i]) + 1
                ptext = "R" if d == dnf else str(pos)
                rows.append({
                    "season": str(season), "round": str(rnd), "race_name": f"GP {rnd}",
                    "date": f"{season}-0{min(rnd, 9)}-01", "circuit_id": f"circ{rnd}", "circuit_name": f"Circuit {rnd}",
                    "country": "X", "driver_id": d, "driver_code": d[:3].upper(), "driver_name": d.title(),
                    "constructor_id": lu[d], "constructor_name": lu[d].title(), "car_number": str(i + 1),
                    "grid": str(int(grid[i])), "position": str(pos), "position_text": ptext,
                    "points": "0", "laps": "50", "status": "Finished" if ptext != "R" else "Engine",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def cfg():
    return make_cfg()


@pytest.fixture
def flat_results():
    return make_flat_results()


@pytest.fixture
def clean_df(flat_results, cfg):
    df, _ = clean_results(flat_results, cfg)
    return df
