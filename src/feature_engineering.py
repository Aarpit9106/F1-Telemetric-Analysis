"""Leakage-safe pre-race feature engineering.

Every historical feature for a driver entry in race *r* is computed from races
with ``race_index < r`` only. This is enforced by computing per-entity sequences
ordered by ``race_index`` and applying ``shift(1)`` before any rolling or
expanding aggregation. ``tests/test_no_leakage.py`` checks this by perturbing
current and future results and asserting that features do not change.

Feature groups (used in ablation experiments):

* A — Basic: grid, pit-lane flag, circuit, season, round, driver, constructor
* B — Recent form: rolling driver & constructor results over the previous 3/5 races
* C — Experience: prior starts, races with current team, rookie/returning flags, gaps
* D — Career context: career rates, in-season standings before the race, grid penalty
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config

NOT_STARTED = {"W", "F"}  # withdrawn / failed to qualify -> no race start

FEATURE_GROUPS: dict[str, dict[str, list[str]]] = {
    "A": {
        "numeric": ["grid_clean", "grid_pitlane", "season", "round"],
        "categorical": ["circuit_id", "driver_id", "constructor_id"],
    },
    "B": {
        "numeric": [
            "drv_avg_pos_3", "drv_avg_pos_5", "drv_podiums_5", "drv_ppr_5", "drv_dnf_rate_5", "drv_avg_grid_5",
            "con_avg_pos_3", "con_avg_pos_5", "con_podiums_5", "con_ppr_5",
            "drv_has_history", "con_has_history",
        ],
        "categorical": [],
    },
    "C": {
        "numeric": [
            "drv_career_starts", "con_prior_races", "drv_races_with_con", "drv_gap_races",
            "is_rookie", "is_returning", "con_changed",
        ],
        "categorical": [],
    },
    "D": {
        "numeric": [
            "drv_career_avg_pos", "drv_career_podium_rate", "drv_season_pts_before", "drv_season_rank_before",
            "con_season_pts_before", "con_season_rank_before", "grid_minus_quali", "quali_missing",
        ],
        "categorical": [],
    },
}

FEATURE_DESCRIPTIONS = {
    "grid_clean": "Starting grid slot; pit-lane starts (grid 0) set to the number of entrants",
    "grid_pitlane": "1 if the driver started from the pit lane",
    "season": "Championship year",
    "round": "Round number within the season (race order)",
    "circuit_id": "Circuit identifier (one-hot)",
    "driver_id": "Driver identifier (one-hot; unseen drivers -> all zeros)",
    "constructor_id": "Official constructor identifier (one-hot; unseen -> all zeros)",
    "drv_avg_pos_3": "Driver mean race order position, previous 3 starts",
    "drv_avg_pos_5": "Driver mean race order position, previous 5 starts",
    "drv_podiums_5": "Driver podiums in previous 5 starts",
    "drv_ppr_5": "Driver standardised points per race, previous 5 starts",
    "drv_dnf_rate_5": "Share of the driver's previous 5 starts that were not classified",
    "drv_avg_grid_5": "Driver mean grid slot, previous 5 starts (qualifying pace proxy)",
    "con_avg_pos_3": "Team mean race order position (all cars), previous 3 races",
    "con_avg_pos_5": "Team mean race order position (all cars), previous 5 races",
    "con_podiums_5": "Team podiums, previous 5 races",
    "con_ppr_5": "Team standardised points per race, previous 5 races",
    "drv_has_history": "1 if the driver has at least one previous start (since history start)",
    "con_has_history": "1 if the team lineage has at least one previous race",
    "drv_career_starts": "Driver starts before this race (counted from history_start_season)",
    "con_prior_races": "Team-lineage races before this race (counted from history_start_season)",
    "drv_races_with_con": "Driver's previous starts with the current team lineage",
    "drv_gap_races": "Championship races held since the driver's previous start",
    "is_rookie": "1 if no previous start",
    "is_returning": "1 if the gap since the previous start >= returning_gap_races",
    "con_changed": "1 if the team lineage differs from the driver's previous start",
    "drv_career_avg_pos": "Driver mean race order position over all previous starts",
    "drv_career_podium_rate": "Driver podium share over all previous starts",
    "drv_season_pts_before": "Driver standardised points earned earlier in this season",
    "drv_season_rank_before": "Driver rank among this race's entrants by season points before the race",
    "con_season_pts_before": "Team standardised points earned earlier in this season",
    "con_season_rank_before": "Team rank among this race's teams by season points before the race",
    "grid_minus_quali": "Grid slot minus qualifying position (grid penalties/promotions); 0 if no qualifying record",
    "quali_missing": "1 if no qualifying record exists for this entry",
}


def groups_to_columns(spec: str) -> tuple[list[str], list[str]]:
    """'A+B+C' -> (numeric columns, categorical columns)."""
    num, cat = [], []
    for g in spec.split("+"):
        g = g.strip()
        if g not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group {g!r}")
        num += FEATURE_GROUPS[g]["numeric"]
        cat += FEATURE_GROUPS[g]["categorical"]
    return num, cat


def _standard_points(pos: pd.Series, classified: pd.Series, scheme: list[int]) -> pd.Series:
    lookup = {i + 1: p for i, p in enumerate(scheme)}
    pts = pos.map(lookup).fillna(0.0)
    return pts.where(classified, 0.0).astype(float)


def build_features(clean: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Return one row per *started* driver entry with all pre-race features.

    ``clean`` must contain every season from ``history_start_season`` so that
    early-study rows have realistic histories.
    """
    cfg = cfg or load_config()
    fcfg = cfg["features"]
    lineage_map = fcfg.get("constructor_lineage", {}) or {}
    windows = fcfg.get("rolling_windows", [3, 5])
    w_short, w_long = min(windows), max(windows)

    df = clean.copy()
    df = df[~df["position_text"].isin(NOT_STARTED)].copy()
    df = df.sort_values(["race_index", "driver_id"]).reset_index(drop=True)

    df["team_lineage"] = df["constructor_id"].map(lambda c: lineage_map.get(c, c))
    df["pos_order"] = df["position"].astype(float)              # race order incl. non-classified
    df["is_podium_h"] = (df["classified"] & (df["position"] <= 3)).astype(float)
    df["not_classified_h"] = (~df["classified"]).astype(float)
    df["std_points"] = _standard_points(df["position"], df["classified"], fcfg["standard_points"])

    # ---------------- driver sequences ----------------
    df = df.sort_values(["driver_id", "race_index"]).reset_index(drop=True)
    gd = df.groupby("driver_id", sort=False)

    def prior_roll(col: str, window: int, how: str = "mean") -> pd.Series:
        shifted = gd[col].shift(1)
        return shifted.groupby(df["driver_id"]).transform(lambda s: getattr(s.rolling(window, min_periods=1), how)())

    df["drv_avg_pos_3"] = prior_roll("pos_order", w_short)
    df["drv_avg_pos_5"] = prior_roll("pos_order", w_long)
    df["drv_podiums_5"] = prior_roll("is_podium_h", w_long, "sum")
    df["drv_ppr_5"] = prior_roll("std_points", w_long)
    df["drv_dnf_rate_5"] = prior_roll("not_classified_h", w_long)
    df["drv_avg_grid_5"] = prior_roll("grid_clean", w_long)

    df["drv_career_starts"] = gd.cumcount().astype(float)
    df["drv_has_history"] = (df["drv_career_starts"] > 0).astype(int)
    df["is_rookie"] = (df["drv_career_starts"] == 0).astype(int)
    prev_idx = gd["race_index"].shift(1)
    df["drv_gap_races"] = df["race_index"] - prev_idx - 1           # NaN for rookies
    df["is_returning"] = (df["drv_gap_races"] >= fcfg["returning_gap_races"]).astype(int)
    prev_team = gd["team_lineage"].shift(1)
    df["con_changed"] = (prev_team.notna() & (prev_team != df["team_lineage"])).astype(int)
    df["drv_races_with_con"] = df.groupby(["driver_id", "team_lineage"], sort=False).cumcount().astype(float)

    cum_pos = gd["pos_order"].cumsum() - df["pos_order"]
    cum_pod = gd["is_podium_h"].cumsum() - df["is_podium_h"]
    n_prev = df["drv_career_starts"].replace(0, np.nan)
    df["drv_career_avg_pos"] = cum_pos / n_prev
    df["drv_career_podium_rate"] = cum_pod / n_prev

    gds = df.groupby(["driver_id", "season"], sort=False)["std_points"]
    df["drv_season_pts_before"] = gds.cumsum() - df["std_points"]

    # ---------------- constructor (team lineage) sequences ----------------
    team_race = (df.groupby(["team_lineage", "race_index", "season"], as_index=False)
                 .agg(t_pos=("pos_order", "mean"), t_pod=("is_podium_h", "sum"), t_pts=("std_points", "sum"))
                 .sort_values(["team_lineage", "race_index"]).reset_index(drop=True))
    gt = team_race.groupby("team_lineage", sort=False)

    def team_prior_roll(col: str, window: int, how: str = "mean") -> pd.Series:
        shifted = gt[col].shift(1)
        return shifted.groupby(team_race["team_lineage"]).transform(
            lambda s: getattr(s.rolling(window, min_periods=1), how)())

    team_race["con_avg_pos_3"] = team_prior_roll("t_pos", w_short)
    team_race["con_avg_pos_5"] = team_prior_roll("t_pos", w_long)
    team_race["con_podiums_5"] = team_prior_roll("t_pod", w_long, "sum")
    team_race["con_ppr_5"] = team_prior_roll("t_pts", w_long)
    team_race["con_prior_races"] = gt.cumcount().astype(float)
    team_race["con_has_history"] = (team_race["con_prior_races"] > 0).astype(int)
    team_race["con_season_pts_before"] = (team_race.groupby(["team_lineage", "season"])["t_pts"].cumsum()
                                          - team_race["t_pts"])
    df = df.merge(team_race.drop(columns=["t_pos", "t_pod", "t_pts", "season"]),
                  on=["team_lineage", "race_index"], how="left")

    # ---------------- in-race ranks of pre-race standings ----------------
    df["drv_season_rank_before"] = df.groupby("race_index")["drv_season_pts_before"].rank(
        ascending=False, method="min")
    team_pts = df.drop_duplicates(["race_index", "team_lineage"])[["race_index", "team_lineage", "con_season_pts_before"]]
    team_pts = team_pts.assign(con_season_rank_before=team_pts.groupby("race_index")["con_season_pts_before"].rank(
        ascending=False, method="min"))
    df = df.merge(team_pts[["race_index", "team_lineage", "con_season_rank_before"]],
                  on=["race_index", "team_lineage"], how="left")

    # ---------------- qualifying-derived (known before the race) ----------------
    df["quali_missing"] = df["quali_position"].isna().astype(int)
    df["grid_minus_quali"] = (df["grid_clean"] - df["quali_position"]).fillna(0.0)

    # Secondary target for the sensitivity experiment: all starters, non-classified = 0.
    df["podium_all_starters"] = df["is_podium_h"]

    df = df.drop(columns=["is_podium_h", "not_classified_h"])
    return df.sort_values(["race_index", "grid_clean", "driver_id"]).reset_index(drop=True)


def assign_split(df: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    """Chronological split label by season; whole races stay together by construction."""
    cfg = cfg or load_config()
    s = cfg["split"]
    split = pd.Series("unused", index=df.index, dtype="object")
    split[df["season"].between(*s["train_seasons"])] = "train"
    split[df["season"].between(*s["validation_seasons"])] = "validation"
    split[df["season"].between(*s["test_seasons"])] = "test"
    return split


def all_feature_columns() -> tuple[list[str], list[str]]:
    return groups_to_columns("+".join(FEATURE_GROUPS))
