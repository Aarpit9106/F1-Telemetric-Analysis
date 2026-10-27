"""Data quality checks, type conversion, target creation and inclusion policy.

The cleaned table keeps EVERY race entry (classified or not) because
non-classified entries still matter for historical features (e.g. a retirement
is part of a driver's recent form). Whether a row may be used as a supervised
example is recorded in ``include_in_model`` and ``exclusion_reason``.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve

POSITION_TEXT_REASONS = {
    "R": "retired (not classified)",
    "D": "disqualified",
    "E": "excluded",
    "W": "withdrawn / did not start",
    "F": "failed to qualify",
    "N": "not classified",
}

KEY = ["season", "round", "driver_id"]


def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Shape, dtypes, missing values and duplicate counts of a table."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_keys": int(df.duplicated(subset=[k for k in KEY if k in df.columns]).sum()),
    }


def convert_types(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Convert numeric columns; values that fail conversion become NaN and are counted."""
    out = df.copy()
    for col in ["season", "round", "grid", "position", "laps"]:
        before = out[col].notna().sum()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        failed = int(before - out[col].notna().sum())
        if failed:
            log.append({"stage": f"convert {col}", "rows_affected": failed, "action": "set to NaN"})
    out["points"] = pd.to_numeric(out["points"], errors="coerce").astype(float)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["driver_id", "constructor_id", "circuit_id", "position_text", "status"]:
        out[col] = out[col].astype("string").str.strip()
    return out


def clean_results(results: pd.DataFrame, cfg: dict | None = None,
                  qualifying: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (clean table, cleaning log)."""
    cfg = cfg or load_config()
    tcfg, dcfg = cfg["target"], cfg["data"]
    log: list[dict] = [{"stage": "loaded flat results", "rows": int(len(results))}]
    report_before = quality_report(results)

    df = results.drop_duplicates().copy()
    if len(df) != len(results):
        log.append({"stage": "drop exact duplicate rows", "rows_removed": int(len(results) - len(df))})

    df = convert_types(df, log)

    bad_ids = df["driver_id"].isna() | df["constructor_id"].isna() | df["season"].isna() | df["round"].isna()
    if bad_ids.any():
        log.append({"stage": "drop rows with invalid identifiers", "rows_removed": int(bad_ids.sum())})
        df = df[~bad_ids]

    dup_key = df.duplicated(subset=KEY, keep="first")
    if dup_key.any():
        log.append({"stage": "drop duplicate driver-race keys (kept first)", "rows_removed": int(dup_key.sum())})
        df = df[~dup_key]
    assert not df.duplicated(subset=KEY).any(), "driver-race key must be unique"

    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)

    # Chronological race index across all seasons (unique per Grand Prix).
    races = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"]).reset_index(drop=True)
    races["race_index"] = np.arange(len(races))
    df = df.merge(races, on=["season", "round"], how="left")
    df["race_id"] = df["season"].astype(str) + "-" + df["round"].astype(str).str.zfill(2)
    df["n_entrants"] = df.groupby("race_id")["driver_id"].transform("size")

    # Grid: 0 means a pit-lane start (or no start). Keep information explicitly.
    df["grid_pitlane"] = (df["grid"] == 0).astype(int)
    df["grid_clean"] = df["grid"].where(df["grid"] > 0, df["n_entrants"]).astype(float)

    # Classified = positionText is a number.
    df["classified"] = df["position_text"].str.fullmatch(r"\d+").fillna(False).astype(bool)
    df["finish_position"] = np.where(df["classified"], df["position"], np.nan)
    df["podium"] = np.where(df["classified"], (df["position"] <= tcfg["podium_max_position"]).astype(float), np.nan)

    # Inclusion policy.
    in_study = df["season"].between(dcfg["study_start_season"], dcfg["study_end_season"])
    df["in_study_period"] = in_study
    reason = pd.Series("", index=df.index, dtype="object")
    reason[~in_study] = "history-only season (before study period)"
    not_class = in_study & ~df["classified"]
    reason[not_class] = df.loc[not_class, "position_text"].map(POSITION_TEXT_REASONS).fillna("unknown non-numeric position")
    bad_pos = in_study & df["classified"] & df["position"].isna()
    reason[bad_pos] = "missing/invalid finishing position"
    df["exclusion_reason"] = reason
    df["include_in_model"] = reason.eq("")

    if qualifying is not None and len(qualifying):
        q = qualifying.copy()
        q["season"] = pd.to_numeric(q["season"], errors="coerce").astype("Int64")
        q["round"] = pd.to_numeric(q["round"], errors="coerce").astype("Int64")
        q["quali_position"] = pd.to_numeric(q["quali_position"], errors="coerce")
        q = q.dropna(subset=["season", "round", "driver_id"]).drop_duplicates(subset=KEY)
        q["season"], q["round"] = q["season"].astype(int), q["round"].astype(int)
        df = df.merge(q[KEY + ["quali_position"]], on=KEY, how="left")
    else:
        df["quali_position"] = np.nan

    df = df.sort_values(["race_index", "grid_clean", "driver_id"]).reset_index(drop=True)

    study = df[df["in_study_period"]]
    excluded = (study.loc[~study["include_in_model"], "exclusion_reason"].value_counts().rename_axis("reason")
                .reset_index(name="rows"))
    model_rows = study[study["include_in_model"]]
    summary = {
        "quality_before_cleaning": report_before,
        "stages": log,
        "rows_all_seasons": int(len(df)),
        "rows_study_period": int(len(study)),
        "rows_history_only": int((~df["in_study_period"]).sum()),
        "races_study_period": int(study["race_id"].nunique()),
        "seasons_study_period": sorted(int(s) for s in study["season"].unique()),
        "rows_included_for_modelling": int(len(model_rows)),
        "rows_excluded_study_period": int(len(study) - len(model_rows)),
        "exclusions_by_reason": excluded.to_dict(orient="records"),
        "podium_rate_included": float(model_rows["podium"].mean()) if len(model_rows) else None,
        "podium_count_included": int(model_rows["podium"].sum()) if len(model_rows) else 0,
        "races_with_fewer_than_3_classified": int((model_rows.groupby("race_id")["podium"].sum() < 3).sum()),
        "qualifying_coverage_study": float(study["quali_position"].notna().mean()) if len(study) else None,
    }
    return df, summary


def coverage_check(clean: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Compare scheduled races with races that have results."""
    if schedule is None or schedule.empty:
        return pd.DataFrame()
    sch = schedule.copy()
    sch["season"], sch["round"] = sch["season"].astype(int), sch["round"].astype(int)
    have = clean[["season", "round"]].drop_duplicates().assign(has_results=True)
    cov = sch.merge(have, on=["season", "round"], how="left")
    cov["has_results"] = cov["has_results"].fillna(False).astype(bool)
    return cov


def run_cleaning(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_config()
    proc = resolve(cfg["data"]["processed_dir"])
    results = pd.read_csv(proc / "results_flat.csv", dtype=str)
    qual_path = proc / "qualifying_flat.csv"
    qualifying = pd.read_csv(qual_path, dtype=str) if qual_path.exists() and qual_path.stat().st_size > 1 else None
    clean, summary = clean_results(results, cfg, qualifying)
    sch_path = proc / "schedule_flat.csv"
    if sch_path.exists() and sch_path.stat().st_size > 1:
        cov = coverage_check(clean, pd.read_csv(sch_path, dtype=str))
        missing = cov[~cov["has_results"]]
        summary["scheduled_races_without_results"] = missing[["season", "round", "race_name"]].to_dict(orient="records")
        cov.groupby("season")["has_results"].agg(["size", "sum"]).rename(
            columns={"size": "scheduled", "sum": "with_results"}).to_csv(proc / "season_coverage.csv")
    clean.to_csv(proc / "results_clean.csv", index=False)
    with open(proc / "cleaning_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return clean, summary


if __name__ == "__main__":
    _, s = run_cleaning()
    print(json.dumps({k: v for k, v in s.items() if k != "quality_before_cleaning"}, indent=2, default=str))
