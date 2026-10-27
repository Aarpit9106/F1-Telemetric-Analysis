"""End-to-end data pipeline: cache -> flat tables -> clean -> features -> split.

    python -m src.data_pipeline              # uses cache, downloads anything missing
    python -m src.data_pipeline --offline    # cache only
    python -m src.data_pipeline --refresh    # re-download from the API
"""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from src.config import load_config, resolve
from src.data_cleaning import run_cleaning
from src.data_collection import build_flat_tables, download_all
from src.feature_engineering import all_feature_columns, assign_split, build_features

ID_COLS = ["season", "round", "race_id", "race_index", "race_name", "date", "circuit_id", "circuit_name",
           "country", "driver_id", "driver_code", "driver_name", "constructor_id", "constructor_name",
           "team_lineage", "grid", "quali_position", "position", "position_text", "status", "points",
           "classified", "finish_position", "podium", "podium_all_starters", "include_in_model",
           "exclusion_reason"]


def split_summary(model_df: pd.DataFrame) -> dict:
    out = {}
    for name, part in model_df.groupby("split"):
        out[name] = {
            "seasons": sorted(int(s) for s in part["season"].unique()),
            "races": int(part["race_id"].nunique()),
            "rows": int(len(part)),
            "podium_rows": int(part["podium"].sum()),
            "podium_rate": round(float(part["podium"].mean()), 4),
        }
    # Integrity: every race in exactly one split
    per_race = model_df.groupby("race_id")["split"].nunique()
    out["races_spanning_multiple_splits"] = int((per_race > 1).sum())
    return out


def run_pipeline(refresh: bool = False, offline: bool = False, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    manifest = download_all(cfg, refresh=refresh, offline=offline)
    build_flat_tables(cfg)
    clean, clean_summary = run_cleaning(cfg)
    feats = build_features(clean, cfg)
    feats["split"] = assign_split(feats, cfg)

    proc = resolve(cfg["data"]["processed_dir"])
    num, cat = all_feature_columns()
    keep = [c for c in ID_COLS if c in feats.columns] + ["split"] + [c for c in num + cat if c not in ID_COLS]
    study = feats[feats["in_study_period"]][keep].copy()
    study.to_csv(proc / "features_study.csv", index=False)

    model_df = study[study["include_in_model"]].copy()
    model_df["podium"] = model_df["podium"].astype(int)
    model_df.to_csv(proc / "model_dataset.csv", index=False)

    summary = {
        "manifest_files": len(manifest["files"]),
        "manifest_errors": manifest["errors"],
        "cleaning": {k: v for k, v in clean_summary.items() if k != "quality_before_cleaning"},
        "started_entries_study": int(len(study)),
        "model_rows": int(len(model_df)),
        "split": split_summary(model_df),
        "n_numeric_features": len(num),
        "n_categorical_features": len(cat),
    }
    with open(proc / "pipeline_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def load_model_dataset(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve(cfg["data"]["processed_dir"]) / "model_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data_pipeline` first.")
    return pd.read_csv(path)


def load_study_features(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve(cfg["data"]["processed_dir"]) / "features_study.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data_pipeline` first.")
    return pd.read_csv(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    s = run_pipeline(refresh=a.refresh, offline=a.offline)
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
