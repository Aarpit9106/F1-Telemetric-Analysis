import numpy as np
import pandas as pd

from src.data_cleaning import clean_results, quality_report


def test_types_and_unique_key(clean_df):
    assert clean_df["season"].dtype.kind == "i"
    assert clean_df["round"].dtype.kind == "i"
    assert clean_df["grid"].dtype.kind in "if"
    assert not clean_df.duplicated(subset=["season", "round", "driver_id"]).any()


def test_exact_duplicates_removed(flat_results, cfg):
    dup = pd.concat([flat_results, flat_results.iloc[:3]], ignore_index=True)
    assert quality_report(dup)["exact_duplicate_rows"] == 3
    df, summary = clean_results(dup, cfg)
    assert len(df) == len(flat_results)
    assert any("duplicate" in s["stage"] for s in summary["stages"])


def test_conflicting_duplicate_key_keeps_one(flat_results, cfg):
    bad = flat_results.iloc[[0]].copy()
    bad["grid"] = "99"
    df, _ = clean_results(pd.concat([flat_results, bad], ignore_index=True), cfg)
    assert not df.duplicated(subset=["season", "round", "driver_id"]).any()


def test_target_generation(clean_df):
    classified = clean_df[clean_df["classified"]]
    expected = (classified["position"] <= 3).astype(float)
    assert (classified["podium"] == expected).all()
    # non-classified rows never receive an ordinary 0 target
    assert clean_df.loc[~clean_df["classified"], "podium"].isna().all()


def test_inclusion_policy(clean_df):
    study = clean_df[clean_df["in_study_period"]]
    retired = study[study["position_text"] == "R"]
    assert (~retired["include_in_model"]).all()
    assert (retired["exclusion_reason"] == "retired (not classified)").all()
    history = clean_df[~clean_df["in_study_period"]]
    assert (~history["include_in_model"]).all()
    included = clean_df[clean_df["include_in_model"]]
    assert included["finish_position"].notna().all()


def test_invalid_position_and_ids(flat_results, cfg):
    bad = flat_results.copy()
    bad.loc[0, "driver_id"] = None
    bad.loc[1, "position"] = "abc"
    df, summary = clean_results(bad, cfg)
    assert df["driver_id"].notna().all()
    assert any(s["stage"].startswith("convert position") for s in summary["stages"])


def test_pitlane_grid(flat_results, cfg):
    fr = flat_results.copy()
    fr.loc[5, "grid"] = "0"
    df, _ = clean_results(fr, cfg)
    row = df[(df["season"] == int(fr.loc[5, "season"])) & (df["round"] == int(fr.loc[5, "round"]))
             & (df["driver_id"] == fr.loc[5, "driver_id"])].iloc[0]
    assert row["grid_pitlane"] == 1
    assert row["grid_clean"] == row["n_entrants"]


def test_race_index_chronological(clean_df):
    r = clean_df[["season", "round", "race_index"]].drop_duplicates().sort_values("race_index")
    assert (np.diff(r["season"] * 100 + r["round"]) > 0).all()
