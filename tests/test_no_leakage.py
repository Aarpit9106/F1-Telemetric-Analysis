"""Leakage tests: features of a race must not depend on that race's outcome or any later race."""
import numpy as np
import pandas as pd

from src.data_cleaning import clean_results
from src.feature_engineering import all_feature_columns, assign_split, build_features
from src.tune_models import build_pipeline

TARGET = (2014, 3)


def _features_for(flat, cfg, season, rnd):
    clean, _ = clean_results(flat, cfg)
    f = build_features(clean, cfg)
    num, cat = all_feature_columns()
    race = f[(f["season"] == season) & (f["round"] == rnd)]
    return race[num + cat].set_index(race["driver_id"].rename("entry")).sort_index()


def _scramble(flat, mask, seed):
    rng = np.random.default_rng(seed)
    out = flat.copy()
    for (s, r), idx in out[mask].groupby(["season", "round"]).groups.items():
        perm = rng.permutation(len(idx)) + 1
        out.loc[idx, "position"] = perm.astype(str)
        out.loc[idx, "position_text"] = perm.astype(str)
        out.loc[idx, "status"] = "Finished"
    return out


def test_current_race_outcome_does_not_change_features(flat_results, cfg):
    base = _features_for(flat_results, cfg, *TARGET)
    mask = (flat_results["season"] == str(TARGET[0])) & (flat_results["round"] == str(TARGET[1]))
    changed = _scramble(flat_results, mask, seed=1)
    assert not changed.loc[mask, "position"].equals(flat_results.loc[mask, "position"])
    pd.testing.assert_frame_equal(base, _features_for(changed, cfg, *TARGET))


def test_future_races_do_not_change_features(flat_results, cfg):
    base = _features_for(flat_results, cfg, *TARGET)
    s = flat_results["season"].astype(int)
    r = flat_results["round"].astype(int)
    future = (s > TARGET[0]) | ((s == TARGET[0]) & (r > TARGET[1]))
    changed = _scramble(flat_results, future, seed=2)
    pd.testing.assert_frame_equal(base, _features_for(changed, cfg, *TARGET))


def test_past_race_change_does_change_features(flat_results, cfg):
    """Sanity check that the test above is sensitive: changing the past must change features."""
    base = _features_for(flat_results, cfg, *TARGET)
    past = (flat_results["season"] == "2014") & (flat_results["round"] == "2")
    changed = _scramble(flat_results, past, seed=3)
    assert not base.equals(_features_for(changed, cfg, *TARGET))


def test_no_outcome_columns_in_feature_list():
    num, cat = all_feature_columns()
    forbidden = {"position", "finish_position", "podium", "points", "status", "laps", "position_text",
                 "podium_all_starters", "pos_order", "std_points"}
    assert forbidden.isdisjoint(num + cat)


def test_chronological_split_integrity(clean_df, cfg):
    f = build_features(clean_df, cfg)
    f["split"] = assign_split(f, cfg)
    f = f[f["split"] != "unused"]
    assert f.groupby("race_index")["split"].nunique().max() == 1
    mx = f.groupby("split")["race_index"].agg(["min", "max"])
    assert mx.loc["train", "max"] < mx.loc["validation", "min"]
    assert mx.loc["validation", "max"] < mx.loc["test", "min"]


def test_preprocessing_fitted_on_training_only(clean_df, cfg):
    f = build_features(clean_df, cfg)
    f["split"] = assign_split(f, cfg)
    f = f[f["include_in_model"]].copy()
    f["podium"] = f["podium"].astype(int)
    train = f[f["split"] == "train"]
    pipe = build_pipeline("knn", "A+B", 0, {"clf__n_neighbors": 3}).fit(train, train["podium"])
    num = pipe.named_steps["pre"].transformers_[0][2]
    imputer = pipe.named_steps["pre"].named_transformers_["num"].named_steps["impute"]
    expected = train[num].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    scaler = pipe.named_steps["pre"].named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, train[num].fillna(train[num].median()).mean().to_numpy())
