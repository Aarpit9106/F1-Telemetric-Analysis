import numpy as np

from src.feature_engineering import build_features, groups_to_columns


def _row(f, driver, season, rnd):
    return f[(f["driver_id"] == driver) & (f["season"] == season) & (f["round"] == rnd)].iloc[0]


def test_rookie_has_missing_history_not_zero(clean_df, cfg):
    f = build_features(clean_df, cfg)
    r = _row(f, "hal", 2015, 1)
    assert r["is_rookie"] == 1 and r["drv_has_history"] == 0 and r["drv_career_starts"] == 0
    # missing history must stay missing (NaN), never an optimistic 0 position
    assert np.isnan(r["drv_avg_pos_5"]) and np.isnan(r["drv_career_avg_pos"]) and np.isnan(r["drv_gap_races"])
    # second race: history now exists
    assert _row(f, "hal", 2015, 2)["drv_career_starts"] == 1


def test_career_counts_continue_across_transfer(clean_df, cfg):
    f = build_features(clean_df, cfg)
    last_blue = _row(f, "dana", 2013, 5)
    first_red = _row(f, "dana", 2014, 1)
    assert first_red["con_changed"] == 1
    assert first_red["drv_races_with_con"] == 0                 # new team experience resets
    assert first_red["drv_career_starts"] == last_blue["drv_career_starts"] + 1   # personal history kept
    assert not np.isnan(first_red["drv_avg_pos_5"])               # personal form kept
    assert _row(f, "dana", 2014, 2)["con_changed"] == 0


def test_driver_form_uses_previous_team_results(clean_df, cfg):
    f = build_features(clean_df, cfg)
    hist = f[(f["driver_id"] == "dana")].sort_values("race_index")
    idx = hist.index[hist["season"].eq(2014) & hist["round"].eq(1)][0]
    pos = hist.loc[:idx, "pos_order"].iloc[:-1].tail(5).mean()
    assert np.isclose(hist.loc[idx, "drv_avg_pos_5"], pos)


def test_returning_driver(clean_df, cfg):
    cfg = dict(cfg)
    cfg["features"] = dict(cfg["features"], returning_gap_races=3)
    f = build_features(clean_df, cfg)
    r = _row(f, "fay", 2014, 1)
    assert r["drv_gap_races"] == 5          # missed all 5 rounds of 2013
    assert r["is_returning"] == 1
    assert r["drv_career_starts"] == 5      # 2012 starts retained


def test_constructor_lineage_continuity(clean_df, cfg):
    f = build_features(clean_df, cfg)
    r = _row(f, "eve", 2014, 1)            # team renamed old_name -> new_name
    assert r["constructor_id"] == "new_name"
    assert r["con_has_history"] == 1
    assert r["con_changed"] == 0           # same lineage, not a transfer
    assert r["con_prior_races"] == 10      # 2012 + 2013 races of the lineage


def test_absent_driver_not_in_race(clean_df, cfg):
    f = build_features(clean_df, cfg)
    assert f[(f["driver_id"] == "gus") & (f["season"] == 2014)].empty
    assert f[(f["driver_id"] == "fay") & (f["season"] == 2013)].empty


def test_season_points_reset_each_season(clean_df, cfg):
    f = build_features(clean_df, cfg)
    first = f[(f["season"] == 2014) & (f["round"] == 1)]
    assert (first["drv_season_pts_before"] == 0).all()
    assert (first["con_season_pts_before"] == 0).all()


def test_feature_groups_cumulative():
    a_num, a_cat = groups_to_columns("A")
    all_num, all_cat = groups_to_columns("A+B+C+D")
    assert set(a_num) < set(all_num) and a_cat == all_cat
