import json

import pandas as pd
import pytest
import requests

from src import data_collection as dc
from src.feature_engineering import assign_split, build_features
from src.predict import ModelUnavailable, load_artefact, predict_race, race_entrants
from src.train import fit_and_evaluate
from tests.conftest import make_cfg


@pytest.fixture
def trained(clean_df, tmp_path):
    cfg = make_cfg(tmp_path)
    f = build_features(clean_df, cfg)
    f["split"] = assign_split(f, cfg)
    m = f[f["include_in_model"] & (f["split"] != "unused")].copy()
    m["podium"] = m["podium"].astype(int)
    fit_and_evaluate("knn", "A+B+C+D", {"clf__n_neighbors": 5}, m, 0, cfg)
    return cfg, f[f["in_study_period"]]


def test_prediction_output_schema(trained):
    cfg, feats = trained
    art = load_artefact("knn", cfg)
    out = predict_race(art, race_entrants(feats, 2015, 2))
    assert {"driver_id", "podium_score", "predicted_podium", "model_rank", "in_training_data"} <= set(out.columns)
    assert set(out["predicted_podium"]) <= {0, 1}
    assert sorted(out["model_rank"]) == list(range(1, len(out) + 1))
    assert out["podium_score"].between(0, 1).all()
    assert not out["in_training_data"].any()            # 2015 is the test season


def test_entrants_reflect_historical_lineup(trained):
    _, feats = trained
    e14 = set(race_entrants(feats, 2014, 1)["driver_id"])
    e15 = set(race_entrants(feats, 2015, 1)["driver_id"])
    assert "gus" not in e14 and "hal" in e15 and "hal" not in e14


def test_unknown_race_raises(trained):
    cfg, feats = trained
    with pytest.raises(ValueError):
        predict_race(load_artefact("knn", cfg), race_entrants(feats, 1999, 1))


def test_missing_model_artefact(tmp_path):
    with pytest.raises(ModelUnavailable):
        load_artefact("svm", make_cfg(tmp_path))


def test_network_failure_gives_clear_error(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["data"].update({"base_url": "http://127.0.0.1:9/ergast/f1", "max_retries": 0, "request_timeout_seconds": 1})
    with pytest.raises(dc.DataSourceError, match="Could not reach"):
        dc.download_season("results", 2014, cfg)


def test_offline_without_cache_fails_clearly(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(dc.DataSourceError, match="Offline mode"):
        dc.download_season("results", 2014, cfg, offline=True)


def test_offline_uses_cache(tmp_path):
    cfg = make_cfg(tmp_path)
    p = dc.raw_path("results", 2014, cfg)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"meta": {"kind": "results", "season": 2014}, "races": []}))
    meta = dc.download_season("results", 2014, cfg, offline=True)
    assert meta["from_cache"] is True


def test_malformed_response_rejected(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)

    class FakeResp:
        status_code = 200
        url = "http://x"
        headers = {}
        text = "{}"

        def json(self):
            return {"unexpected": 1}

    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: FakeResp())
    with pytest.raises(dc.SchemaError):
        dc.fetch_json("http://x", {}, cfg)


def test_malformed_result_rows_are_logged_not_guessed():
    races = [{"season": "2014", "round": "1", "Circuit": {"circuitId": "c"},
              "Results": [{"Driver": {"driverId": "a"}, "Constructor": {"constructorId": "t"},
                           "position": "1", "positionText": "1", "grid": "1", "status": "Finished"},
                          {"Driver": {}, "Constructor": {"constructorId": "t"}, "position": "2"}]}]
    problems = []
    df = dc.flatten_results(races, problems)
    assert len(df) == 1 and len(problems) == 1
    assert isinstance(df, pd.DataFrame)
