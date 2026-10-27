import joblib
import numpy as np
import pytest

from src.feature_engineering import assign_split, build_features
from src.tune_models import build_pipeline, tune
from src.train import MODELS, fit_and_evaluate
from tests.conftest import make_cfg

SMALL_PARAMS = {
    "decision_tree": {"clf__max_depth": 3},
    "knn": {"clf__n_neighbors": 5},
    "svm": {"clf__kernel": "rbf", "clf__C": 1.0},
    "random_forest": {"clf__n_estimators": 20, "clf__max_depth": 4},
}


@pytest.fixture
def model_df(clean_df, cfg):
    f = build_features(clean_df, cfg)
    f["split"] = assign_split(f, cfg)
    f = f[f["include_in_model"] & (f["split"] != "unused")].copy()
    f["podium"] = f["podium"].astype(int)
    return f


@pytest.mark.parametrize("name", MODELS)
def test_each_model_trains_and_predicts(model_df, name):
    train = model_df[model_df["split"] == "train"]
    test = model_df[model_df["split"] == "test"]
    pipe = build_pipeline(name, "A+B+C+D", 0, SMALL_PARAMS[name]).fit(train, train["podium"])
    pred = pipe.predict(test)
    assert set(np.unique(pred)) <= {0, 1}
    assert len(pred) == len(test)


def test_unseen_categories_are_handled(model_df):
    train = model_df[(model_df["split"] == "train")]
    test = model_df[model_df["driver_id"] == "hal"]          # rookie never seen in training
    assert not test.empty and "hal" not in set(train["driver_id"])
    pipe = build_pipeline("random_forest", "A", 0, SMALL_PARAMS["random_forest"]).fit(train, train["podium"])
    assert len(pipe.predict(test)) == len(test)


def test_tuning_uses_validation_only(model_df):
    out = tune("decision_tree", "A+B", model_df, seed=0, n_jobs=1)
    best = out["best"]
    assert 0 <= best["val_average_precision"] <= 1
    assert 0 <= best["train_average_precision"] <= 1
    assert best["n_candidates"] == len(out["all"])
    # test rows must never reach the search
    assert (out["all"]["model"] == "decision_tree").all()


def test_fit_and_evaluate_saves_loadable_artefact(model_df, tmp_path):
    cfg = make_cfg(tmp_path)
    res, _, final, preds = fit_and_evaluate("decision_tree", "A+B", SMALL_PARAMS["decision_tree"], model_df, 0, cfg)
    path = tmp_path / "models" / "decision_tree.joblib"
    assert path.exists()
    art = joblib.load(path)
    assert art["feature_set"] == "A+B"
    assert art["trained_on_seasons"] == [2013, 2014]
    for split in ["train", "validation", "test"]:
        assert 0 <= res[split]["f1"] <= 1
    assert res["timing"]["fit_seconds_final"] > 0
    assert set(preds["predicted"].unique()) <= {0, 1}
