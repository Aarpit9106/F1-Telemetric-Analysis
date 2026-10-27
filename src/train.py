"""Train, tune and evaluate the four model families; save artefacts and telemetry.

    python -m src.train               # full run: ablation + tuning + curves + final models
    python -m src.train --fast        # fewer feature-set experiments (A and A+B+C+D only)

Protocol
--------
1. For each model x feature set: tune hyperparameters on train (2014-2020),
   score on validation (2021-2022). Selection metric: validation average precision.
2. For each model pick the feature set with the best validation AP.
3. Fit that configuration on train only -> training & validation metrics, gap,
   complexity curves, learning curves, permutation importance.
4. Refit on train+validation -> evaluate ONCE on the test seasons (2023-2024).
5. Preferred model = highest validation AP (decided before looking at test).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import load_config, resolve
from src.data_pipeline import load_model_dataset
from src.evaluate import evaluate_model, model_scores
from src.feature_engineering import groups_to_columns
from src.telemetry import environment_info, prediction_latency
from src.tune_models import COMPLEXITY_PARAM, MODEL_LABELS, build_pipeline, tune

MODELS = ["decision_tree", "knn", "svm", "random_forest"]
warnings.filterwarnings("ignore", category=UserWarning)


def _xy(df: pd.DataFrame, splits: list[str]):
    part = df[df["split"].isin(splits)].sort_values("race_index")
    return part, part["podium"].to_numpy()


def _ap(model, X, y) -> float:
    from sklearn.metrics import average_precision_score
    s, _ = model_scores(model, X)
    return float(average_precision_score(y, s))


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
def run_ablation(df, seed, feature_sets, models):
    rows, all_res = [], []
    for name in models:
        for fs in feature_sets:
            t0 = time.perf_counter()
            out = tune(name, fs, df, seed)
            b = out["best"]
            b["train_val_gap_ap"] = b["train_average_precision"] - b["val_average_precision"]
            num, cat = groups_to_columns(fs)
            b["n_raw_features"] = len(num) + len(cat)
            rows.append(b)
            all_res.append(out["all"])
            print(f"  {name:14s} {fs:9s} valAP={b['val_average_precision']:.4f} "
                  f"trainAP={b['train_average_precision']:.4f} ({time.perf_counter() - t0:.1f}s)")
    return pd.DataFrame(rows), pd.concat(all_res, ignore_index=True)


def complexity_curve(name, fs, params, df, seed):
    pname, values = COMPLEXITY_PARAM[name]
    tr, ytr = _xy(df, ["train"])
    va, yva = _xy(df, ["validation"])
    pts = []
    for v in values:
        p = dict(params)
        p[pname] = v
        m = build_pipeline(name, fs, seed, p).fit(tr, ytr)
        pts.append({"param": pname.replace("clf__", ""), "value": "None" if v is None else v,
                    "train_ap": _ap(m, tr, ytr), "val_ap": _ap(m, va, yva)})
    return pts


def learning_curve(name, fs, params, df, seed):
    tr_all = df[df["split"] == "train"]
    va, yva = _xy(df, ["validation"])
    seasons = sorted(tr_all["season"].unique())
    pts = []
    for k in range(1, len(seasons) + 1):
        part = tr_all[tr_all["season"].isin(seasons[-k:])]  # most recent k training seasons
        y = part["podium"].to_numpy()
        m = build_pipeline(name, fs, seed, params).fit(part, y)
        pts.append({"train_seasons": f"{seasons[-k]}-{seasons[-1]}", "n_seasons": k, "n_rows": int(len(part)),
                    "train_ap": _ap(m, part, y), "val_ap": _ap(m, va, yva)})
    return pts


def rolling_origin(name, fs, params, df, seed, first_eval: int = 2017):
    """Train on all seasons before Y, evaluate on Y, for Y in the development period."""
    dev = df[df["split"].isin(["train", "validation"])]
    out = []
    for y in sorted(dev["season"].unique()):
        if y < first_eval:
            continue
        tr = dev[dev["season"] < y]
        te = dev[dev["season"] == y]
        m = build_pipeline(name, fs, seed, params).fit(tr, tr["podium"])
        r = evaluate_model(m, te, te["podium"], te["race_id"], with_curves=False)
        out.append({"eval_season": int(y), "train_rows": int(len(tr)),
                    "average_precision": r["average_precision"], "roc_auc": r["roc_auc"],
                    "f1": r["f1"], "top3_hit_rate": r["top3_hit_rate"]})
    return out


def perm_importance(model, X, y, fs, seed):
    num, cat = groups_to_columns(fs)
    cols = num + cat
    r = permutation_importance(model, X[cols], y, scoring="average_precision", n_repeats=5,
                               random_state=seed, n_jobs=1)
    return sorted([{"feature": c, "importance_mean": float(m), "importance_std": float(s)}
                   for c, m, s in zip(cols, r.importances_mean, r.importances_std)],
                  key=lambda d: -d["importance_mean"])


def impurity_importance(model):
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    names = model.named_steps["pre"].get_feature_names_out()
    imp = sorted(zip(names, clf.feature_importances_), key=lambda t: -t[1])
    return [{"feature": n, "importance": float(v)} for n, v in imp[:40]]


def fit_and_evaluate(name, fs, params, df, seed, cfg):
    tr, ytr = _xy(df, ["train"])
    va, yva = _xy(df, ["validation"])
    dev, ydev = _xy(df, ["train", "validation"])
    te, yte = _xy(df, ["test"])

    t0 = time.perf_counter()
    dev_model = build_pipeline(name, fs, seed, params).fit(tr, ytr)
    fit_train_only = time.perf_counter() - t0
    m_train = evaluate_model(dev_model, tr, ytr, tr["race_id"], with_curves=False)
    m_val = evaluate_model(dev_model, va, yva, va["race_id"])

    t0 = time.perf_counter()
    final = build_pipeline(name, fs, seed, params).fit(dev, ydev)
    fit_final = time.perf_counter() - t0
    m_test = evaluate_model(final, te, yte, te["race_id"])
    lat = prediction_latency(final, te)

    score, kind = model_scores(final, te)
    preds = te[["season", "round", "race_id", "race_name", "driver_id", "driver_name", "constructor_id",
                "grid", "grid_clean", "finish_position", "podium"]].copy()
    preds["model"] = name
    preds["score"] = score
    preds["predicted"] = final.predict(te)

    n_out = len(final.named_steps["pre"].get_feature_names_out())
    num, cat = groups_to_columns(fs)
    artefact = {
        "model": final, "name": name, "label": MODEL_LABELS[name], "feature_set": fs,
        "numeric_features": num, "categorical_features": cat, "params": params,
        "trained_on_seasons": sorted(int(s) for s in dev["season"].unique()),
        "data_cutoff_season": int(cfg["data"]["study_end_season"]),
        "score_kind": kind, "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
    }
    mdir = resolve(cfg["models"]["dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, mdir / f"{name}.joblib", compress=3)

    result = {
        "model": name, "label": MODEL_LABELS[name], "status": "trained", "feature_set": fs,
        "best_params": {k.replace("clf__", ""): v for k, v in params.items()},
        "n_model_inputs_after_encoding": int(n_out), "score_kind": kind,
        "train": m_train, "validation": m_val, "test": m_test,
        "train_val_gap_ap": m_train["average_precision"] - m_val["average_precision"],
        "train_val_gap_f1": m_train["f1"] - m_val["f1"],
        "timing": {"fit_seconds_train_only": fit_train_only, "fit_seconds_final": fit_final,
                   "final_fit_rows": int(len(dev)), **lat},
        "artefact": f"models/{name}.joblib",
    }
    return result, dev_model, final, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="only feature sets A and A+B+C+D")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    np.random.seed(seed)
    out_dir = resolve(cfg["reports"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_model_dataset(cfg)
    feature_sets = ["A", "A+B+C+D"] if args.fast else cfg["models"]["feature_set_experiments"]
    with open(out_dir / "environment.json", "w") as fh:
        json.dump(environment_info(), fh, indent=2)

    print("[1/4] Feature-set ablation with per-set hyperparameter tuning (validation seasons)")
    ablation, tuning_all = run_ablation(df, seed, feature_sets, args.models)
    ablation.drop(columns=["best_params_raw"]).assign(
        best_params=lambda d: d["best_params"].map(json.dumps, na_action="ignore")
    ).to_csv(out_dir / "ablation.csv", index=False)
    tuning_all.assign(params=tuning_all["params"].map(lambda p: json.dumps(p, default=_json_safe))).to_csv(
        out_dir / "tuning_all.csv", index=False)

    print("[2/4] Final fits, curves and test evaluation")
    results, all_preds, importances = {}, [], []
    for name in args.models:
        sub = ablation[ablation["model"] == name]
        best = sub.loc[sub["val_average_precision"].idxmax()]
        fs, params = best["feature_set"], best["best_params_raw"]
        res, dev_model, final, preds = fit_and_evaluate(name, fs, params, df, seed, cfg)
        va, yva = _xy(df, ["validation"])
        print(f"  {name}: fs={fs} valAP={res['validation']['average_precision']:.4f} "
              f"testAP={res['test']['average_precision']:.4f}")
        res["complexity_curve"] = complexity_curve(name, fs, params, df, seed)
        res["learning_curve"] = learning_curve(name, fs, params, df, seed)
        res["rolling_origin"] = rolling_origin(name, fs, params, df, seed)
        res["permutation_importance_val"] = perm_importance(dev_model, va, yva, fs, seed)
        res["impurity_importance"] = impurity_importance(dev_model)
        results[name] = res
        all_preds.append(preds)
        for row in res["permutation_importance_val"]:
            importances.append({"model": name, **row})

    print("[3/4] Target-policy sensitivity (all starters, non-classified = 0)")
    sens = sensitivity_all_starters(results, cfg, seed)

    print("[4/4] Preferred model + saving outputs")
    pref = max(results.values(), key=lambda r: r["validation"]["average_precision"])
    preferred = {
        "criterion": "highest validation average precision (decided before test evaluation)",
        "model": pref["model"], "label": pref["label"],
        "validation_average_precision": pref["validation"]["average_precision"],
        "test_average_precision": pref["test"]["average_precision"],
    }
    with open(out_dir / "model_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_safe)
    with open(out_dir / "preferred_model.json", "w") as fh:
        json.dump(preferred, fh, indent=2)
    with open(out_dir / "sensitivity_all_starters.json", "w") as fh:
        json.dump(sens, fh, indent=2, default=_json_safe)
    pd.concat(all_preds).to_csv(out_dir / "test_predictions.csv", index=False)
    pd.DataFrame(importances).to_csv(out_dir / "permutation_importance.csv", index=False)
    comparison_table(results).to_csv(out_dir / "comparison.csv", index=False)
    print(comparison_table(results).round(4).to_string(index=False))
    print("Preferred:", preferred)
    from src.evaluate import run_post_analysis
    post = run_post_analysis(cfg)
    print("Grid-only reference AP (val/test):", round(post["grid_baseline"]["validation"]["average_precision"], 4),
          round(post["grid_baseline"]["test"]["average_precision"], 4))


def sensitivity_all_starters(results, cfg, seed):
    """Re-train each selected configuration with every starter included (non-classified = 0)."""
    from src.data_pipeline import load_study_features
    feats = load_study_features(cfg)
    feats = feats[~feats["position_text"].isin(["W", "F"])].copy()
    feats["podium"] = feats["podium_all_starters"].astype(int)
    out = {}
    for name, r in results.items():
        params = {f"clf__{k}": v for k, v in r["best_params"].items()}
        tr, ytr = _xy(feats, ["train"])
        va, yva = _xy(feats, ["validation"])
        m = build_pipeline(name, r["feature_set"], seed, params).fit(tr, ytr)
        ev = evaluate_model(m, va, yva, va["race_id"], with_curves=False)
        out[name] = {"policy": "all_starters", "validation_rows": int(len(va)),
                     "validation_average_precision": ev["average_precision"], "validation_f1": ev["f1"],
                     "validation_top3_hit_rate": ev["top3_hit_rate"],
                     "classified_only_validation_average_precision": r["validation"]["average_precision"]}
    return out


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        row = {"model": r["label"], "feature_set": r["feature_set"]}
        for split in ["train", "validation", "test"]:
            for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
                row[f"{split}_{m}"] = r[split][m]
            row[f"{split}_top3_hit_rate"] = r[split].get("top3_hit_rate")
        row["train_val_gap_ap"] = r["train_val_gap_ap"]
        row["fit_seconds_final"] = r["timing"]["fit_seconds_final"]
        row["per_row_batch_us"] = r["timing"]["per_row_batch_us"]
        row["single_row_median_ms"] = r["timing"]["single_row_median_ms"]
        row["best_params"] = json.dumps(r["best_params"], default=_json_safe)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
