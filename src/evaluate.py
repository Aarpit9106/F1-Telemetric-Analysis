"""Evaluation metrics and curves for podium classification (positive class: podium = 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)

POSITIVE = 1


def model_scores(model, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Continuous podium score: probability if available, else decision function."""
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(clf, "predict_proba") and getattr(clf, "probability", True):
        return model.predict_proba(X)[:, 1], "probability"
    return model.decision_function(X), "decision_function"


def classification_metrics(y_true, y_pred, score, score_kind: str = "probability") -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    both = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)) if both else None,
        "average_precision": float(average_precision_score(y_true, score)) if both else None,
        "brier": float(brier_score_loss(y_true, score)) if (score_kind == "probability" and both) else None,
        "support_pos": int(y_true.sum()),
        "support_neg": int((1 - y_true).sum()),
        "predicted_pos": int(y_pred.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_auc_note": None if both else "unavailable: split contains only one class",
    }
    return out


def top3_hit_rate(race_ids, y_true, score) -> float:
    """Race-level metric: share of predicted top-3 (by score) per race that actually finished on the podium."""
    d = pd.DataFrame({"race": np.asarray(race_ids), "y": np.asarray(y_true), "s": np.asarray(score)})
    d["rank"] = d.groupby("race")["s"].rank(ascending=False, method="first")
    top = d[d["rank"] <= 3]
    return float(top["y"].mean()) if len(top) else float("nan")


def curves(y_true, score, score_kind: str, n_points: int = 120) -> dict:
    """ROC, PR and (for probabilities) calibration curves, thinned for storage."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"roc": None, "pr": None, "calibration": None}

    def thin(*arrs):
        n = len(arrs[0])
        idx = np.unique(np.linspace(0, n - 1, min(n, n_points)).astype(int))
        return [np.asarray(a)[idx].round(5).tolist() for a in arrs]

    fpr, tpr, _ = roc_curve(y_true, score)
    prec, rec, _ = precision_recall_curve(y_true, score)
    out = {"roc": dict(zip(["fpr", "tpr"], thin(fpr, tpr))),
           "pr": dict(zip(["precision", "recall"], thin(prec, rec))),
           "prevalence": float(y_true.mean()),
           "calibration": None}
    if score_kind == "probability":
        frac, mean_pred = calibration_curve(y_true, np.clip(score, 0, 1), n_bins=10, strategy="quantile")
        out["calibration"] = {"mean_predicted": mean_pred.round(4).tolist(), "fraction_positive": frac.round(4).tolist()}
    return out


def evaluate_model(model, X: pd.DataFrame, y, race_ids=None, with_curves: bool = True) -> dict:
    score, kind = model_scores(model, X)
    pred = model.predict(X)
    m = classification_metrics(y, pred, score, kind)
    m["score_kind"] = kind
    if race_ids is not None:
        m["top3_hit_rate"] = top3_hit_rate(race_ids, y, score)
    if with_curves:
        m["curves"] = curves(y, score, kind)
    return m


# ---------------------------------------------------------------------------
# Reference baseline and uncertainty (run after training: python -m src.evaluate)
# ---------------------------------------------------------------------------
def grid_baseline(df: pd.DataFrame) -> dict:
    """Non-ML reference: score = -grid slot; predict podium if starting in the top three."""
    out = {}
    for split in ["train", "validation", "test"]:
        d = df[df["split"] == split]
        if d.empty:
            continue
        score = -d["grid_clean"].to_numpy()
        pred = (d["grid_clean"] <= 3).astype(int).to_numpy()
        m = classification_metrics(d["podium"], pred, score, "decision_function")
        m["top3_hit_rate"] = top3_hit_rate(d["race_id"], d["podium"], score)
        out[split] = m
    return out


def race_bootstrap(pred: pd.DataFrame, score_cols: list[str], n_boot: int = 2000, seed: int = 42) -> dict:
    """Resample whole races with replacement; 95% percentile CI of AP for each score column,
    and of the paired AP difference between each column and the first one."""
    rng = np.random.default_rng(seed)
    races = pred["race_id"].unique()
    groups = {r: g for r, g in pred.groupby("race_id")}
    boots = {c: [] for c in score_cols}
    diffs = {c: [] for c in score_cols[1:]}
    for _ in range(n_boot):
        sample = pd.concat([groups[r] for r in rng.choice(races, size=len(races), replace=True)])
        vals = {c: average_precision_score(sample["podium"], sample[c]) for c in score_cols}
        for c in score_cols:
            boots[c].append(vals[c])
        for c in score_cols[1:]:
            diffs[c].append(vals[c] - vals[score_cols[0]])
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {
        "n_boot": n_boot, "n_races": int(len(races)),
        "ap_ci95": {c: ci(v) for c, v in boots.items()},
        f"ap_diff_vs_{score_cols[0]}_ci95": {c: ci(v) for c, v in diffs.items()},
        f"share_boot_better_than_{score_cols[0]}": {c: float(np.mean(np.array(v) > 0)) for c, v in diffs.items()},
    }


def run_post_analysis(cfg: dict | None = None) -> dict:
    import json

    from src.config import load_config, resolve
    cfg = cfg or load_config()
    res_dir = resolve(cfg["reports"]["results_dir"])
    df = pd.read_csv(resolve(cfg["data"]["processed_dir"]) / "model_dataset.csv")
    base = grid_baseline(df)
    preds = pd.read_csv(res_dir / "test_predictions.csv")
    gcol = "grid_clean" if "grid_clean" in preds.columns else "grid"
    wide = preds.pivot_table(index=["race_id", "driver_id", "podium", gcol], columns="model",
                             values="score").reset_index()
    wide["grid_baseline"] = -wide[gcol].where(wide[gcol] > 0, 25)
    models = [c for c in ["decision_tree", "knn", "svm", "random_forest"] if c in wide.columns]
    boot = race_bootstrap(wide, ["grid_baseline"] + models, seed=int(cfg["project"]["random_seed"]))
    out = {"grid_baseline": base, "test_bootstrap": boot,
           "note": "Grid baseline is a non-ML reference (rank by starting slot), not one of the four models."}
    with open(res_dir / "baseline_and_uncertainty.json", "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    import json as _json
    r = run_post_analysis()
    print(_json.dumps({"baseline": {k: {m: r["grid_baseline"][k][m] for m in ["average_precision", "f1",
                                                                             "top3_hit_rate"]}
                                    for k in r["grid_baseline"]}, "bootstrap": r["test_bootstrap"]}, indent=2))
