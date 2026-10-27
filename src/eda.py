"""Exploratory data analysis: reusable tables (used by the app) + static report figures.

    python -m src.eda      # writes reports/figures/eda_*.png and reports/results/eda_summary.json

All numbers are computed from the processed dataset on disk; nothing is hard-coded.
Correlations and rates are descriptive and do not imply causation.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.feature_engineering import FEATURE_GROUPS  # noqa: E402

NUMERIC_EDA = ["grid_clean", "drv_avg_pos_5", "drv_podiums_5", "drv_ppr_5", "con_avg_pos_5", "con_ppr_5",
               "drv_career_starts", "drv_races_with_con", "drv_season_pts_before", "con_season_pts_before",
               "drv_career_podium_rate", "podium"]


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })


def class_balance(model_df: pd.DataFrame) -> pd.DataFrame:
    g = model_df.groupby("split")["podium"].agg(rows="size", podiums="sum")
    g["non_podiums"] = g["rows"] - g["podiums"]
    g["podium_rate"] = (g["podiums"] / g["rows"]).round(4)
    return g.reindex([s for s in ["train", "validation", "test"] if s in g.index])


def podium_rate_by_grid(model_df: pd.DataFrame, max_grid: int = 20) -> pd.DataFrame:
    d = model_df[model_df["grid_clean"] <= max_grid]
    return d.groupby("grid_clean")["podium"].agg(rate="mean", n="size").reset_index()


def driver_season_podiums(study: pd.DataFrame) -> pd.DataFrame:
    return study.groupby(["season", "driver_name"])["podium_all_starters"].sum().reset_index(name="podiums")


def circuit_patterns(model_df: pd.DataFrame, min_races: int = 4) -> pd.DataFrame:
    pod = model_df[model_df["podium"] == 1]
    g = pod.groupby("circuit_name").agg(races=("race_id", "nunique"), mean_podium_grid=("grid_clean", "mean"),
                                         podiums_from_outside_top6=("grid_clean", lambda s: int((s > 6).sum())))
    g["share_outside_top6"] = (g["podiums_from_outside_top6"] / (g["races"] * 3)).round(3)
    return g[g["races"] >= min_races].sort_values("mean_podium_grid", ascending=False).reset_index()


def experience_vs_podium(model_df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 0, 10, 25, 50, 100, 150, 200, 400]
    labels = ["rookie (0)", "1-10", "11-25", "26-50", "51-100", "101-150", "151-200", "200+"]
    d = model_df.assign(exp_bin=pd.cut(model_df["drv_career_starts"], bins=bins, labels=labels))
    return d.groupby("exp_bin", observed=False)["podium"].agg(rate="mean", n="size").reset_index()


def missing_history(study: pd.DataFrame) -> dict:
    hist_cols = FEATURE_GROUPS["B"]["numeric"] + ["drv_gap_races", "drv_career_avg_pos", "drv_career_podium_rate"]
    hist_cols = [c for c in hist_cols if c in study.columns]
    return {
        "rookie_entries": int(study["is_rookie"].sum()),
        "returning_entries": int(study["is_returning"].sum()),
        "team_change_entries": int(study["con_changed"].sum()),
        "teams_without_history_entries": int((study["con_has_history"] == 0).sum()),
        "missing_by_feature": {c: int(study[c].isna().sum()) for c in hist_cols},
    }


def correlation(model_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in NUMERIC_EDA if c in model_df.columns]
    return model_df[cols].corr(method="spearman").round(3)


def feature_target_deciles(model_df: pd.DataFrame, col: str, q: int = 10) -> pd.DataFrame:
    d = model_df[[col, "podium"]].dropna()
    d = d.assign(bin=pd.qcut(d[col], q=q, duplicates="drop"))
    g = d.groupby("bin", observed=True).agg(mid=(col, "median"), rate=("podium", "mean"), n=("podium", "size"))
    return g.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Static figures for the report
# ---------------------------------------------------------------------------
def _save(fig, name: str, cfg: dict) -> str:
    out = resolve(cfg["reports"]["figures_dir"]) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def make_report_figures(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = resolve(cfg["data"]["processed_dir"])
    model_df = pd.read_csv(proc / "model_dataset.csv")
    study = pd.read_csv(proc / "features_study.csv")
    figs = {}

    cb = class_balance(model_df)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(cb.index, cb["non_podiums"], label="Non-podium (0)", color="#9aa5b1")
    ax.bar(cb.index, cb["podiums"], bottom=cb["non_podiums"], label="Podium (1)", color="#d1495b")
    for i, (_, r) in enumerate(cb.iterrows()):
        ax.text(i, r["rows"] + 20, f"{r['podium_rate']:.1%} podium", ha="center", fontsize=8)
    ax.set_ylabel("Classified driver entries")
    ax.set_title("Class balance by chronological split")
    ax.legend(fontsize=8, loc="lower right")
    figs["class_balance"] = _save(fig, "eda_class_balance.png", cfg)

    pg = podium_rate_by_grid(model_df)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(pg["grid_clean"].astype(int), pg["rate"], color="#30638e")
    ax.set_xlabel("Starting grid slot")
    ax.set_ylabel("Podium rate")
    ax.set_xticks(range(1, 21))
    ax.set_title("Podium rate by starting grid slot (2014-2024, classified entries)")
    figs["grid_rate"] = _save(fig, "eda_podium_rate_by_grid.png", cfg)

    fig, ax = plt.subplots(figsize=(5, 4))
    d = model_df[model_df["grid_clean"] <= 20]
    h = ax.hist2d(d["grid_clean"], d["finish_position"], bins=[20, 20], range=[[0.5, 20.5], [0.5, 20.5]], cmap="Blues")
    fig.colorbar(h[3], ax=ax, label="Entries")
    ax.plot([1, 20], [1, 20], color="#d1495b", lw=1, ls="--")
    ax.set_xlabel("Grid slot")
    ax.set_ylabel("Classified finishing position")
    ax.set_title("Grid vs finishing position")
    figs["grid_vs_finish"] = _save(fig, "eda_grid_vs_finish.png", cfg)

    ex = experience_vs_podium(model_df)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(ex["exp_bin"].astype(str), ex["rate"], color="#edae49")
    for i, n in enumerate(ex["n"]):
        ax.text(i, ex["rate"].iloc[i] + 0.005, f"n={n}", ha="center", fontsize=7)
    ax.set_xlabel("Driver prior starts (since 2000)")
    ax.set_ylabel("Podium rate")
    ax.set_title("Experience vs podium rate (descriptive, not causal)")
    plt.setp(ax.get_xticklabels(), rotation=20)
    figs["experience"] = _save(fig, "eda_experience.png", cfg)

    corr = correlation(model_df)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title("Spearman correlation of selected features")
    figs["correlation"] = _save(fig, "eda_correlation.png", cfg)

    tp = study.groupby(["season", "team_lineage"])["podium_all_starters"].sum().unstack(fill_value=0)
    top = tp.sum().sort_values(ascending=False).head(6).index
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for t in top:
        ax.plot(tp.index, tp[t], marker="o", ms=3, label=t)
    ax.set_xlabel("Season")
    ax.set_ylabel("Podiums")
    ax.set_title("Podiums per season — top six team lineages")
    ax.legend(fontsize=7, ncol=3)
    figs["team_trend"] = _save(fig, "eda_team_trend.png", cfg)

    summary = {
        "rows": int(len(model_df)), "columns": int(model_df.shape[1]),
        "class_balance": cb.reset_index().to_dict(orient="records"),
        "podium_rate_grid_1": float(pg.loc[pg["grid_clean"] == 1, "rate"].iloc[0]),
        "podium_rate_grid_2_3": float(model_df[model_df["grid_clean"].between(2, 3)]["podium"].mean()),
        "podium_rate_grid_4_6": float(model_df[model_df["grid_clean"].between(4, 6)]["podium"].mean()),
        "podium_rate_grid_11_plus": float(model_df[model_df["grid_clean"] >= 11]["podium"].mean()),
        "share_of_podiums_from_top3_grid": float((model_df[model_df["podium"] == 1]["grid_clean"] <= 3).mean()),
        "missing_history": missing_history(study),
        "experience": ex.assign(exp_bin=ex["exp_bin"].astype(str)).to_dict(orient="records"),
        "spearman_with_podium": corr["podium"].drop("podium").sort_values().to_dict(),
        "figures": figs,
    }
    out = resolve(cfg["reports"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "eda_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


if __name__ == "__main__":
    s = make_report_figures()
    print(json.dumps({k: v for k, v in s.items() if k != "figures"}, indent=2, default=str))
