"""Plotly figures for the app and Matplotlib figures for the report.

    python -m src.visualizations     # writes reports/figures/model_*.png from saved results

Every figure is drawn from saved experiment outputs or the processed dataset.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import load_config, resolve

# Model colours (consistent across app and report)
MODEL_COLORS = {
    "decision_tree": "#2a9d8f",
    "knn": "#e9c46a",
    "svm": "#8d6cab",
    "random_forest": "#e76f51",
}
LABEL_COLORS = {
    "Decision Tree": MODEL_COLORS["decision_tree"],
    "K-Nearest Neighbours": MODEL_COLORS["knn"],
    "Support Vector Machine": MODEL_COLORS["svm"],
    "Random Forest": MODEL_COLORS["random_forest"],
}
PODIUM_RED = "#e10600"
NEUTRAL = "#8a94a6"


def _compact(fig: go.Figure, height: int = 330, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=40, b=8), title_font_size=14,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0) if legend else None,
                      showlegend=legend, font=dict(size=12))
    return fig


# ---------------------------------------------------------------- EDA (plotly)
def fig_class_balance(cb: pd.DataFrame) -> go.Figure:
    d = cb.reset_index()
    fig = go.Figure([
        go.Bar(name="Non-podium (0)", x=d["split"], y=d["non_podiums"], marker_color=NEUTRAL),
        go.Bar(name="Podium (1)", x=d["split"], y=d["podiums"], marker_color=PODIUM_RED,
               text=[f"{r:.1%}" for r in d["podium_rate"]], textposition="outside"),
    ])
    fig.update_layout(barmode="stack", title="Class balance by split", yaxis_title="Driver entries")
    return _compact(fig)


def fig_grid_rate(pg: pd.DataFrame) -> go.Figure:
    fig = px.bar(pg, x="grid_clean", y="rate", hover_data=["n"], labels={"grid_clean": "Grid slot", "rate": "Podium rate"},
                 title="Podium rate by starting grid slot")
    fig.update_traces(marker_color="#30638e")
    fig.update_yaxes(tickformat=".0%")
    return _compact(fig, legend=False)


def fig_grid_vs_finish(df: pd.DataFrame) -> go.Figure:
    d = df[(df["grid_clean"] <= 20) & (df["finish_position"] <= 20)]
    fig = px.density_heatmap(d, x="grid_clean", y="finish_position", nbinsx=20, nbinsy=20,
                             color_continuous_scale="Blues",
                             labels={"grid_clean": "Grid slot", "finish_position": "Finishing position"},
                             title="Grid slot vs classified finishing position")
    fig.add_trace(go.Scatter(x=[1, 20], y=[1, 20], mode="lines", line=dict(color=PODIUM_RED, dash="dash"),
                             name="grid = finish"))
    return _compact(fig, height=380)


def fig_trend(d: pd.DataFrame, x: str, y: str, color: str, title: str, ylab: str) -> go.Figure:
    fig = px.line(d, x=x, y=y, color=color, markers=True, title=title, labels={y: ylab, x: "Season"})
    return _compact(fig, height=380)


def fig_correlation(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Spearman correlation (selected numeric features)")
    fig.update_traces(textfont_size=9)
    return _compact(fig, height=480, legend=False)


def fig_deciles(d: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.line(d, x="mid", y="rate", markers=True, hover_data=["n"],
                  labels={"mid": f"{feature} (bin median)", "rate": "Podium rate"},
                  title=f"Podium rate across quantile bins of {feature}")
    fig.update_yaxes(tickformat=".0%")
    return _compact(fig, legend=False)


def fig_histogram(df: pd.DataFrame, feature: str) -> go.Figure:
    d = df[[feature, "podium"]].dropna().assign(Outcome=lambda x: x["podium"].map({1: "Podium", 0: "Non-podium"}))
    fig = px.histogram(d, x=feature, color="Outcome", barmode="overlay", histnorm="probability density",
                       color_discrete_map={"Podium": PODIUM_RED, "Non-podium": NEUTRAL},
                       title=f"Distribution of {feature} by outcome")
    return _compact(fig)


# ---------------------------------------------------------------- model telemetry (plotly)
def fig_confusion(cm: dict, title: str) -> go.Figure:
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                    x=["Pred: non-podium", "Pred: podium"], y=["Actual: non-podium", "Actual: podium"], title=title)
    fig.update_coloraxes(showscale=False)
    return _compact(fig, height=300, legend=False)


def fig_curves(results: dict, split: str, kind: str, models: list[str] | None = None) -> go.Figure:
    fig = go.Figure()
    prevalence = None
    for name, r in results.items():
        if models and name not in models:
            continue
        c = (r.get(split) or {}).get("curves") or {}
        if not c.get(kind):
            continue
        prevalence = c.get("prevalence", prevalence)
        if kind == "roc":
            x, y = c["roc"]["fpr"], c["roc"]["tpr"]
            lab = f"{r['label']} (AUC {r[split]['roc_auc']:.3f})"
        else:
            x, y = c["pr"]["recall"], c["pr"]["precision"]
            lab = f"{r['label']} (AP {r[split]['average_precision']:.3f})"
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=lab, line=dict(color=MODEL_COLORS[name], width=2)))
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(color=NEUTRAL, dash="dot")))
        fig.update_layout(title=f"ROC curve — {split}", xaxis_title="False positive rate", yaxis_title="True positive rate")
    else:
        if prevalence is not None:
            fig.add_hline(y=prevalence, line_dash="dot", line_color=NEUTRAL, annotation_text="podium base rate")
        fig.update_layout(title=f"Precision-recall curve — {split}", xaxis_title="Recall", yaxis_title="Precision")
    return _compact(fig, height=380)


def fig_calibration(r: dict, split: str = "test") -> go.Figure | None:
    cal = ((r.get(split) or {}).get("curves") or {}).get("calibration")
    if not cal:
        return None
    fig = go.Figure([
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration", line=dict(color=NEUTRAL, dash="dot")),
        go.Scatter(x=cal["mean_predicted"], y=cal["fraction_positive"], mode="lines+markers",
                   name=r["label"], line=dict(color=MODEL_COLORS[r["model"]])),
    ])
    fig.update_layout(title=f"Reliability (calibration) — {split}", xaxis_title="Mean model score",
                      yaxis_title="Observed podium share")
    return _compact(fig)


def fig_complexity(r: dict) -> go.Figure:
    d = pd.DataFrame(r["complexity_curve"])
    d["value"] = d["value"].astype(str)
    fig = go.Figure([
        go.Scatter(x=d["value"], y=d["train_ap"], mode="lines+markers", name="Training AP", line=dict(color="#264653")),
        go.Scatter(x=d["value"], y=d["val_ap"], mode="lines+markers", name="Validation AP", line=dict(color=PODIUM_RED)),
    ])
    sel = r["best_params"].get(d["param"].iloc[0])
    fig.update_layout(title=f"Complexity curve — {r['label']} ({d['param'].iloc[0]}; selected = {sel})",
                      xaxis_title=d["param"].iloc[0], yaxis_title="Average precision", xaxis_type="category")
    return _compact(fig)


def fig_learning(r: dict) -> go.Figure:
    d = pd.DataFrame(r["learning_curve"])
    fig = go.Figure([
        go.Scatter(x=d["n_rows"], y=d["train_ap"], mode="lines+markers", name="Training AP", line=dict(color="#264653"),
                   text=d["train_seasons"]),
        go.Scatter(x=d["n_rows"], y=d["val_ap"], mode="lines+markers", name="Validation AP", line=dict(color=PODIUM_RED),
                   text=d["train_seasons"]),
    ])
    fig.update_layout(title=f"Learning curve — {r['label']} (most recent k training seasons)",
                      xaxis_title="Training rows", yaxis_title="Average precision")
    return _compact(fig)


def fig_ablation(ab: pd.DataFrame, metric: str = "val_average_precision") -> go.Figure:
    d = ab.copy()
    d["label"] = d["model"].map({"decision_tree": "Decision Tree", "knn": "K-Nearest Neighbours",
                                 "svm": "Support Vector Machine", "random_forest": "Random Forest"})
    fig = px.bar(d, x="feature_set", y=metric, color="label", barmode="group", color_discrete_map=LABEL_COLORS,
                 labels={metric: metric.replace("_", " "), "feature_set": "Feature groups", "label": ""},
                 title=f"Feature-group ablation — {metric.replace('_', ' ')}")
    lo = max(0.0, float(d[metric].min()) - 0.05)
    fig.update_yaxes(range=[lo, min(1.0, float(d[metric].max()) + 0.03)])
    return _compact(fig, height=380)


def fig_metric_compare(cmp: pd.DataFrame, split: str = "test") -> go.Figure:
    metrics = ["precision", "recall", "f1", "roc_auc", "average_precision", "top3_hit_rate"]
    rows = []
    for _, r in cmp.iterrows():
        for m in metrics:
            rows.append({"model": r["model"], "metric": m, "value": r[f"{split}_{m}"]})
    d = pd.DataFrame(rows)
    fig = px.bar(d, x="metric", y="value", color="model", barmode="group", color_discrete_map=LABEL_COLORS,
                 title=f"Metric comparison — {split} set", labels={"value": "", "metric": "", "model": ""})
    fig.update_yaxes(range=[0, 1])
    return _compact(fig, height=380)


def fig_gap(cmp: pd.DataFrame) -> go.Figure:
    d = cmp.melt(id_vars="model", value_vars=["train_average_precision", "validation_average_precision",
                                               "test_average_precision"], var_name="split", value_name="AP")
    d["split"] = d["split"].str.replace("_average_precision", "")
    fig = px.bar(d, x="model", y="AP", color="split", barmode="group",
                 color_discrete_map={"train": "#264653", "validation": "#f4a261", "test": PODIUM_RED},
                 title="Average precision on train / validation / test")
    fig.update_yaxes(range=[0, 1])
    return _compact(fig, height=380)


def fig_timing(cmp: pd.DataFrame) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Final fit time (s)", "Batch latency (µs / row)"))
    colors = [LABEL_COLORS.get(m, NEUTRAL) for m in cmp["model"]]
    fig.add_trace(go.Bar(x=cmp["model"], y=cmp["fit_seconds_final"], marker_color=colors, showlegend=False), 1, 1)
    fig.add_trace(go.Bar(x=cmp["model"], y=cmp["per_row_batch_us"], marker_color=colors, showlegend=False), 1, 2)
    fig.update_xaxes(tickangle=-30)
    return _compact(fig, height=360, legend=False)


def fig_importance(rows: list[dict], title: str, value: str = "importance_mean", top: int = 15) -> go.Figure:
    d = pd.DataFrame(rows).head(top).iloc[::-1]
    err = d["importance_std"] if "importance_std" in d else None
    fig = go.Figure(go.Bar(x=d[value], y=d["feature"], orientation="h", marker_color="#30638e",
                           error_x=dict(type="data", array=err) if err is not None else None))
    fig.update_layout(title=title)
    return _compact(fig, height=max(320, 22 * len(d) + 80), legend=False)


def fig_rolling(results: dict) -> go.Figure:
    fig = go.Figure()
    for name, r in results.items():
        d = pd.DataFrame(r.get("rolling_origin", []))
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["eval_season"], y=d["average_precision"], mode="lines+markers",
                                 name=r["label"], line=dict(color=MODEL_COLORS[name])))
    fig.update_layout(title="Rolling-origin evaluation (train on all earlier seasons)", xaxis_title="Evaluated season",
                      yaxis_title="Average precision")
    return _compact(fig)


# ---------------------------------------------------------------- report figures (matplotlib)
def make_model_report_figures(cfg: dict | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = cfg or load_config()
    res_dir = resolve(cfg["reports"]["results_dir"])
    fig_dir = resolve(cfg["reports"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "model_results.json") as fh:
        results = json.load(fh)
    ab = pd.read_csv(res_dir / "ablation.csv")
    out = {}
    names = list(results)

    def save(fig, name):
        fig.tight_layout()
        p = fig_dir / name
        fig.savefig(p, dpi=160)
        plt.close(fig)
        out[name] = str(p)

    # PR + ROC on test
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for n in names:
        r = results[n]
        c = r["test"]["curves"]
        axes[0].plot(c["pr"]["recall"], c["pr"]["precision"], color=MODEL_COLORS[n],
                     label=f"{r['label']} (AP {r['test']['average_precision']:.3f})")
        axes[1].plot(c["roc"]["fpr"], c["roc"]["tpr"], color=MODEL_COLORS[n],
                     label=f"{r['label']} (AUC {r['test']['roc_auc']:.3f})")
    axes[0].axhline(results[names[0]]["test"]["curves"]["prevalence"], ls=":", color=NEUTRAL, label="base rate")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test 2023-24)")
    axes[1].plot([0, 1], [0, 1], ls=":", color=NEUTRAL)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test 2023-24)")
    for ax in axes:
        ax.legend(fontsize=7)
    save(fig, "model_pr_roc_test.png")

    # Confusion matrices (test)
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.8))
    for ax, n in zip(np.atleast_1d(axes), names):
        cm = results[n]["test"]["confusion_matrix"]
        z = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(z, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, z[i, j], ha="center", va="center", fontsize=10,
                        color="white" if z[i, j] > z.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1], ["actual 0", "actual 1"], fontsize=7)
        ax.set_title(results[n]["label"], fontsize=8)
    save(fig, "model_confusion_test.png")

    # Ablation
    fig, ax = plt.subplots(figsize=(8, 3.4))
    sets = list(dict.fromkeys(ab["feature_set"]))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        sub = ab[ab["model"] == n].set_index("feature_set").reindex(sets)
        ax.bar(np.arange(len(sets)) + i * w - 0.4 + w / 2, sub["val_average_precision"], w,
               color=MODEL_COLORS[n], label=results[n]["label"])
    ax.set_xticks(range(len(sets)), sets)
    ax.set_ylim(max(0, ab["val_average_precision"].min() - 0.08), min(1, ab["val_average_precision"].max() + 0.04))
    ax.set(ylabel="Validation AP", title="Feature-group ablation (tuned per set, validation 2021-22)")
    ax.legend(fontsize=7, ncol=4)
    save(fig, "model_ablation.png")

    # Complexity curves
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 2.9))
    for ax, n in zip(np.atleast_1d(axes), names):
        d = pd.DataFrame(results[n]["complexity_curve"])
        xs = d["value"].astype(str)
        ax.plot(xs, d["train_ap"], "o-", color="#264653", ms=3, label="train")
        ax.plot(xs, d["val_ap"], "o-", color=PODIUM_RED, ms=3, label="validation")
        ax.set_title(f"{results[n]['label']}\n{d['param'].iloc[0]}", fontsize=8)
        ax.tick_params(axis="x", labelrotation=60, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("Average precision")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    save(fig, "model_complexity.png")

    # Learning curves
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 2.7), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        d = pd.DataFrame(results[n]["learning_curve"])
        ax.plot(d["n_rows"], d["train_ap"], "o-", color="#264653", ms=3, label="train")
        ax.plot(d["n_rows"], d["val_ap"], "o-", color=PODIUM_RED, ms=3, label="validation")
        ax.set_title(results[n]["label"], fontsize=8)
        ax.set_xlabel("training rows", fontsize=7)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("Average precision")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    save(fig, "model_learning.png")

    # Permutation importance of the preferred model
    with open(res_dir / "preferred_model.json") as fh:
        pref = json.load(fh)["model"]
    imp = pd.DataFrame(results[pref]["permutation_importance_val"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"], color="#30638e")
    ax.set(xlabel="Drop in validation AP when permuted",
           title=f"Permutation importance — {results[pref]['label']}")
    ax.tick_params(labelsize=7)
    save(fig, "model_perm_importance.png")

    # Timing
    cmp = pd.read_csv(res_dir / "comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    cols = [LABEL_COLORS[m] for m in cmp["model"]]
    axes[0].bar(cmp["model"], cmp["fit_seconds_final"], color=cols)
    axes[0].set(title="Final fit time (s)")
    axes[1].bar(cmp["model"], cmp["per_row_batch_us"], color=cols)
    axes[1].set(title="Batch prediction latency (µs/row)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    save(fig, "model_timing.png")

    # Rolling origin
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for n in names:
        d = pd.DataFrame(results[n]["rolling_origin"])
        ax.plot(d["eval_season"], d["average_precision"], "o-", ms=3, color=MODEL_COLORS[n], label=results[n]["label"])
    ax.set(xlabel="Evaluated season", ylabel="AP", title="Rolling-origin evaluation (development seasons)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "model_rolling_origin.png")

    # Decision tree (readable subset)
    try:
        import joblib
        from sklearn.tree import plot_tree
        art = joblib.load(resolve(cfg["models"]["dir"]) / "decision_tree.joblib")
        m = art["model"]
        fig, ax = plt.subplots(figsize=(11, 4.6))
        plot_tree(m.named_steps["clf"], max_depth=2, feature_names=list(m.named_steps["pre"].get_feature_names_out()),
                  class_names=["non-podium", "podium"], filled=True, impurity=False, proportion=True, fontsize=7, ax=ax)
        ax.set_title("Decision Tree — top three levels (final model)")
        save(fig, "model_decision_tree.png")
    except Exception as err:  # pragma: no cover - figure is optional
        out["decision_tree_error"] = str(err)
    return out


if __name__ == "__main__":
    print(json.dumps(make_model_report_figures(), indent=2))
