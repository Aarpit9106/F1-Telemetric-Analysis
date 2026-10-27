import pandas as pd
import streamlit as st

from app_pages.common import fmt, load_result_csv, load_result_json, note, require_results, setup_page
from src.visualizations import fig_ablation, fig_curves, fig_gap, fig_metric_compare, fig_rolling, fig_timing

setup_page()
st.title("Model Comparison")
results = require_results()
cmp = load_result_csv("comparison.csv")
pref = load_result_json("preferred_model.json")
if results is None or cmp is None:
    st.stop()

if pref:
    st.success(f"**Preferred model: {pref['label']}** — selection criterion: {pref['criterion']}. "
               f"Validation AP {fmt(pref['validation_average_precision'])}, test AP {fmt(pref['test_average_precision'])}.")

split = st.radio("Metrics table split", ["test", "validation", "train"], horizontal=True)
cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "top3_hit_rate"]
tbl = cmp[["model", "feature_set"] + [f"{split}_{c}" for c in cols] +
          ["train_val_gap_ap", "fit_seconds_final", "per_row_batch_us"]].copy()
tbl.columns = ["Model", "Features"] + [c.replace("_", " ") for c in cols] + ["train-val AP gap", "fit s", "µs/row"]
st.dataframe(tbl.round(3), hide_index=True)
note("The podium base rate is ~17%, so accuracy alone is misleading (always predicting 'no podium' scores ~83%). "
     "AP and the race-level top-3 hit rate are the most informative columns.")

post = load_result_json("baseline_and_uncertainty.json")
if post:
    gb = post["grid_baseline"].get(split, {})
    st.markdown(f"**Non-ML reference — rank by grid slot** ({split}): AP {fmt(gb.get('average_precision'))}, "
                f"F1 {fmt(gb.get('f1'))}, top-3 hit rate {fmt(gb.get('top3_hit_rate'))}. "
                "Any model worth deploying should beat this simple rule.")

t1, t2, t3, t4 = st.tabs(["Performance", "Generalisation", "Feature ablation", "Cost & robustness"])
with t1:
    st.plotly_chart(fig_metric_compare(cmp, split))
    st.plotly_chart(fig_curves(results, "test", "pr"))
    st.plotly_chart(fig_curves(results, "test", "roc"))
with t2:
    st.plotly_chart(fig_gap(cmp))
    if post:
        b = post["test_bootstrap"]
        diff_key = [k for k in b if k.startswith("ap_diff_vs_")][0]
        share_key = [k for k in b if k.startswith("share_boot_better")][0]
        rows = [{"model": results[m]["label"] if m in results else "Grid-slot reference",
                 "test AP 95% CI": f"{b['ap_ci95'][m][0]:.3f} – {b['ap_ci95'][m][1]:.3f}",
                 "AP − grid reference (95% CI)": (f"{b[diff_key][m][0]:+.3f} – {b[diff_key][m][1]:+.3f}"
                                                  if m in b[diff_key] else "—"),
                 "bootstrap share better than grid": (fmt(b[share_key][m], pct=True) if m in b[share_key] else "—")}
                for m in b["ap_ci95"]]
        st.subheader("Is the difference meaningful?")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        note(f"Race-level bootstrap ({b['n_boot']} resamples of the {b['n_races']} test races). Overlapping intervals "
             "and difference intervals that include 0 mean the test data cannot separate the models.")
    note("Large train → validation drops indicate overfitting. Validation → test differences also reflect "
         "season-to-season change (e.g. the 2022 regulation reset and one team's 2023 dominance).")
    st.plotly_chart(fig_rolling(results))
    note("Rolling origin: for each development season Y, the selected configuration is trained on all seasons "
         "before Y and evaluated on Y. Configurations were chosen on 2021–22, so those two points are optimistic.")
with t3:
    ab = load_result_csv("ablation.csv")
    if ab is not None:
        metric = st.radio("Metric", ["val_average_precision", "train_average_precision", "train_val_gap_ap"],
                          horizontal=True)
        st.plotly_chart(fig_ablation(ab, metric))
        piv = ab.pivot(index="feature_set", columns="model", values="val_average_precision").round(4)
        st.dataframe(piv)
        note("A = basic, B = recent form, C = experience, D = career context/standings. Each cell is the best "
             "validation AP after tuning that model on that feature set.")
with t4:
    st.plotly_chart(fig_timing(cmp))
    sens = load_result_json("sensitivity_all_starters.json")
    if sens:
        st.subheader("Target-policy sensitivity")
        s = pd.DataFrame([{"model": results[k]["label"], "classified-only val AP": v[
            "classified_only_validation_average_precision"], "all-starters val AP": v["validation_average_precision"],
                           "all-starters top-3 hit": v["validation_top3_hit_rate"]} for k, v in sens.items()])
        st.dataframe(s.round(3), hide_index=True)
        note("Re-training with every starter (retirements labelled 0) tests how much the classified-only policy "
             "changes the picture. The two APs are on different populations, so compare them cautiously.")
    env = load_result_json("environment.json")
    if env:
        note(f"Timings measured on {env.get('platform')}, {env.get('cpu_count')} CPUs, Python {env.get('python')}, "
             f"scikit-learn {env.get('scikit_learn')}. They will differ on other machines.")
