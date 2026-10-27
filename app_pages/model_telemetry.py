import pandas as pd
import streamlit as st

from app_pages.common import fmt, load_result_csv, load_result_json, model_label, note, require_results, setup_page
from src.predict import MODEL_NAMES
from src.visualizations import (fig_calibration, fig_complexity, fig_confusion, fig_curves, fig_importance,
                                fig_learning)

INTERPRET = {
    "decision_tree": "A single tree splits on the most informative feature at each node. Depth controls capacity: "
                     "shallow trees underfit (they can only express a few grid/form rules); deep trees memorise "
                     "individual races, so training AP rises while validation AP falls.",
    "knn": "KNN labels an entry by its nearest past entries in scaled feature space. Small K = high variance "
           "(overfits to near-identical past entries); large K smooths towards the base rate (underfits). "
           "Prediction cost grows with the training set because every query scans stored rows.",
    "svm": "The SVM maximises the margin between podium and non-podium entries. C trades margin width against "
           "training errors (large C = more complex boundary); the RBF kernel adds non-linear boundaries controlled "
           "by gamma. Scores are margin distances, not probabilities.",
    "random_forest": "A forest averages many de-correlated trees (bootstrap rows + random feature subsets), which "
                     "reduces variance compared with one tree. Impurity importance is biased towards high-cardinality "
                     "and continuous features, so permutation importance on validation is reported as well.",
}

setup_page()
st.title("Model Telemetry")
results = require_results()
if results is None:
    st.stop()

choices = [m for m in MODEL_NAMES]
name = st.selectbox("Model", choices, format_func=model_label)
r = results.get(name)
if r is None:
    st.warning(f"{model_label(name)} has no saved results. Run `python -m src.train`.")
    st.stop()

st.markdown(f"<span class='f1-tag'>status: {r['status']}</span><span class='f1-tag'>features: {r['feature_set']}</span>"
            f"<span class='f1-tag'>{r['n_model_inputs_after_encoding']} inputs after encoding</span>"
            f"<span class='f1-tag'>artefact: {r['artefact']}</span>", unsafe_allow_html=True)
st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Validation AP", fmt(r["validation"]["average_precision"]))
c2.metric("Test AP", fmt(r["test"]["average_precision"]))
c3.metric("Train–val AP gap", fmt(r["train_val_gap_ap"]))
c4.metric("Test F1", fmt(r["test"]["f1"]))

t1, t2, t3, t4, t5 = st.tabs(["Metrics", "Curves", "Fit diagnostics", "Feature sets", "Timing & params"])

with t1:
    rows = []
    for split in ["train", "validation", "test"]:
        m = r[split]
        rows.append({"split": split, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                     "F1": m["f1"], "ROC-AUC": m["roc_auc"], "AP": m["average_precision"],
                     "top-3 hit rate": m.get("top3_hit_rate"), "Brier": m.get("brier"),
                     "podiums (support)": m["support_pos"], "predicted podiums": m["predicted_pos"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    note("Train metrics come from the model fitted on 2014–2020; validation from the same model on 2021–22. "
         "Test metrics come from the same configuration refitted on 2014–2022 and evaluated once on 2023–24. "
         "Precision = share of predicted podiums that were real; recall = share of real podiums that were found.")
    a, b = st.columns(2)
    a.plotly_chart(fig_confusion(r["validation"]["confusion_matrix"], "Confusion matrix — validation"))
    b.plotly_chart(fig_confusion(r["test"]["confusion_matrix"], "Confusion matrix — test"))

with t2:
    split = st.radio("Split", ["test", "validation"], horizontal=True)
    st.plotly_chart(fig_curves({name: r}, split, "pr"))
    st.plotly_chart(fig_curves({name: r}, split, "roc"))
    cal = fig_calibration(r, split)
    if cal is not None:
        st.plotly_chart(cal)
        note("Points below the diagonal mean the score overstates the observed podium share — expected with "
             "balanced class weights. Treat scores as rankings, not probabilities.")
    else:
        note("This model outputs a decision score, so no reliability curve is drawn.")

with t3:
    st.plotly_chart(fig_complexity(r))
    st.plotly_chart(fig_learning(r))
    note("Other hyperparameters are held at their selected values. A widening gap between training and validation "
         "curves indicates overfitting; both low and close together indicates underfitting / limited signal. "
         "Classical models have no training epochs, so complexity and learning curves replace epoch plots.")
    st.info(INTERPRET[name])
    imp = r.get("permutation_importance_val") or []
    if imp:
        st.plotly_chart(fig_importance(imp, "Permutation importance (validation AP drop)"))
    if r.get("impurity_importance"):
        with st.expander("Impurity-based importance (encoded inputs, biased — read with care)"):
            st.dataframe(pd.DataFrame(r["impurity_importance"]).round(4), hide_index=True)

with t4:
    ab = load_result_csv("ablation.csv")
    if ab is not None:
        sub = ab[ab["model"] == name][["feature_set", "n_raw_features", "train_average_precision",
                                        "val_average_precision", "train_val_gap_ap", "val_f1", "val_roc_auc",
                                        "mean_fit_time_s", "best_params"]]
        st.dataframe(sub.round(4), hide_index=True)
        note("Each feature set was tuned separately on the validation seasons; the selected set is the row with "
             "the highest validation AP.")

with t5:
    t = r["timing"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Final fit (s)", f"{t['fit_seconds_final']:.3f}")
    c2.metric("Batch latency (µs/row)", f"{t['per_row_batch_us']:.1f}")
    c3.metric("Single-row latency (ms)", f"{t['single_row_median_ms']:.2f}")
    st.json(r["best_params"])
    env = load_result_json("environment.json")
    if env:
        note(f"Measured on {env.get('platform')} · Python {env.get('python')} · scikit-learn {env.get('scikit_learn')}"
             f" · {env.get('cpu_count')} CPUs. Timings vary by machine and dataset size.")
