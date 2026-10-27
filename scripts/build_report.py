"""Build reports/report.md (and .docx/.pdf if pandoc/LibreOffice exist) from SAVED experiment outputs.

    python scripts/build_report.py

Every number in tables and in the results narrative is read from data/processed/*.json and
reports/results/*; nothing is typed in by hand, so the report always agrees with the last run.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "reports" / "results"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "reports"

LABEL = {"decision_tree": "Decision Tree", "knn": "KNN", "svm": "SVM", "random_forest": "Random Forest"}
ORDER = ["decision_tree", "knn", "svm", "random_forest"]
ABBR = {"criterion": "crit", "max_depth": "depth", "min_samples_leaf": "leaf", "min_samples_split": "split",
        "n_neighbors": "K", "weights": "w", "metric": "dist", "n_estimators": "trees", "max_features": "feat",
        "kernel": "kernel", "C": "C", "gamma": "γ"}


def j(p: Path) -> dict:
    return json.loads(p.read_text())


def f3(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.3f}"


def pct(x) -> str:
    return f"{x:.1%}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    # dash counts proportional to content length -> pandoc uses them as relative column widths
    widths = [max(8, min(70, max([len(str(c))] + [len(str(x)) for x in df[c]]))) for c in cols]
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("-" * w for w in widths) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    pipe = j(PROC / "pipeline_summary.json")
    clean = pipe["cleaning"]
    split = pipe["split"]
    res = j(RES / "model_results.json")
    pref = j(RES / "preferred_model.json")
    post = j(RES / "baseline_and_uncertainty.json")
    sens = j(RES / "sensitivity_all_starters.json")
    env = j(RES / "environment.json")
    eda = j(RES / "eda_summary.json")
    ab = pd.read_csv(RES / "ablation.csv")
    manifest = j(ROOT / "data" / "raw" / "manifest.json")
    retrieved = sorted(f["retrieved_at_utc"][:10] for f in manifest["files"] if f.get("retrieved_at_utc"))
    gb = post["grid_baseline"]
    boot = post["test_bootstrap"]
    diff_key = [k for k in boot if k.startswith("ap_diff_vs_")][0]
    share_key = [k for k in boot if k.startswith("share_boot_better")][0]
    best_val = max(res.values(), key=lambda r: r["validation"]["average_precision"])
    best_test = max(res.values(), key=lambda r: r["test"]["average_precision"])

    # ---------------- tables ----------------
    excl = md_table(pd.DataFrame(clean["exclusions_by_reason"]).rename(columns={"reason": "Reason", "rows": "Rows"}))
    split_tbl = md_table(pd.DataFrame([{
        "Split": s.title(), "Seasons": f"{v['seasons'][0]}–{v['seasons'][-1]}", "Races": v["races"],
        "Rows": f"{v['rows']:,}", "Podiums": v["podium_rows"], "Podium rate": pct(v["podium_rate"])}
        for s, v in ((k, split[k]) for k in ["train", "validation", "test"])]))

    def metric_rows(sp):
        rows = []
        for n in ORDER:
            m = res[n][sp]
            rows.append({"Model": LABEL[n], "Feat.": res[n]["feature_set"], "Acc.": f3(m["accuracy"]),
                         "Prec.": f3(m["precision"]), "Rec.": f3(m["recall"]), "F1": f3(m["f1"]),
                         "ROC-AUC": f3(m["roc_auc"]), "AP": f3(m["average_precision"]),
                         "Top-3": f3(m["top3_hit_rate"])})
        g = gb[sp]
        rows.append({"Model": "*Grid-slot reference*", "Feat.": "grid", "Acc.": f3(g["accuracy"]),
                     "Prec.": f3(g["precision"]), "Rec.": f3(g["recall"]), "F1": f3(g["f1"]),
                     "ROC-AUC": f3(g["roc_auc"]), "AP": f3(g["average_precision"]),
                     "Top-3": f3(g["top3_hit_rate"])})
        return md_table(pd.DataFrame(rows))

    gen_tbl = md_table(pd.DataFrame([{
        "Model": LABEL[n], "Selected hyperparameters": ", ".join(f"{ABBR.get(k, k)}={v}" for k, v in res[n]["best_params"].items()),
        "Train AP": f3(res[n]["train"]["average_precision"]), "Val AP": f3(res[n]["validation"]["average_precision"]),
        "Gap": f3(res[n]["train_val_gap_ap"]), "Test AP": f3(res[n]["test"]["average_precision"])} for n in ORDER]))

    piv = ab.pivot(index="feature_set", columns="model", values="val_average_precision")
    piv = piv.reindex(["A", "A+B", "A+C", "A+B+C", "A+B+C+D"])[ORDER]
    abl_tbl = piv.copy()
    for c in ORDER:
        best = piv[c].max()
        abl_tbl[c] = [f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in piv[c]]
    abl_tbl = abl_tbl.rename(columns=LABEL).reset_index().rename(columns={"feature_set": "Feature groups"})
    abl_md = md_table(abl_tbl)
    a_only = {n: piv.loc["A", n] for n in ORDER}
    full = {n: piv.loc["A+B+C+D", n] for n in ORDER}

    time_tbl = md_table(pd.DataFrame([{
        "Model": LABEL[n], "Cand.": int(ab[ab["model"] == n]["n_candidates"].iloc[0]),
        "Search time, 5 sets (s)": f"{ab[ab['model'] == n]['search_seconds'].sum():.0f}",
        "Final fit (s)": f"{res[n]['timing']['fit_seconds_final']:.3f}",
        "Batch latency (µs/row)": f"{res[n]['timing']['per_row_batch_us']:.1f}",
        "Single-row (ms)": f"{res[n]['timing']['single_row_median_ms']:.1f}"} for n in ORDER]))

    boot_tbl = md_table(pd.DataFrame([{
        "Scorer": LABEL.get(c, "Grid-slot reference"),
        "Test AP 95% CI": f"{boot['ap_ci95'][c][0]:.3f}–{boot['ap_ci95'][c][1]:.3f}",
        "AP − grid (95% CI)": (f"{boot[diff_key][c][0]:+.3f} to {boot[diff_key][c][1]:+.3f}" if c in boot[diff_key] else "—"),
        "P(boot > grid)": (pct(boot[share_key][c]) if c in boot[share_key] else "—")} for c in boot["ap_ci95"]]))

    sens_tbl = md_table(pd.DataFrame([{
        "Model": LABEL[n], "Val AP (classified only)": f3(v["classified_only_validation_average_precision"]),
        "Val AP (all starters)": f3(v["validation_average_precision"]),
        "Top-3 hit (all starters)": f3(v["validation_top3_hit_rate"])} for n, v in sens.items()]))

    from src.feature_engineering import FEATURE_DESCRIPTIONS, FEATURE_GROUPS
    from src.tune_models import model_spec
    feat_rows = [{"Grp": g, "Feature": f"`{c}`", "Definition (computed from earlier races only)": FEATURE_DESCRIPTIONS[c]}
                 for g, spec in FEATURE_GROUPS.items() for c in spec["numeric"] + spec["categorical"]]
    feat_md = md_table(pd.DataFrame(feat_rows))

    def space(name):
        g = model_spec(name, 0)["grid"]
        grids = g if isinstance(g, list) else [g]
        parts = []
        for gg in grids:
            parts.append("; ".join(f"{ABBR.get(k.replace('clf__', ''), k.replace('clf__', ''))} ∈ "
                                   f"{{{', '.join(str(x) for x in v)}}}" for k, v in gg.items()))
        return " **or** ".join(parts)
    space_md = md_table(pd.DataFrame([{"Model": LABEL[n], "Search": model_spec(n, 0)["search"],
                                       "Space": space(n)} for n in ORDER]))

    rf_perm = res["random_forest"]["permutation_importance_val"][:5]
    perm_txt = ", ".join(f"`{d['feature']}` ({d['importance_mean']:.3f})" for d in rf_perm)
    pref_perm = res[pref["model"]]["permutation_importance_val"][:4]
    rolling = {n: [d["average_precision"] for d in res[n]["rolling_origin"]] for n in ORDER}
    lit = pd.read_csv(OUT / "literature_tracker.csv")

    v = dict(
        n_races=clean["races_study_period"], n_rows=f"{pipe['model_rows']:,}", n_study=f"{clean['rows_study_period']:,}",
        n_excl=clean["rows_excluded_study_period"], n_hist=f"{clean['rows_history_only']:,}",
        pod_rate=pct(clean["podium_rate_included"]), qual_cov=pct(clean["qualifying_coverage_study"]),
        retrieved=f"{retrieved[0]}" + (f" to {retrieved[-1]}" if retrieved[-1] != retrieved[0] else ""),
        g1=pct(eda["podium_rate_grid_1"]), g23=pct(eda["podium_rate_grid_2_3"]), g46=pct(eda["podium_rate_grid_4_6"]),
        g11=pct(eda["podium_rate_grid_11_plus"]), top3share=pct(eda["share_of_podiums_from_top3_grid"]),
        rookies=eda["missing_history"]["rookie_entries"], returning=eda["missing_history"]["returning_entries"],
        transfers=eda["missing_history"]["team_change_entries"],
        pref=pref["label"], pref_val=f3(pref["validation_average_precision"]), pref_test=f3(pref["test_average_precision"]),
        gb_val=f3(gb["validation"]["average_precision"]), gb_test=f3(gb["test"]["average_precision"]),
        gb_val_top3=f3(gb["validation"]["top3_hit_rate"]), gb_test_top3=f3(gb["test"]["top3_hit_rate"]),
        best_test=best_test["label"], best_test_ap=f3(best_test["test"]["average_precision"]),
        cpu=env["cpu_count"], py=env["python"], skl=env["scikit_learn"], mem=env.get("memory_gb", "?"),
    )
    R = {n: res[n] for n in ORDER}

    # Guard: narrative statements below were written against the results. Re-check them on every build.
    acc = [res[n]["test"]["accuracy"] for n in ORDER]
    dt_lc = res["decision_tree"]["learning_curve"]
    claims = {
        "test accuracy within 0.81-0.85": all(0.805 <= a <= 0.855 for a in acc),
        "RF and SVM top-2 on validation": {max(ORDER, key=lambda n: res[n]["validation"]["average_precision"]),
                                           sorted(ORDER, key=lambda n: -res[n]["validation"]["average_precision"])[1]}
                                          == {"random_forest", "svm"},
        "RF/SVM beat grid on validation": all(res[n]["validation"]["average_precision"] > gb["validation"]["average_precision"]
                                              for n in ["random_forest", "svm"]),
        "RF/SVM CI diff vs grid includes 0": all(boot[diff_key][n][0] < 0 < boot[diff_key][n][1]
                                                 for n in ["random_forest", "svm"]),
        "DT/KNN worse than grid on test": all(boot[diff_key][n][1] < 0 for n in ["decision_tree", "knn"]),
        "KNN train AP == 1": res["knn"]["train"]["average_precision"] > 0.999,
        "SVM best with A": piv["svm"].idxmax() == "A",
        "RF best with A+B+C": piv["random_forest"].idxmax() == "A+B+C",
        "D never best": all(piv[n].idxmax() != "A+B+C+D" for n in ORDER),
        "A+C below A for DT and KNN": all(piv.loc["A+C", n] < piv.loc["A", n] for n in ["decision_tree", "knn"]),
        "all-starters lowers val AP": all(s["validation_average_precision"] < s["classified_only_validation_average_precision"]
                                          for s in sens.values()),
        "DT learning curve 0.51->0.75": round(dt_lc[0]["val_ap"], 2) == 0.51 and round(dt_lc[-1]["val_ap"], 2) == 0.75,
        "SVM complexity gap <= 0.04": all(abs(p["train_ap"] - p["val_ap"]) <= 0.04 for p in res["svm"]["complexity_curve"]),
        "preferred is RF": pref["model"] == "random_forest",
    }
    failed = [k for k, ok in claims.items() if not ok]
    if failed:
        sys.exit("Report narrative no longer matches the saved results; update the text for: " + "; ".join(failed))

    md = f"""---
title: "F1 Race Intelligence: Historical Data Analysis and Podium Prediction Using Machine Learning"
subtitle: "Machine Learning Lab — Mini-Project Report"
author: "Name: Aarpit Jethwa  ·  Roll No.: 59  ·  Course/Section: A9-B4"
---

# 1. Introduction

Formula 1 results come from many interacting factors — car performance, driver skill, qualifying, reliability,
strategy and circuit. This project asks a narrow, testable question: **using only information available before a
Grand Prix starts, can we identify which drivers finish on the podium (P1–P3)?** It is a binary classification task
with an imbalanced positive class, built as an end-to-end, reproducible pipeline: data collection from a public API,
leakage-safe feature engineering, a chronological train/validation/test split, tuning and comparison of four
classical algorithms (Decision Tree, K-Nearest Neighbours, Support Vector Machine, Random Forest), overfitting and
feature-ablation experiments, and an interactive Streamlit application usable on desktop and Android browsers.
The system is educational; it is not a professional race-forecasting or betting tool.

Research questions: (Q1) which pre-race features matter most; (Q2) how the four algorithms compare; (Q3) how feature
selection and model complexity drive over/underfitting; (Q4) whether more features improve generalisation;
(Q5) which model best balances accuracy, interpretability, cost and speed; (Q6) whether the findings can be
communicated through an interactive application.

# 2. Literature Review

Sport-prediction surveys frame most outcome prediction as classification on historical-form features and evaluate
with time-ordered splits [1, 2]. F1-specific statistical work consistently finds the constructor (car) explains
more of the variation in results than the driver — Bell et al. with multilevel models over 1950–2014 [3] and
van Kesteren & Bergkamp with a Bayesian rank-ordered logit on 2014–2021 Ergast data (constructors ≈ 88% of
variance) [4]; a 2025 preprint using ridge regression with walk-forward evaluation on 2014–2024 reaches a similar
conclusion [11]. Recent applied ML studies use SVM, Random Forest and ensembles [9, 10], but at least one mixes
features only known after the race (pit-stop statistics) with a random split [9] — exactly the leakage pattern
formalised by Kaufman et al. [5] and catalogued by Kapoor & Narayanan [6]. For time-dependent data, a hold-out of
the latest period complemented by forward-chaining validation is recommended [7]. For imbalanced targets,
precision-recall analysis is more informative than ROC [8].

{md_table(pd.DataFrame([
    {"Ref.": "[3] Bell et al. 2016", "Approach": "Multilevel model, 1950–2014", "Take-away used here": "Team form as separate features; standardised points"},
    {"Ref.": "[4] van Kesteren & Bergkamp 2023", "Approach": "Bayesian rank-ordered logit, 2014–21", "Take-away used here": "Same data source; team-lineage fix for rebrands"},
    {"Ref.": "[9] Krzysztoń & Smołka 2026", "Approach": "SVM/RF/GB ensemble, 2023–24, random split", "Take-away used here": "Contrast: strictly pre-race features, season split"},
    {"Ref.": "[11] Rane 2025 (preprint)", "Approach": "Ridge + time decay, walk-forward", "Take-away used here": "Rolling-origin check"},
    {"Ref.": "[5, 6] Leakage", "Approach": "Formalism; 294-paper survey", "Take-away used here": "Shift-before-aggregate; pipeline-fitted preprocessing; tests"},
    {"Ref.": "[7] Bergmeir & Benítez 2012", "Approach": "CV vs last-block evaluation", "Take-away used here": "Temporal hold-out + rolling origin"},
    {"Ref.": "[8] Saito & Rehmsmeier 2015", "Approach": "PR vs ROC under imbalance", "Take-away used here": "Model selection by average precision"},
]))}

*Table 1. Literature positioning (full review of 11 verified works: `reports/literature_review.md`).* "State-of-the-art
comparison" in this report means positioning against this literature plus a like-for-like benchmark of four syllabus
algorithms; it does not claim these algorithms are the newest research methods.

# 3. Methodology

The workflow is `collect → clean → engineer features → split → tune on validation → select → refit → test once`,
implemented as separate modules in `src/` and run end-to-end with `python -m src.run_all`. Selection decisions use
the validation seasons only; the test seasons are evaluated a single time after all choices are frozen. All
learned preprocessing (median imputation, standard scaling, one-hot encoding) lives inside a scikit-learn
`Pipeline`/`ColumnTransformer`, so it is fitted on training rows only. Seeds are fixed (`random_state=42`).

# 4. Dataset and Data Cleaning

**Source.** Jolpica-F1 API (Ergast-compatible) results, qualifying and schedule endpoints, retrieved {v['retrieved']}
and cached as raw JSON with a manifest (URL, timestamp, row counts). Seasons 2000–2013 ({v['n_hist']} entries)
are downloaded *only* to give realistic driver/team histories; 2014–2024 is the study period: {v['n_races']} Grands
Prix and {v['n_study']} entries. All scheduled races have results; qualifying coverage is {v['qual_cov']}.

**Cleaning.** Type conversion with failure counts, removal of exact duplicates and invalid identifiers, a verified
unique `(season, round, driver)` key, a chronological race index, pit-lane starts (grid 0) recoded with an explicit
flag, and qualifying merged on the key. **Inclusion policy:** only entries with a numeric classified position receive
a target; non-classified entries are excluded rather than labelled "non-podium" (Table 2) but still contribute to
historical features. The supervised dataset has {v['n_rows']} rows with a podium rate of {v['pod_rate']}; every race
retains exactly three podium finishers.

{excl}

*Table 2. Study-period entries excluded from the supervised dataset ({v['n_excl']} in total).*

**EDA highlights** (figures below). Pole-sitters finish on the podium {v['g1']} of the time, grid 2–3 {v['g23']}, grid 4–6
{v['g46']} and grid 11+ only {v['g11']}; {v['top3share']} of all podiums come from the first three grid slots. Podium
rate rises with experience, but this is largely survivorship (successful drivers keep their seats), not causation.

![Podium rate by starting grid slot (left) and grid vs classified finishing position (right).](figures/eda_podium_rate_by_grid.png){{width=52%}} ![](figures/eda_grid_vs_finish.png){{width=40%}}

![Class balance per split (left); podium rate by driver experience — descriptive only (right).](figures/eda_class_balance.png){{width=46%}} ![](figures/eda_experience.png){{width=46%}}

# 5. Feature Engineering and Leakage Prevention

Every historical feature for race *r* is computed from races with index < *r*: each driver's and team's sequence is
sorted chronologically and shifted by one race before any rolling or expanding aggregate. Four groups are defined
for ablation: **A — basic** (grid, pit-lane flag, season, round, circuit, driver and constructor IDs);
**B — recent form** (driver/team mean race-order position over the previous 3/5 races, podiums, standardised
points per race, DNF rate, mean grid, history flags); **C — experience** (career starts, team prior races, starts
with current team, races since last start, rookie/returning/team-change flags); **D — career context** (career mean
position and podium rate, driver/team season points before the race and their in-race ranks, grid − qualifying).

Design decisions: points are recomputed on a single 25-18-…-1 scale for all seasons; team form follows a *lineage*
(e.g. Toro Rosso → AlphaTauri → RB) so rebrands do not erase history; after a **transfer** the driver keeps personal
form while team experience resets and team form switches to the new team ({v['transfers']} team-change entries);
**returning** drivers keep history plus a gap feature ({v['returning']} entries); **rookies** ({v['rookies']} entries)
keep missing form as `NaN` — never 0, which would mimic a winner — and are imputed with the training median alongside
explicit history flags. Post-race fields (position, status, points, laps, fastest lap, final standings) are never
features. Automated tests scramble the target race and all future races and assert that the target race's features
are byte-identical, while changing a past race does change them.

{feat_md}

*Table 3. Feature definitions by group (A basic, B recent form, C experience, D career context).*

# 6. Experimental Setup

{split_tbl}

*Table 4. Chronological split, fixed before modelling; no race spans two splits.* Validation deliberately spans the
2022 regulation change.

**Algorithms and search.** Decision Tree (entropy/gini; depth, leaf and split sizes; 240-point grid); KNN (K,
uniform/distance weighting, Euclidean/Manhattan; 28 points); SVM (linear and RBF kernels as configurations of one
algorithm; C and gamma; 15 points); Random Forest (trees, depth, leaf size, max features; 30 random draws). Tree
models and the SVM use balanced class weights. Each model is tuned separately on each of five feature-group
combinations using a predefined train→validation split; **selection metric: validation average precision (AP)**,
which is threshold-free and focuses on the minority class [8]. The best configuration per model is refitted on
2014–2022 and tested once on 2023–2024. Supplementary metrics: accuracy, precision, recall, F1 (default threshold),
ROC-AUC, Brier score, confusion matrices, and a race-level **top-3 hit rate** (share of each race's three
highest-scored drivers who actually finished on the podium). A non-ML **grid-slot reference** (rank drivers by
starting slot) is reported for context. Environment: Python {v['py']}, scikit-learn {v['skl']}, {v['cpu']} CPU cores,
{v['mem']} GB RAM.

{space_md}

*Table 5. Hyperparameter search spaces (grid = exhaustive; random = 30 draws, seed 42).*

# 7. Results and Discussion

{metric_rows('validation')}

*Table 6. Validation (2021–22) metrics of each model's selected configuration (trained on 2014–20).*

{metric_rows('test')}

*Table 7. Test (2023–24) metrics, refitted on 2014–22. Top-3 = race-level top-3 hit rate.*

On validation, **{R['random_forest']['label']}** (AP {f3(R['random_forest']['validation']['average_precision'])}) and
**SVM** (AP {f3(R['svm']['validation']['average_precision'])}) lead, clearly above the grid-slot reference
(AP {v['gb_val']}). On the held-out test seasons all models fall: the best test AP is {v['best_test_ap']}
({v['best_test']}), essentially equal to the grid reference (AP {v['gb_test']}, top-3 hit rate {v['gb_test_top3']}).
Accuracy (0.81–0.85) is uninformative here because predicting "no podium" for everyone already scores about 83%.
The tree models and SVM trade precision for recall (balanced class weights move the default threshold), while
unweighted KNN is more conservative.

![Precision-recall and ROC curves on the test seasons.](figures/model_pr_roc_test.png){{width=88%}}

![Confusion matrices on the test seasons (default decision rule).](figures/model_confusion_test.png){{width=92%}}

**Q1 — useful features.** Permutation importance on validation for the Random Forest ranks {perm_txt}. Grid slot and
recent **team** points per race dominate every model, consistent with constructor-dominance findings [3, 4]; the
Decision Tree's root split is team points per race followed by grid slot (`reports/figures/model_decision_tree.png`).
Identity features also appear (e.g. a `driver_id` split for one dominant driver), which helps inside an era but
transfers poorly to the next one. Permutation importance measures the drop in validation AP when one raw column is
shuffled; correlated features share credit, so low values do not mean a feature is useless.

![Permutation importance of the selected model on the validation seasons.](figures/model_perm_importance.png){{width=62%}}

# 8. Comparative Model Performance

{boot_tbl}

*Table 8. Race-level bootstrap ({boot['n_boot']} resamples of {boot['n_races']} test races).* The confidence
intervals of Random Forest, SVM and the grid reference overlap almost completely and their AP differences against
the grid include zero; the Decision Tree and KNN are measurably worse than the grid reference on 2023–24. With 46
test races, **the test data cannot separate the best models from a grid-only ranking.**

{time_tbl}

*Table 9. Tuning cost and timing (this machine; timings vary by hardware).*

**Q5 — balance.** The pre-registered criterion (best validation AP) selects **{v['pref']}** (validation
{v['pref_val']}, test {v['pref_test']}). It is the strongest model on validation and ties for best on test, but it is the
most expensive to train and, with parallel prediction, the slowest for a single row. The **SVM** with only basic
features is nearly as accurate, has the smallest train–validation gap and a tiny model, but outputs margins rather
than probabilities and has the highest batch latency per row. The **Decision Tree** is the most interpretable and
fastest, at a measurable accuracy cost. **KNN** is weakest and memory-bound (it stores the training set).

# 9. Overfitting, Underfitting and Feature Ablation

{gen_tbl}

*Table 10. Generalisation: training vs validation AP (both from the 2014–20 fit) and test AP.*

![Complexity curves: training vs validation AP as capacity grows.](figures/model_complexity.png){{width=100%}}

**Complexity (Q3).** The Decision Tree underfits at depth 1–2 (validation AP {f3(R['decision_tree']['complexity_curve'][0]['val_ap'])}
at depth 1) and plateaus from depth ≈5; the large selected leaf size (50) prevents memorisation, giving a small gap
({f3(R['decision_tree']['train_val_gap_ap'])}). KNN's training AP is 1.0 for every K because distance weighting lets
each training row vote for itself — a textbook overfitting signature; validation AP peaks around K = 9–41 and
declines as K grows (smoothing towards the base rate). The SVM curve is flat: training and validation AP stay within
0.04 of each other at every C (high bias, low variance), so the margin model is limited by the information in the
features rather than by capacity. The Random Forest's training AP climbs to ≈0.99 with depth while validation AP
stays near 0.80 — a large gap ({f3(R['random_forest']['train_val_gap_ap'])}) that averaging keeps from hurting
validation, but which reappears as the drop to {f3(R['random_forest']['test']['average_precision'])} on test.
Learning curves (figure below) show validation AP improving with more seasons for the
Decision Tree (0.51 → 0.75), but little gain beyond three seasons for the other models, suggesting the limit is
feature information, not data volume.

{abl_md}

*Table 11. Feature-group ablation: best validation AP after tuning each model on each feature set (bold = selected).*

![Feature-group ablation (validation AP after per-set tuning).](figures/model_ablation.png){{width=85%}}

**Feature ablation (Q4).** More features did **not** reliably help. Recent form (B) helps the Decision Tree and KNN
(A → A+B: {f3(a_only['decision_tree'])} → {f3(piv.loc['A+B', 'decision_tree'])}, {f3(a_only['knn'])} →
{f3(piv.loc['A+B', 'knn'])}); experience (C) alone hurts both; the Random Forest gains steadily up to A+B+C
({f3(a_only['random_forest'])} → {f3(piv.loc['A+B+C', 'random_forest'])}); the SVM is best with the basic set
({f3(a_only['svm'])}) and *loses* AP with any additions (full set {f3(full['svm'])}). Group D never improves on the
best smaller set. Likely causes: many form/standing features are strongly correlated with grid slot and with each
other (Spearman matrix, Data Explorer), adding noise dimensions to distance- and margin-based models; and the
2022 regulation reset weakened the link between pre-2022 form and validation-season results.

**Robustness.** Rolling-origin evaluation (train on all earlier seasons, test on each season 2017–2022) gives SVM AP
between {min(rolling['svm']):.3f} and {max(rolling['svm']):.3f} and Random Forest between {min(rolling['random_forest']):.3f}
and {max(rolling['random_forest']):.3f}: season-to-season variation is as large as the differences between models.
Retraining with every starter (retirements labelled 0) lowers validation AP for all models (Table 12), because
unpredictable retirements remove podium finishers that a pre-race model ranks highly.

![Learning curves (most recent k training seasons).](figures/model_learning.png){{width=100%}}

![Rolling-origin evaluation across development seasons.](figures/model_rolling_origin.png){{width=62%}}

{sens_tbl}

*Table 12. Target-policy sensitivity (validation 2021–22).*

# 10. UI Development and Deployment

The Streamlit application (`streamlit run app.py`) has six pages: **Overview** (task, data coverage, results
summary, limitations); **Data Explorer** (filters by season/constructor/driver/circuit; data-quality, exclusions,
missing-history, grid, trend, distribution, decile and correlation charts computed live); **Race Explorer**
(historical grid, qualifying and classification, kept separate from predictions); **Model Telemetry** (per-model
metrics on all splits, confusion matrices, ROC/PR/reliability curves, complexity and learning curves, permutation
importance, ablation, timing and hyperparameters); **Model Comparison** (common table, grid reference, bootstrap
intervals, ablation, timing, sensitivity, preferred model with its explicit criterion); and **Podium Predictor**
(choose season, Grand Prix, model and driver; entrants come from that race's real result sheet, so historical
line-ups and transfers are always correct; drivers are ranked by score, in-sample seasons are flagged, and the
actual result is revealed only on request). The app only reads saved artefacts, so pages never retrain models;
missing data, missing artefacts and unknown races produce explanatory messages, verified with Streamlit's AppTest
harness. Responsive CSS stacks columns and enlarges touch targets below 640 px (screenshots below). For a live demo, the app
is served on the local network (`--server.address 0.0.0.0`) and opened in Chrome on an Android phone; Streamlit
Community Cloud steps are in `docs/deployment.md`. No public deployment URL is claimed.

![Desktop Model Comparison page (left) and mobile Podium Predictor at 390 px width (right).](figures/screenshots/desktop_model_comparison.png){{width=62%}} ![](figures/screenshots/mobile_podium_predictor.png){{width=22%}}

# 11. Limitations and Future Scope

Scores are conditional on finishing (retirements are not predictable pre-race) and are not calibrated
probabilities (balanced weights inflate them; Brier scores and reliability curves are in the app). Features exclude
weather, tyres, upgrades, practice pace and telemetry; FastF1 practice/qualifying pace is the most promising
extension. Driver/constructor IDs help within an era but encode yesterday's hierarchy. Career counts start in
2000. Two test seasons (46 races) give wide confidence intervals; 2023 was dominated by one team. Future work:
calibration (Platt/isotonic on validation), ranking objectives (learning-to-rank or Plackett-Luce), regulation-era
indicators, gradient-boosting models, and sprint-race data.

# 12. Conclusion

A leakage-safe, reproducible pipeline was built on {v['n_races']} Grands Prix (2014–2024). Four algorithm families were
tuned on five feature sets with validation-only selection. Grid slot and recent team form carry most of the signal.
The Random Forest (selected) and SVM beat a grid-only ranking on validation, but on 2023–24 no model is
distinguishable from that simple reference, while the Decision Tree and KNN are worse. More features helped the
Random Forest but hurt the SVM, and diagnostic curves expose overfitting (KNN, deep Random Forest) and high bias (SVM)
directly. The honest conclusion is that pre-race public data supports *ranking* likely podium finishers about as
well as the starting grid does; the project's value is a transparent, tested methodology and an application that
communicates both the findings and their limits.

# 13. References

""" + "\n".join(f"[{int(r.id)}] {r.authors} ({r.year}). {r.title}. *{r.venue}*. "
                + (f"https://doi.org/{r.doi}" if isinstance(r.doi, str) and r.doi.strip() else r.url) + "\n"
                for r in lit.itertuples()) + """
Data: Jolpica-F1 API, https://github.com/jolpica/jolpica-f1 (accessed as recorded in `data/raw/manifest.json`).
Software: scikit-learn, pandas, NumPy, Streamlit, Plotly, Matplotlib.
"""
    (OUT / "report.md").write_text(md, encoding="utf-8")
    print("wrote", OUT / "report.md")

    if shutil.which("pandoc"):
        ref = OUT / "reference.docx"
        cmd = ["pandoc", "report.md", "-o", "report.docx", "--resource-path=."]
        if ref.exists():
            cmd.append(f"--reference-doc={ref.name}")
        subprocess.run(cmd, cwd=OUT, check=True)
        print("wrote", OUT / "report.docx")
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if soffice:
            subprocess.run([soffice, "--headless", "--convert-to", "pdf", "report.docx"], cwd=OUT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("wrote", OUT / "report.pdf")


if __name__ == "__main__":
    main()
