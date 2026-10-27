# Literature review

All eleven works below were checked against a publisher page, DOI resolver (doi.org), Crossref record, arXiv
or RePEc entry (see `literature_tracker.csv`, column `verified_on_url`). Where the full text could not be
opened, findings are taken from the abstract and this is stated. Findings are paraphrased.

> **Scope note.** "State-of-the-art comparison" in this project means (a) positioning the work against published
> research and (b) benchmarking four classical algorithms against each other on one task. It does **not** claim
> that Decision Trees, KNN, SVMs or Random Forests are the newest research methods.

## 1. Sport outcome prediction with machine learning

**[1] Bunker, R. P. & Thabtah, F. (2019). A machine learning framework for sport result prediction.**
*Applied Computing and Informatics*, 15(1), 27–33. https://doi.org/10.1016/j.aci.2017.09.005
Reviews neural-network approaches to sport result prediction and proposes a staged framework (domain
understanding → data → features → model → evaluation). No empirical benchmark. Typical predictors are historical
results and performance indicators. *Relevance:* the pipeline structure and the use of historical-form features
as pre-match predictors.

**[2] Horvat, T. & Job, J. (2020). The use of machine learning in sport outcome prediction: A review.**
*WIREs Data Mining and Knowledge Discovery*, 10(5), e1380. https://doi.org/10.1002/widm.1380
Screens 100+ studies. Most frame prediction as classification, perform feature selection, and evaluate with a
chronological split (sometimes alongside k-fold). Narrative review without meta-analysis. *Relevance:* supports
the classification framing and the chronological split adopted here.

## 2. Formula 1 performance modelling

**[3] Bell, A., Smith, J., Sabel, C. E. & Jones, K. (2016). Formula for success: Multilevel modelling of Formula
One driver and constructor performance, 1950–2014.** *Journal of Quantitative Analysis in Sports*, 12(2), 99–112.
https://doi.org/10.1515/jqas-2015-0050
Cross-classified multilevel model of standardised points (1950–2014) with team, team-year and driver levels. Team
effects outweigh driver effects and have grown over time; team effects are weaker in wet races and on street
circuits. Explanatory, with no out-of-sample test. *Relevance:* motivates separate constructor-form features and
standardised points across scoring eras.

**[4] van Kesteren, E.-J. & Bergkamp, T. (2023). Bayesian analysis of Formula One race results: disentangling
driver skill and constructor advantage.** *Journal of Quantitative Analysis in Sports*, 19(4), 273–293.
https://doi.org/10.1515/jqas-2022-0021 (preprint arXiv:2203.08489)
Bayesian multilevel rank-ordered logit on Ergast data 2014–2021 (160 races), with seasonal form. Constructors
explain roughly 88% of variance in results. Non-finishers are excluded and rebrands treated as new teams; the
authors do not position it as a forecasting model. *Relevance:* same data source and era; their exclusion of
non-finishers parallels our classified-only policy, while our team-lineage mapping addresses their rebrand issue.

**[11] Rane, S. (2025). Predicting Formula 1 race outcomes: Decomposing the roles of drivers and constructors
through linear modeling.** arXiv:2508.00200 (preprint, not peer reviewed). https://arxiv.org/abs/2508.00200
Regularised adjusted plus-minus (ridge) with time decay on Ergast/Jolpica results 2014–2024, evaluated walk-forward
(train on past races, predict the next); reports mean Kendall's τ ≈ 0.63 and a constructor share of about 64%
of variance, plus a top-3 logistic check. Excludes non-finishers; outputs rankings rather than probabilities.
*Relevance:* strongest methodological precedent for leakage-free walk-forward evaluation on the same seasons.

## 3. Recent machine-learning F1 prediction

**[9] Krzysztoń, S. & Smołka, J. (2026). Application of machine learning for predicting Formula 1 race results.**
*Journal of Computer Sciences Institute*, 38, 81–86. https://doi.org/10.35784/jcsi.8462
OpenF1 data for 2023–2024 (~1,600 records, 15 features incl. grid, qualifying, previous wins and pit-stop
statistics); four-class target (winner, P2–3, P4–10, P11–20); soft-voting ensemble of RBF-SVM, Random Forest and
Gradient Boosting with Optuna tuning, 70/30 split + 5-fold CV. Macro-F1 ≈ 0.78; the winner class was hardest.
Accuracy figures differ between the abstract and the results table. The split is not described as
chronological and pit-stop statistics are only known after a race (our assessment: a leakage risk).
*Relevance:* closest recent comparator; contrasts with our strictly pre-race features and season-based split.

**[10] El Haber, E., Sawaya, E., Attieh, M., Tannous, A., Ghazaly, W. & Owayjan, M. (2025). Formula 1 race winner
prediction using Random Forest and SHAP analysis.** *2025 Int. Conf. on Control, Automation, and Instrumentation
(IC2AI)*, IEEE, 1270–1274. https://doi.org/10.1109/IC2AI62984.2025.10932140
Random Forest for race-winner prediction with SHAP explanations. Only the abstract and metadata were verified;
data, split and numeric results could not be checked and are not cited. *Relevance:* precedent for Random Forests
plus post-hoc attribution in F1; we use permutation importance for the same purpose.

## 4. Leakage and temporal validation

**[5] Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in data mining.** *ACM Transactions on
Knowledge Discovery from Data*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579
Defines leakage as information about the target the model should not legitimately have, drawing on competition
case studies, and recommends a strict "learn–predict" separation in time and data handling. Conceptual.
*Relevance:* basis for building every feature from earlier races only and excluding post-race fields.

**[6] Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based
science.** *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
Survey finding leakage in 294 papers across 17 fields; proposes a taxonomy (e.g. preprocessing on full data,
temporal leakage, illegitimate features) and "model info sheets". In their civil-war case study, complex models
lost their advantage once leakage was fixed. *Relevance:* our preprocessing is fitted inside the pipeline on
training rows only, and the split is temporal; both are tested automatically.

**[7] Bergmeir, C. & Benítez, J. M. (2012). On the use of cross-validation for time series predictor
evaluation.** *Information Sciences*, 191, 192–213. https://doi.org/10.1016/j.ins.2011.12.028
Compares cross-validation variants with last-block evaluation for time-series predictors; blocked
cross-validation is recommended as a robust default. Studies univariate regression, not grouped classification.
*Relevance:* justifies our hold-out of the latest seasons plus a forward-chaining (rolling-origin) check.

## 5. Evaluation under class imbalance

**[8] Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when
evaluating binary classifiers on imbalanced datasets.** *PLOS ONE*, 10(3), e0118432.
https://doi.org/10.1371/journal.pone.0118432
Simulations and a literature review show ROC curves look unchanged as imbalance grows, while precision-recall
curves expose poor precision; PR curves are recommended for imbalanced data. *Relevance:* podiums are ~17% of
classified entries, so we select models by average precision and report PR curves alongside ROC.

## 6. Comparison table

| # | Work | Data | Method | Evaluation | Key limitation | Adopted here |
|---|---|---|---|---|---|---|
| 1 | Bunker & Thabtah 2019 | Review | ANN-focused framework | — | No empirical test | Staged pipeline |
| 2 | Horvat & Job 2020 | 100+ studies | Review | — | Narrative only | Classification + chronological split |
| 3 | Bell et al. 2016 | F1 1950–2014 | Multilevel model | Variance partition | Explanatory only | Constructor form; standardised points |
| 4 | van Kesteren & Bergkamp 2023 | Ergast 2014–21 | Bayesian rank-ordered logit | LOO-CV, PPC | Not for forecasting; rebrands split | Same source/era; team lineage |
| 9 | Krzysztoń & Smołka 2026 | OpenF1 2023–24 | SVM+RF+GB ensemble | Macro-F1, accuracy | Random split; post-race features | Contrast: pre-race only, season split |
| 10 | El Haber et al. 2025 | Not verified | RF + SHAP | Not verified | Abstract only | Feature attribution idea |
| 11 | Rane 2025 | Jolpica 2014–24 | Ridge (RAPM) + time decay | Walk-forward Kendall τ | Preprint; rankings only | Rolling-origin evaluation |
| 5 | Kaufman et al. 2012 | Case studies | Leakage formalism | — | Conceptual | Shift-before-aggregate; leakage tests |
| 6 | Kapoor & Narayanan 2023 | 294 papers | Taxonomy | Reproduction | Lower-bound count | Pipeline-fitted preprocessing |
| 7 | Bergmeir & Benítez 2012 | Time series | CV vs hold-out | Simulation | Regression only | Temporal hold-out + rolling origin |
| 8 | Saito & Rehmsmeier 2015 | Simulated + review | PR vs ROC | — | Idealised scores | AP for selection; PR curves |

## 7. What this project adds (as an educational implementation)
- A podium classification task on Jolpica 2014–2024 with **only pre-race information**, verified by automated
  perturbation tests — the main gap visible in recent applied F1 ML work [9].
- Explicit handling of transfers, returning drivers, rookies and team rebrands, which the statistical studies
  [3, 4] treat as modelling simplifications.
- A transparent, like-for-like benchmark of four syllabus algorithms with per-feature-set tuning, overfitting
  diagnostics and timing — not a claim of superior forecasting accuracy.
