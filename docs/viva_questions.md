# Viva preparation — questions and concise answers

Numbers quoted here come from `reports/results/` (run of the committed code). If you re-run the pipeline, re-check
them against `reports/results/comparison.csv` and `baseline_and_uncertainty.json`.

## Problem and target
**1. What exactly are you predicting?**
Whether a driver finishes P1–P3 in a Grand Prix (`podium = 1`), using only information available before the start.
One row = one driver in one race; binary classification.

**2. Why is accuracy not your main metric?**
Only ~17.5% of classified entries are podiums. Predicting "no podium" for everyone gives ~83% accuracy and is useless.
I select models by average precision (area under the precision–recall curve) and also report F1, recall, precision,
ROC-AUC and a race-level top-3 hit rate.

**3. What did you do with retirements and disqualifications?**
They are excluded from the supervised dataset (676 retirements, 31 withdrawals, 9 DSQs in 2014–2024) instead of being
labelled "non-podium", because a retirement is an in-race event a pre-race model cannot know. They still count in
historical features (DNF rate, recent form). A sensitivity run including all starters lowers validation AP for every
model — retirements remove drivers the model rightly ranked highly.

## Data and leakage
**4. Where does the data come from and how is it reproducible?**
Jolpica-F1 API (Ergast-compatible). `python -m src.data_pipeline` downloads, caches raw JSON with a manifest
(URL, timestamp, row counts), then cleans and builds features. `--offline` reruns from the cache.

**5. Why download 2000–2013 if you only model 2014–2024?**
So that in 2014 every driver and team already has a realistic history (career starts, recent form). Those seasons are
never rows in train/validation/test.

**6. What is data leakage and how did you prevent it?**
Using information the model would not have at prediction time. All historical features are shifted by one race
before rolling/expanding aggregation, post-race columns (position, status, points, laps, fastest lap, final standings)
are never features, season standings are rebuilt from earlier races only, and preprocessing is fitted inside a
`Pipeline` on training rows. `tests/test_no_leakage.py` scrambles the target race and all future races and asserts
the target race's features are identical, and that changing a *past* race does change them.

**7. Why a chronological split instead of a random split?**
A random split lets the model see 2023 races while predicting 2022 races from the same season — future information
leaks into training and inflates scores. Train 2014–20, validation 2021–22, test 2023–24; whole races stay together.

**8. Why does validation include 2022?**
2022 introduced new car regulations. Selecting a model that survives a rule change is a more honest test of
generalisation than tuning inside a stable era.

## Features
**9. Which features matter most?**
Grid slot and the team's recent points per race. Permutation importance for the Random Forest puts `grid_clean` first,
then team and driver points-per-race. The Decision Tree's first split is team points per race, then grid. This matches
published findings that the constructor explains most of the variance in F1 results.

**10. How do you handle a driver changing teams?**
Personal form follows the driver; `drv_races_with_con` resets to 0, `con_changed = 1`, and team form becomes that of
the new team. Tested in `test_career_counts_continue_across_transfer`.

**11. And rookies?**
They have no history, so form features are `NaN` — not 0, because 0 would look like a race winner's average
position. The pipeline imputes the training median and adds `drv_has_history` / `is_rookie` flags.

**12. What about team rebrands like Toro Rosso → AlphaTauri → RB?**
A lineage map in `config/settings.yaml` keeps team form continuous across rebrands; the official constructor ID is
still used as the categorical feature.

**13. Why standardised points?**
Scoring rules changed (2014 double points, 2019–24 fastest-lap point). Recomputing all seasons on one 25-18-…-1 scale
keeps form comparable.

## Algorithms
**14. Explain each model in one line.**
Decision Tree: recursive splits that maximise information gain. KNN: majority (or distance-weighted) vote of the K
most similar past entries. SVM: maximum-margin boundary, RBF kernel for non-linearity, C controls softness. Random
Forest: many trees on bootstrap samples with random feature subsets, averaged to reduce variance.

**15. Are linear and RBF SVM two algorithms?**
No — two kernels/configurations of the same algorithm. The grid searched both; RBF (C = 1, γ = 0.003) was selected.

**16. Why does KNN have training AP = 1.0?**
With distance weighting, each training row is its own nearest neighbour at distance 0, so it always predicts its own
label. That is a classic overfitting signature; validation AP (≈0.73) is the honest number.

**17. Why scale for KNN and SVM but not trees?**
Distances and margins depend on feature magnitudes; trees split on thresholds and are scale-invariant.

**18. Why balanced class weights?**
To stop tree models and the SVM from ignoring the minority class. Side effect: the scores are inflated and not
calibrated probabilities (visible in the reliability curves), so the app labels them as scores.

## Tuning, over/underfitting, ablation
**19. How did you tune hyperparameters?**
Grid search (Decision Tree 240, KNN 28, SVM 15 candidates) and randomised search for the Random Forest (30 draws),
each fitted on 2014–20 and scored on 2021–22 via `PredefinedSplit`, selecting by validation AP. The test seasons were
never used in tuning.

**20. Give evidence of overfitting and underfitting.**
Decision Tree at depth 1: training and validation AP both low (≈0.51 / 0.44) — underfitting. Random Forest: training AP
≈0.99 vs validation ≈0.80 — high variance, partly controlled by averaging. SVM: training and validation AP within 0.04
across all C — high bias; the limit is information in the features. KNN: training 1.0 vs validation ≈0.73.

**21. Did adding feature groups help?**
Not consistently. Random Forest improved up to A+B+C; the SVM was best with the basic group only and lost AP with more
features; experience-only (A+C) hurt the tree and KNN; group D never beat a smaller set. Correlated, redundant
features add noise to distance/margin models, and the 2022 rule change weakened older form.

**22. Why no epoch plots?**
These are classical models without epochs. I show complexity curves (capacity parameter vs train/validation AP),
learning curves (more seasons of data), and rolling-origin results instead.

## Results
**23. Which model is best?**
By the pre-registered criterion (highest validation AP) the Random Forest (validation AP 0.805, test AP 0.690).
On the test seasons, however, Random Forest and SVM are statistically indistinguishable from simply ranking drivers by
grid slot (grid reference test AP 0.693; bootstrap intervals of the difference include zero), while the Decision
Tree and KNN are worse than the grid.

**24. So is the ML useless?**
No: on validation both Random Forest and SVM clearly beat the grid ranking (0.805 / 0.799 vs 0.715). On 2023–24 they do
not. Two test seasons (46 races) are a small sample, 2023 was dominated by one team, and our features omit pace
information beyond grid. The honest conclusion is that public pre-race data supports ranking podium candidates about
as well as the grid; the project's contribution is a leakage-safe, tested methodology.

**25. Why did performance drop from validation to test?**
Season-to-season change: regulation-era shifts, different competitive orders in 2023 vs 2024, and identity features
that memorise the previous hierarchy. In the rolling-origin check (2017–22) SVM AP ranges 0.73–0.81 and Random Forest 0.68–0.81 from one season to
the next — as large as the differences between models.

**26. Is the difference between Random Forest and SVM meaningful?**
No. Their test AP intervals overlap heavily. I would deploy the SVM if model size and stability mattered, the Random
Forest if ranking quality on validation mattered most, and the Decision Tree for explanation.

**27. What do precision and recall mean here?**
Recall: share of actual podium finishers the model flagged. Precision: share of flagged drivers who actually reached
the podium. Balanced weights give high recall (0.78–0.86 on test for tree models/SVM) but precision near 0.5.

## App and deployment
**28. How does the predictor avoid leakage and wrong line-ups?**
Entrants are read from that race's own result sheet, features come from the leakage-safe table, and the app warns when
the selected season was part of the model's training data (in-sample). Actual results are hidden until the user
reveals them.

**29. Does the app retrain models?**
No. It only loads saved artefacts (`models/*.joblib`, `reports/results/`). Retraining is `python -m src.train`.

**30. How is it cross-platform?**
It runs in any modern browser. Responsive CSS stacks columns, enlarges touch targets and shrinks charts under 640 px.
For the demo, run with `--server.address 0.0.0.0` and open the laptop's IP on an Android phone on the same Wi-Fi.

## Limitations
**31. Main limitations?**
Conditional on finishing; scores not calibrated; no weather/tyre/practice/telemetry features; identity features
encode eras; career counts start in 2000; small test set; timing figures are machine-dependent.

**32. What would you do next?**
Calibrate scores (Platt/isotonic on validation), add FastF1 practice pace, add regulation-era indicators, try a ranking
objective (learning-to-rank or Plackett–Luce) and gradient boosting, and include sprint races.
