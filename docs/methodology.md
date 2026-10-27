# Methodology

## 1. Task
Binary classification. One observation = one driver in one Grand Prix. `podium = 1` if the classified finishing
position is 1, 2 or 3, else 0. Positive class throughout: `podium = 1`.

## 2. Data and inclusion policy
- Source: Jolpica-F1 API (Ergast-compatible), seasons 2000–2024 downloaded and cached (`data/README.md`).
- 2000–2013 are **history-only**: they give realistic career counts and recent form at the start of 2014, but
  are never model rows. Starting in 2000 covers the full career of every driver on the 2014+ grid.
- Study period 2014–2024: 228 Grands Prix, 4,626 entries (including withdrawals), 3,910 classified entries.
- **Primary policy — classified only.** 716 entries are excluded: 676 retirements (R), 31 withdrawals / did not
  start (W), 9 disqualifications (D). Every race keeps exactly three podium finishers (684 positives, 17.5%).
- Sensitivity policy (reported separately): all starters, non-classified = 0.

## 3. Cleaning steps (`src/data_cleaning.py`)
1. Shape, dtypes, missing values, exact duplicates (quality report saved before cleaning).
2. Drop exact duplicate rows; drop rows with invalid identifiers; enforce a unique `(season, round, driver_id)` key.
3. Numeric conversion with failures counted (not silently filled).
4. Chronological `race_index` across all seasons.
5. Grid `0` = pit-lane start → `grid_pitlane = 1`, `grid_clean = number of entrants`.
6. `classified` from `positionText`; target only for classified rows; exclusion reason for every other row.
7. Qualifying positions merged on the driver-race key; coverage checked against the schedule
   (all 228 scheduled study races have results; qualifying coverage 99.7%).

## 4. Chronological split (fixed before modelling)
| Split | Seasons | Races | Rows | Podium rate |
|---|---|---|---|---|
| Train | 2014–2020 | 138 | 2,324 | 17.8% |
| Validation | 2021–2022 | 44 | 764 | 17.3% |
| Test | 2023–2024 | 46 | 822 | 16.8% |

Splitting by season keeps every race inside one split. Validation spans the 2022 regulation change on purpose:
selecting models that survive a rule change is more honest than tuning on a stable era. The test seasons were
never used for any decision; they are evaluated once after selection.

## 5. Features (`src/feature_engineering.py`)
All historical features for race *r* use only races with `race_index < r`: each entity's sequence is sorted
chronologically and `shift(1)` is applied before every rolling/expanding aggregation. Constructor form is
computed per team **lineage** (e.g. Toro Rosso → AlphaTauri → RB) so a rebrand does not erase history, while the
official constructor ID remains the categorical feature.

| Group | Features |
|---|---|
| A — Basic | grid slot, pit-lane flag, season, round, circuit ID, driver ID, constructor ID |
| B — Recent form | driver mean race-order position (prev 3/5), podiums (prev 5), standardised points/race (prev 5), DNF rate (prev 5), mean grid (prev 5); team mean position (prev 3/5), team podiums and points/race (prev 5); history flags |
| C — Experience | driver career starts, team prior races, driver starts with current team, races since previous start, rookie / returning / team-change flags |
| D — Career context | driver career mean position and podium rate, driver & team season points before the race and their in-race ranks, grid − qualifying (penalties), qualifying-missing flag |

Design rules:
- **Transfers:** personal form follows the driver; `drv_races_with_con` resets and `con_changed = 1`; team form is
  that of the *new* team.
- **Returning drivers:** history kept; `drv_gap_races` and `is_returning` (gap ≥ 10 races) describe the gap.
- **Rookies / new teams:** form features stay `NaN` (never 0, which would look like winning form). The model
  pipeline imputes the **training** median and the explicit `drv_has_history` / `con_has_history` flags let models
  treat these rows separately.
- **Points:** recomputed with one 25-18-15-12-10-8-6-4-2-1 scale for all seasons, so the 2014 double-points finale
  and the 2019–2024 fastest-lap point do not distort form.
- **Race order position** (Ergast `position`) is used for form; it places non-classified cars behind classified
  ones by laps completed, so retirements count against recent form.

Excluded from features (outcome or post-race information): finishing position, status, laps, points,
fastest lap, final standings, and anything from the target race. `tests/test_no_leakage.py` scrambles the target
race and all future races and asserts the target race's features are identical.

## 6. Preprocessing
`Pipeline(ColumnTransformer, classifier)` — numeric: median imputation (+ StandardScaler for KNN and SVM);
categorical: one-hot with `handle_unknown="ignore"` (unseen drivers/teams map to all-zero). The transformer is
fitted inside the pipeline, i.e. on training rows only (verified in tests).

## 7. Models and tuning
| Model | Imbalance handling | Search |
|---|---|---|
| Decision Tree (`criterion` entropy/gini) | `class_weight="balanced"` | grid, 240 candidates |
| KNN | none (distance weighting optional) | grid, 28 candidates |
| SVM (linear and RBF kernels — one algorithm) | `class_weight="balanced"` | grid, 15 candidates |
| Random Forest | `class_weight="balanced"` | randomised, 30 candidates |

Tuning uses `PredefinedSplit`: fit on train, score on validation. **Selection metric: validation average
precision (AP)** — threshold-free and sensitive to the minority class. Every model is tuned separately for each
feature-group combination (A, A+B, A+C, A+B+C, A+B+C+D); the combination with the best validation AP is kept.

## 8. Final evaluation
1. Selected configuration fitted on train → training metrics, validation metrics, train–validation gap,
   complexity curves, learning curves, permutation importance (validation).
2. Same configuration refitted on train + validation (2014–2022) → evaluated once on test (2023–2024).
3. Preferred model = highest **validation** AP (decided before looking at test results).

Metrics: accuracy, precision, recall, F1 (default decision rule), ROC-AUC, AP, Brier score (probability
models), confusion matrix, plus a race-level **top-3 hit rate** (share of each race's three highest-scored
drivers who finished on the podium). Timing: refit wall time; batch latency (median of 7 runs over the test set)
and single-row latency (median of 50 calls).

## 9. Additional experiments
- **Complexity curves:** DT `max_depth`, KNN `n_neighbors`, SVM `C`, RF `max_depth` (others fixed at selected values).
- **Learning curves:** most recent *k* training seasons, k = 1…7.
- **Rolling origin:** for each season Y in 2017–2022, train on all earlier study seasons, evaluate on Y.
- **Target-policy sensitivity:** retrain on all starters (non-classified = 0), compare validation AP.
