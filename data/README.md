# Data

## Source
- **Jolpica-F1 API** (Ergast-compatible): `https://api.jolpi.ca/ergast/f1` — project: https://github.com/jolpica/jolpica-f1
- Endpoints used, per season:
  - `/{season}/results.json` — race classification, grid, status, driver, constructor, circuit
  - `/{season}/qualifying.json` — qualifying positions (study seasons only)
  - `/{season}.json` — race schedule (used for coverage checks)
- FastF1 is **not** used in the core pipeline: telemetry cannot be joined reliably to every historical race and is
  not needed for pre-race features. It is listed as future work.

## Seasons
| Purpose | Seasons | Used as model rows? |
|---|---|---|
| History warm-up (driver/team form, career counts) | 2000–2013 | No |
| Study period | 2014–2024 | Yes (classified entries only) |

Boundaries are configurable in `config/settings.yaml` (`history_start_season`, `study_start_season`,
`study_end_season`).

## Reproduce
```bash
python -m src.data_pipeline            # downloads anything not cached, then cleans and builds features
python -m src.data_pipeline --offline  # cache only — fails with a clear message if a file is missing
python -m src.data_pipeline --refresh  # re-download everything
```
Requests are paged (100 rows/page) and paused 0.4 s apart to respect the API's rate limits. A full download
is ~200 requests. Retrieval time, URL and row counts for each file are recorded in `data/raw/manifest.json`.

## Layout
```
data/raw/results/<season>.json      raw API responses (never modified), with retrieval metadata
data/raw/qualifying/<season>.json
data/raw/schedule/<season>.json
data/raw/manifest.json              coverage + retrieval log
data/processed/results_flat.csv     flattened raw results (all seasons)
data/processed/results_clean.csv    typed, de-duplicated, with target + inclusion flags (all entries)
data/processed/features_study.csv   all started entries 2014-2024 with pre-race features
data/processed/model_dataset.csv    supervised rows (classified entries) with split labels
data/processed/cleaning_summary.json, pipeline_summary.json, season_coverage.csv
```

## Inclusion policy
Only entries with a numeric classified position (`positionText` is a number) receive a target. Retired (R),
disqualified (D), excluded (E), withdrawn (W), failed-to-qualify (F) and not-classified (N) entries are excluded
from the supervised dataset and counted in `cleaning_summary.json`. They still contribute to *historical* features
(e.g. a retirement lowers a driver's recent form and raises their recent DNF rate).

## GitHub vs. submission ZIP
`data/raw` and `data/processed` are git-ignored (reproducible). The submission ZIP includes them so the app runs
offline immediately after extraction.
