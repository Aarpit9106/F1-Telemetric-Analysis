# F1 Race Intelligence
**Historical Data Analysis and Podium Prediction Using Machine Learning** — a Machine Learning Lab mini-project.

Given only information available *before* a Formula 1 Grand Prix starts, predict whether a driver finishes on the
podium (P1–P3). The project covers a reproducible data pipeline (Jolpica-F1 API), leakage-safe feature engineering,
a chronological train/validation/test split, four tuned algorithm families (Decision Tree, KNN, SVM, Random Forest),
overfitting and feature-ablation experiments, a 6-page Streamlit app that works on desktop and mobile browsers,
38 automated tests, and a ~10-page report.

> Educational project. Predictions are estimates from historical data, not race forecasts or betting advice.

## Quick start (submission ZIP — data, models and results included)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt          # Python 3.11+ (tested on 3.13)
streamlit run app.py                     # opens http://localhost:8501
```
On an Android phone (same Wi-Fi): `streamlit run app.py --server.address 0.0.0.0`, then open
`http://<laptop-IP>:8501` in Chrome. See `docs/deployment.md`.

## Reproduce everything from scratch
```bash
python -m src.run_all              # data -> features -> EDA -> tuning/training/evaluation -> figures (~25-35 min)
python -m pytest                   # 38 tests
python scripts/build_report.py     # regenerates reports/report.md (+ .docx/.pdf if pandoc/LibreOffice exist)
```
Individual steps:

| Step | Command | Output |
|---|---|---|
| Download + cache data | `python -m src.data_collection [--refresh/--offline]` | `data/raw/`, `manifest.json` |
| Clean, features, split | `python -m src.data_pipeline [--offline]` | `data/processed/*.csv`, `pipeline_summary.json` |
| EDA figures | `python -m src.eda` | `reports/figures/eda_*.png`, `eda_summary.json` |
| Tune, ablate, train, evaluate | `python -m src.train` | `models/*.joblib`, `reports/results/*` |
| Grid reference + bootstrap | `python -m src.evaluate` (also run by `train`) | `baseline_and_uncertainty.json` |
| Report figures | `python -m src.visualizations` | `reports/figures/model_*.png` |
| Screenshots (optional) | `python scripts/capture_screenshots.py` | `reports/figures/screenshots/` |

The data download makes ~200 paged API requests with pauses to respect Jolpica's rate limits; occasional
"Rate limited… waiting" messages are expected. If the API is unreachable, the pipeline stops with a clear message;
`--offline` uses the cache. Nothing is ever fabricated.

## Results (from `reports/results/`, test = 2023–2024, refitted on 2014–2022)
| Model | Features | Val AP | Test AP | Test F1 | Test ROC-AUC | Test top-3 hit |
|---|---|---|---|---|---|---|
| Decision Tree | A+B | 0.754 | 0.647 | 0.635 | 0.916 | 0.645 |
| KNN | A+B | 0.735 | 0.582 | 0.571 | 0.882 | 0.565 |
| SVM (RBF) | A | 0.799 | 0.672 | 0.583 | 0.899 | 0.551 |
| **Random Forest** (selected) | A+B+C | **0.805** | **0.690** | 0.657 | 0.929 | 0.681 |
| *Grid-slot reference (non-ML)* | grid | 0.715 | 0.693 | 0.669 | 0.910 | 0.681 |

- **Selection criterion (fixed before testing):** highest validation average precision → Random Forest.
- Random Forest and SVM beat the grid ranking on validation, but on 2023–24 they are statistically
  indistinguishable from it (race-level bootstrap); Decision Tree and KNN are worse than the grid.
- Grid slot and the team's recent points per race carry most of the signal. More feature groups helped the Random
  Forest but hurt the SVM.

AP = average precision (podium class, base rate ≈ 17%). Top-3 hit = share of each race's three highest-scored drivers
who actually finished on the podium. Timing figures are in `comparison.csv` and depend on the machine.

## Project structure
```
app.py                      Streamlit entry point (st.navigation)
app_pages/                  overview, data_explorer, race_explorer, model_telemetry, model_comparison, podium_predictor
config/settings.yaml        seasons, split, features, lineage map, seeds
src/
  data_collection.py        API retrieval, caching, schema validation, manifest
  data_cleaning.py          quality checks, types, duplicates, target, inclusion policy
  feature_engineering.py    leakage-safe features (groups A-D), chronological split
  data_pipeline.py          orchestration -> model_dataset.csv / features_study.csv
  eda.py                    EDA tables + report figures
  tune_models.py            model specs, preprocessing pipelines, validation tuning
  train.py                  ablation, final fits, curves, importance, sensitivity, artefacts
  evaluate.py               metrics, curves, grid reference, race bootstrap
  telemetry.py              timing, environment capture, result loaders
  predict.py                artefact loading and race-level prediction
  visualizations.py         Plotly (app) + Matplotlib (report) figures
  run_all.py                one-command reproduction
tests/                      38 pytest tests (cleaning, features, leakage, training, prediction, network failure)
data/                       raw cache + processed tables (git-ignored; see data/README.md)
models/                     trained pipelines (.joblib) + README
reports/                    report.md/.docx/.pdf, literature_review.md, literature_tracker.csv, figures/, results/
notebooks/                  executed EDA notebook
docs/                       methodology, viva questions, rubric mapping, deployment, troubleshooting
scripts/                    report builder, notebook builder, screenshot capture
```

## Documentation
- `docs/methodology.md` — inclusion policy, features, leakage rules, split, tuning, evaluation protocol
- `docs/rubric_mapping.md` — where each rubric criterion is satisfied
- `docs/viva_questions.md` — 32 likely questions with concise answers
- `docs/deployment.md` — local, LAN/Android, Streamlit Community Cloud (no live URL is claimed)
- `docs/troubleshooting.md` — common problems and known limitations
- `reports/literature_review.md` — 11 verified references with a comparison table

## GitHub
```bash
git init && git add . && git commit -m "F1 Race Intelligence"
```
`.gitignore` excludes `data/raw` and `data/processed` (reproducible) and caches. Trained models and result files are
small and committed so the app works after cloning; regenerate the data with `python -m src.data_pipeline`.

## License
MIT (code). Race data © its providers, retrieved from the Jolpica-F1 API at run time; this project is not affiliated
with Formula 1.
