# Rubric-to-project mapping

| Criterion (marks) | Requirement | Where it is in the project | How to demonstrate |
|---|---|---|---|
| **Data cleaning and splitting (3)** | Quality checks | `src/data_cleaning.py::quality_report`, `data/processed/cleaning_summary.json` | Data Explorer → *Data quality* tab |
| | Preprocessing | `src/data_cleaning.py` (types, duplicates, grid 0, classification), `src/tune_models.py::build_preprocessor` (impute/scale/one-hot inside the pipeline) | Report §4 |
| | Leakage prevention | `src/feature_engineering.py` (shift-before-roll), `tests/test_no_leakage.py` (6 tests incl. scrambling current/future races) | `pytest tests/test_no_leakage.py -v` |
| | Chronological split | `config/settings.yaml` → `split`; `src/feature_engineering.py::assign_split`; `pipeline_summary.json` → `races_spanning_multiple_splits: 0` | Overview page; report §6 |
| **Model comparison & analytics (4)** | ≥ 4 distinct algorithms | Decision Tree, KNN, SVM (linear + RBF configs), Random Forest — `src/tune_models.py::model_spec` | Model Comparison page |
| | Hyperparameter tuning | `src/tune_models.py::tune` (PredefinedSplit on validation), `reports/results/tuning_all.csv` | Telemetry → *Feature sets* tab |
| | Common metrics | `src/evaluate.py` (accuracy, precision, recall, F1, ROC-AUC, AP, Brier, confusion matrix, top-3 hit rate), `reports/results/comparison.csv` | Model Comparison table |
| | Over/underfitting + ablation | `src/train.py` (complexity curves, learning curves, rolling origin, ablation), `reports/results/ablation.csv` | Telemetry → *Fit diagnostics*; Comparison → *Feature ablation* |
| | Charts & interpretation | `src/visualizations.py`, `reports/figures/model_*.png`, report §7–§9 | Report |
| **UI deployment & cross-platform (4)** | Interactive UI | `app.py` + `app_pages/` (6 pages, Streamlit) | `streamlit run app.py` |
| | Desktop + Android browsers | Responsive CSS (`app_pages/common.py`), compact Plotly charts, touch-sized controls; screenshots `reports/figures/screenshots/desktop_*.png`, `mobile_*.png` | Open `http://<laptop-ip>:8501` on a phone on the same Wi-Fi |
| | Deployment | `docs/deployment.md` (local network + Streamlit Community Cloud steps) | — |
| **Code, GitHub, report (4)** | Modular code | `src/` (collection, cleaning, features, pipeline, EDA, tuning, training, evaluation, telemetry, prediction, visualisation) | — |
| | README, requirements | `README.md`, `requirements.txt` (pinned) | — |
| | Testing | `tests/` (38 tests) | `pytest` |
| | GitHub-ready | `.gitignore` (data/caches excluded), `LICENSE` | `git init && git add . && git commit` |
| | 10-page report | `reports/report.pdf` / `report.docx` / `report.md` | — |
| | Literature (≥ 5 verified) | `reports/literature_review.md`, `reports/literature_tracker.csv` (11 papers, DOIs/URLs checked) | — |
| **Viva** | Preparation | `docs/viva_questions.md` | — |
