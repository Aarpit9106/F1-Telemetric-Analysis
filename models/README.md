# Model artefacts

Created by `python -m src.train`. Each `<model>.joblib` is a dictionary:

| Key | Meaning |
|---|---|
| `model` | fitted scikit-learn `Pipeline` (ColumnTransformer preprocessing + classifier), refitted on 2014–2022 |
| `name`, `label` | model identifier and display name |
| `feature_set` | selected feature groups (e.g. `A+B+C`) |
| `numeric_features`, `categorical_features` | exact input column order expected by the pipeline |
| `params` | selected hyperparameters |
| `trained_on_seasons`, `data_cutoff_season` | training seasons and data cutoff |
| `score_kind` | `probability` (predict_proba) or `decision_function` (SVM margin) |
| `environment` | Python / scikit-learn / pandas / NumPy versions used for training |

Load with `src.predict.load_artefact(name)`. Artefacts must be loaded with the same scikit-learn version
(pinned in `requirements.txt`). Experiment outputs (metrics, curves, ablation, timing) are in `reports/results/`.
