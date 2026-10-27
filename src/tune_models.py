"""Model definitions, preprocessing pipelines and validation-based hyperparameter tuning.

Tuning uses a *predefined* split: models are fitted on the training seasons and
scored on the validation seasons (PredefinedSplit). The test seasons are never
passed to this module.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import groups_to_columns

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}

MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbours",
    "svm": "Support Vector Machine",
    "random_forest": "Random Forest",
}

# Parameter that best represents model capacity, used for complexity curves.
COMPLEXITY_PARAM = {
    "decision_tree": ("clf__max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]),
    "knn": ("clf__n_neighbors", [1, 3, 5, 9, 15, 25, 41, 61, 91, 151, 251]),
    "svm": ("clf__C", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]),
    "random_forest": ("clf__max_depth", [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, None]),
}


def model_spec(name: str, seed: int) -> dict[str, Any]:
    """Estimator, scaling need, search space and search type for each algorithm."""
    if name == "decision_tree":
        return {
            "estimator": DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=seed),
            "scale": False,
            "search": "grid",
            "grid": {
                "clf__criterion": ["entropy", "gini"],
                "clf__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
                "clf__min_samples_leaf": [1, 5, 10, 20, 50],
                "clf__min_samples_split": [2, 20, 50],
            },
        }
    if name == "knn":
        return {
            "estimator": KNeighborsClassifier(),
            "scale": True,
            "search": "grid",
            "grid": {
                "clf__n_neighbors": [5, 9, 15, 25, 41, 61, 91],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        }
    if name == "svm":
        return {
            "estimator": SVC(class_weight="balanced", random_state=seed, cache_size=500),
            "scale": True,
            "search": "grid",
            "grid": [
                {"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1]},
                {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.003, 0.01, 0.03]},
            ],
        }
    if name == "random_forest":
        return {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            "scale": False,
            "search": "random",
            "n_iter": 30,
            "grid": {
                "clf__n_estimators": [100, 200, 400],
                "clf__max_depth": [4, 6, 8, 10, 14, None],
                "clf__min_samples_leaf": [1, 3, 5, 10, 20],
                "clf__max_features": ["sqrt", 0.3, 0.5],
            },
        }
    raise KeyError(name)


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Median imputation (+ optional scaling) for numeric, one-hot for categorical.

    Missing numeric values only arise for drivers/teams without history; the
    explicit history-indicator features let models separate those rows instead
    of treating the imputed median as real form.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(name: str, feature_set: str, seed: int, params: dict | None = None) -> Pipeline:
    spec = model_spec(name, seed)
    num, cat = groups_to_columns(feature_set)
    pipe = Pipeline([("pre", build_preprocessor(num, cat, spec["scale"])), ("clf", spec["estimator"])])
    if params:
        pipe.set_params(**params)
    return pipe


def train_val_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, PredefinedSplit]:
    """Concatenate train + validation rows with a PredefinedSplit (-1 = train, 0 = validation)."""
    dev = df[df["split"].isin(["train", "validation"])].sort_values("race_index").reset_index(drop=True)
    fold = np.where(dev["split"] == "validation", 0, -1)
    return dev, dev["podium"].to_numpy(), PredefinedSplit(fold)


def tune(name: str, feature_set: str, df: pd.DataFrame, seed: int, n_jobs: int = -1) -> dict[str, Any]:
    """Search hyperparameters for one model and feature set; return best + all results."""
    spec = model_spec(name, seed)
    pipe = build_pipeline(name, feature_set, seed)
    dev, y, cv = train_val_frame(df)
    common = dict(scoring=SCORING, refit=False, cv=cv, return_train_score=True, n_jobs=n_jobs, error_score="raise")
    if spec["search"] == "grid":
        search = GridSearchCV(pipe, spec["grid"], **common)
    else:
        search = RandomizedSearchCV(pipe, spec["grid"], n_iter=spec["n_iter"], random_state=seed, **common)
    t0 = time.perf_counter()
    search.fit(dev, y)
    elapsed = time.perf_counter() - t0
    res = pd.DataFrame(search.cv_results_)
    best_i = int(res["rank_test_average_precision"].idxmin())
    row = res.loc[best_i]
    best = {
        "model": name,
        "feature_set": feature_set,
        "best_params": {k.replace("clf__", ""): (None if (isinstance(v, float) and np.isnan(v)) else v)
                        for k, v in row["params"].items()},
        "best_params_raw": row["params"],
        "n_candidates": int(len(res)),
        "search_seconds": float(elapsed),
        "mean_fit_time_s": float(row["mean_fit_time"]),
    }
    for m in SCORING:
        best[f"train_{m}"] = float(row[f"split0_train_{m}"])
        best[f"val_{m}"] = float(row[f"split0_test_{m}"])
    keep = ["params", "mean_fit_time", "mean_score_time"] + \
           [c for c in res.columns if c.startswith("split0_test_") or c.startswith("split0_train_")]
    all_res = res[keep].copy()
    all_res.insert(0, "feature_set", feature_set)
    all_res.insert(0, "model", name)
    all_res["params"] = all_res["params"].map(lambda p: {k.replace("clf__", ""): v for k, v in p.items()})
    return {"best": best, "all": all_res}
