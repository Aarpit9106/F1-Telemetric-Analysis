"""Timing measurements, environment capture and loading of saved experiment outputs."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import load_config, resolve


def environment_info() -> dict[str, Any]:
    import joblib
    import sklearn
    info = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            kb = int(fh.readline().split()[1])
            info["memory_gb"] = round(kb / 1024 / 1024, 1)
    except OSError:
        pass
    return info


def time_call(fn: Callable[[], Any], repeats: int = 7) -> dict[str, float]:
    """Run fn several times; return median and min wall time in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(times)), "min_s": float(np.min(times)), "repeats": repeats}


def prediction_latency(model, X: pd.DataFrame, single_repeats: int = 50, batch_repeats: int = 7) -> dict[str, float]:
    """Batch latency over the full set and single-row latency (median)."""
    batch = time_call(lambda: model.predict(X), batch_repeats)
    rows = X.sample(n=min(single_repeats, len(X)), random_state=0)
    singles = []
    for i in range(len(rows)):
        r = rows.iloc[[i]]
        t0 = time.perf_counter()
        model.predict(r)
        singles.append(time.perf_counter() - t0)
    return {
        "batch_rows": int(len(X)),
        "batch_median_s": batch["median_s"],
        "per_row_batch_us": batch["median_s"] / max(len(X), 1) * 1e6,
        "single_row_median_ms": float(np.median(singles)) * 1e3,
    }


# ---------------------------------------------------------------------------
# Loading saved outputs (used by the Streamlit app)
# ---------------------------------------------------------------------------
def results_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return resolve(cfg["reports"]["results_dir"])


def load_json(name: str, cfg: dict | None = None) -> dict | None:
    path = results_dir(cfg) / name
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(name: str, cfg: dict | None = None) -> pd.DataFrame | None:
    path = results_dir(cfg) / name
    return pd.read_csv(path) if path.exists() else None
