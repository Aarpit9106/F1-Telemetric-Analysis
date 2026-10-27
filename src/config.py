"""Configuration loading and project paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path_like: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p


def season_range(bounds: list[int]) -> list[int]:
    """Expand an inclusive [start, end] pair into a list of seasons."""
    start, end = int(bounds[0]), int(bounds[1])
    return list(range(start, end + 1))
