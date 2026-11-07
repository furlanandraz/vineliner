"""Machine-readable run report (PROJECT.md §13)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def open_choices(default_config: dict, run_config: dict) -> dict:
    """Keys set by this run that default.toml leaves unset: the OPEN choices this run made."""
    chosen = {}
    for section, table in run_config.items():
        if not isinstance(table, dict):
            continue
        base = default_config.get(section, {})
        for key, value in table.items():
            if key not in base:
                chosen[f"{section}.{key}"] = value
    return chosen


def write_report(path: str | Path, report: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2, default=_json_default))
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
