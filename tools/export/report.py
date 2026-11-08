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
    """Write strict JSON: NaN and infinities become null, so any JSON reader accepts the file."""
    path = Path(path)
    path.write_text(json.dumps(_clean(report), indent=2, allow_nan=False), encoding="utf-8")
    return path


def _clean(value):
    """Plain JSON types throughout. NumPy float64 is a Python float, so floats are checked directly."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)  # paths and anything else
