"""Machine-readable run report (PROJECT.md §13), written by the plugin on request."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_report(path: str | Path, report: dict) -> Path:
    """Write strict JSON: NaN and infinities become null, so any JSON reader accepts the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(report), indent=2, allow_nan=False), encoding="utf-8")
    return path


def clean_json(value):
    """Plain JSON types throughout. NumPy float64 is a Python float, so floats are checked directly."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)  # paths and anything else
