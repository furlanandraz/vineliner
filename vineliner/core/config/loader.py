"""The run configuration: packaged defaults with the plugin dialog's values on top.

core/defaults.toml holds every setting. plugin_config() merges the dialog's
row width range into it: the searched widths (21 of them, both ends included)
and the range the final spacing check accepts.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

DEFAULTS_FILE = Path(__file__).resolve().parents[1] / "defaults.toml"
WIDTH_OPTIONS = 21  # widths searched across the dialog's range, both ends included


def defaults() -> dict[str, Any]:
    """A fresh copy of the packaged defaults."""
    return read_toml(DEFAULTS_FILE)


def plugin_config(row_width_min_m: float, row_width_max_m: float, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults with the row width range (metres) from the dialog, then any overrides merged on top."""
    lo, hi = float(row_width_min_m), float(row_width_max_m)
    if not 0 < lo <= hi:
        raise ValueError(f"the row width range must satisfy 0 < minimum <= maximum, got {lo}-{hi} m")
    step = (hi - lo) / (WIDTH_OPTIONS - 1) if hi > lo else 0.01  # one width when the range is a single value
    config = merge(defaults(), {
        "spacing": {"nominal_m": round((lo + hi) / 2, 6), "min_m": lo, "max_m": hi, "step_m": step},
        "validation": {"spacing_min_m": lo, "spacing_max_m": hi},
    })
    return merge(config, overrides or {})


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return base updated by override. Tables merge key by key; other values replace."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_toml(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)
