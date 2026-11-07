"""Load TOML run configurations and merge per-area overrides."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"


def load_config(*overrides: str | Path, base: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the base config, then merge each override file on top, in order."""
    config = read_toml(base)
    for path in overrides:
        config = merge(config, read_toml(path))
    return config


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


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Relative paths in a config are relative to the project root, not the working directory."""
    p = Path(path)
    return p if p.is_absolute() else root / p
