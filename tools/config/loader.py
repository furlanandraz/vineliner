"""Load TOML run configurations and merge overrides.

A config file may start with a top-level key

    extends = "other.toml"

(relative to that file). The other file is read first and this file's keys are
merged on top, so variants only state what they change. Chains are allowed;
cycles are refused.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"


def load_config(*overrides: str | Path, base: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the base config, then merge each override file (and what it extends) on top, in order."""
    config = read_toml(base)
    for path in overrides:
        config = merge(config, read_config_file(path))
    return config


def read_config_file(path: str | Path, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """One config file with its `extends` chain resolved."""
    path = Path(path).resolve()
    if path in _chain:
        cycle = " -> ".join(p.name for p in (*_chain, path))
        raise ValueError(f"config 'extends' cycle: {cycle}")
    data = read_toml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ValueError(f"{path.name}: 'extends' must be a file name, got {parent!r}")
    parent_path = Path(parent) if Path(parent).is_absolute() else path.parent / parent
    return merge(read_config_file(parent_path, (*_chain, path)), data)


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
