"""Run configuration (PROJECT.md §12).

A run's configuration is configs/default.toml with zero or more override files
merged on top, in order. An override file only needs the keys it changes:

    from tools.config import RunSettings, load_config

    config = load_config("configs/bordeaux.toml")
    settings = RunSettings.from_config(config)   # refuses unset OPEN keys
    settings.spacing.values()                     # (1.2, 1.22, ..., 1.6)

The translation increment is not a configuration value. It is derived from the
analysis raster's pixel size (PROJECT.md §7).
"""

from .loader import CONFIG_DIR, DEFAULT_CONFIG, PROJECT_ROOT, load_config, merge, read_toml, resolve_path
from .settings import ConfigError, RunSettings
from .spacing import SpacingRange

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_CONFIG",
    "PROJECT_ROOT",
    "ConfigError",
    "RunSettings",
    "SpacingRange",
    "load_config",
    "merge",
    "read_toml",
    "resolve_path",
]
