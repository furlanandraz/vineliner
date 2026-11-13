"""Run configuration (PROJECT.md §12).

A run's configuration is configs/default.toml with zero or more override files
merged on top, in order. An override file only needs the keys it changes, and
may start with `extends = "other.toml"` to build on another file:

    from tools.config import RunSettings, load_config

    config = load_config("configs/centroid-pretest.toml")
    settings = RunSettings.from_config(config)   # refuses unset OPEN keys
    settings.spacing.values()                     # (1.2, 1.22, ..., 1.6)

The translation increment is not a configuration value. It is derived from the
analysis raster's pixel size (PROJECT.md §7).
"""

from .loader import CONFIG_DIR, DEFAULT_CONFIG, PROJECT_ROOT, load_config, merge, read_config_file, read_toml, resolve_path
from .settings import INPUT_LAYERS, ConfigError, RunSettings
from .spacing import SpacingRange

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_CONFIG",
    "INPUT_LAYERS",
    "PROJECT_ROOT",
    "ConfigError",
    "RunSettings",
    "SpacingRange",
    "load_config",
    "merge",
    "read_config_file",
    "read_toml",
    "resolve_path",
]
