"""Run configuration (PROJECT.md §12).

A run's configuration is core/defaults.toml with the plugin dialog's row width
range merged on top:

    from vineliner.core.config import RunSettings, plugin_config

    settings = RunSettings.from_config(plugin_config(1.2, 1.6))
    settings.spacing.values()   # (1.2, 1.22, ..., 1.6)

The translation increment is not a configuration value. It is derived from the
analysis raster's pixel size (PROJECT.md §7).
"""

from .loader import DEFAULTS_FILE, WIDTH_OPTIONS, defaults, merge, plugin_config, read_toml
from .settings import INPUT_LAYERS, ConfigError, RunSettings
from .spacing import SpacingRange

__all__ = [
    "DEFAULTS_FILE",
    "INPUT_LAYERS",
    "WIDTH_OPTIONS",
    "ConfigError",
    "RunSettings",
    "SpacingRange",
    "defaults",
    "merge",
    "plugin_config",
    "read_toml",
]
