"""Vineliner: a QGIS plugin that finds vineyard rows in aerial imagery and draws them as lines.

QGIS calls classFactory() when it loads the plugin. The row-finding method
lives in core/ (NumPy, SciPy and Shapely only, no QGIS imports); qgis_io/ and
processing_provider/ connect it to QGIS layers and the Processing framework.
"""


def classFactory(iface):  # noqa: N802 (name required by QGIS)
    from .plugin import VinelinerPlugin

    return VinelinerPlugin(iface)
