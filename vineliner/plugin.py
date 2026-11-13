"""The plugin object QGIS loads: registers the Processing provider and a Raster menu entry."""

from __future__ import annotations

import importlib
import os
import sys

from qgis.core import Qgis, QgsApplication, QgsMessageLog
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

PLUGIN_DIR = os.path.dirname(__file__)
ICON = os.path.join(PLUGIN_DIR, "icon.png")
MENU = "&Vineliner"
ALGORITHM_ID = "vineliner:detect_rows"
LOG_TAG = "Vineliner"


def missing_dependencies() -> list[str]:
    """Python packages the core needs that this QGIS does not provide (QGIS 3.40+ on Windows ships them all)."""
    missing = []
    if sys.version_info < (3, 11):
        missing.append(f"Python 3.11 or later (this QGIS runs {sys.version.split()[0]})")
    for name in ("numpy", "scipy", "shapely"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        if name == "shapely" and int(module.__version__.split(".")[0]) < 2:
            missing.append(f"shapely 2 or later (found {module.__version__})")
    return missing


class VinelinerPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.action = None

    def initProcessing(self):  # noqa: N802 (QGIS naming)
        from .processing_provider.provider import VinelinerProvider

        self.provider = VinelinerProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):  # noqa: N802 (QGIS naming)
        missing = missing_dependencies()
        if missing:
            message = "Vineliner needs " + ", ".join(missing) + " in QGIS's Python environment."
            QgsMessageLog.logMessage(message, LOG_TAG, Qgis.MessageLevel.Critical)
            self.iface.messageBar().pushCritical("Vineliner", message)
            return
        self.initProcessing()
        self.action = QAction(QIcon(ICON), "Detect vine rows…", self.iface.mainWindow())
        self.action.setObjectName("vinelinerDetectRows")
        self.action.triggered.connect(self.open_dialog)
        self.iface.addPluginToRasterMenu(MENU, self.action)
        self.iface.addToolBarIcon(self.action)  # the Plugins toolbar: shown by default, unlike the Raster toolbar

    def open_dialog(self):
        import processing

        processing.execAlgorithmDialog(ALGORITHM_ID)

    def unload(self):
        if self.action is not None:
            self.iface.removePluginRasterMenu(MENU, self.action)
            self.iface.removeToolBarIcon(self.action)
            self.action = None
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
