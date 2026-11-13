"""The Vineliner Processing provider: one algorithm, Detect vine rows."""

from __future__ import annotations

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .detect_rows import DetectRowsAlgorithm

ICON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.png")


class VinelinerProvider(QgsProcessingProvider):
    def loadAlgorithms(self):  # noqa: N802 (QGIS naming)
        self.addAlgorithm(DetectRowsAlgorithm())

    def id(self) -> str:
        return "vineliner"

    def name(self) -> str:
        return self.tr("Vineliner")

    def longName(self) -> str:  # noqa: N802 (QGIS naming)
        return self.tr("Vineliner: vineyard rows from imagery")

    def icon(self) -> QIcon:
        return QIcon(ICON)
