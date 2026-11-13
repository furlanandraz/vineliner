"""Coordinate transforms with GDAL's OSR instead of QgsCoordinateTransform.

QGIS 3.44 can get into a state where its cached transform for a pair of CRSs
hands the coordinates back unchanged, in every thread, without an error, until
QGIS restarts (reproduced: map rendering, then a Processing worker thread
replacing an expired one; afterwards EPSG:4326 -> EPSG:32633 returned degrees
as they were). That put a vineyard 0.001 m across on a 1 x 1 px grid.

GDAL's OSR keeps its own PROJ state and stays right in that state, and GDAL
warps the rasters anyway, so the polygons, the analysis grid and the rasters
all go through the same engine and the same PROJ coordinate operation.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from osgeo import osr
from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsProcessingException, QgsRectangle

EDGE_POINTS = 21  # points per edge when a rectangle is transformed (as QgsCoordinateTransform.transformBoundingBox)


def gdal_srs(crs: QgsCoordinateReferenceSystem) -> str:
    """The CRS as GDAL takes it: the EPSG code when it has one, else WKT."""
    return crs.authid() if crs.authid().upper().startswith("EPSG:") else crs.toWkt(Qgis.CrsWktVariant.Preferred)


def _osr(crs: QgsCoordinateReferenceSystem) -> osr.SpatialReference:
    srs = osr.SpatialReference()
    srs.SetFromUserInput(gdal_srs(crs))
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)  # x east, y north, as QGIS layers have them
    return srs


def _label(crs: QgsCoordinateReferenceSystem) -> str:
    return crs.authid() or crs.description()


class Transform:
    """source -> destination for points, rectangles and Shapely geometries; raises QgsProcessingException on failure."""

    def __init__(self, source: QgsCoordinateReferenceSystem, destination: QgsCoordinateReferenceSystem, what: str):
        self.source, self.destination, self.what = source, destination, what
        self.identity = source == destination
        self._ct = None
        if not self.identity:
            try:
                with osr.ExceptionMgr(useExceptions=True):
                    self._ct = osr.CoordinateTransformation(_osr(source), _osr(destination))
            except RuntimeError as exc:
                raise QgsProcessingException(
                    f"{what} cannot be transformed from {_label(source)} to {_label(destination)}: {exc}") from exc

    def points(self, xy) -> np.ndarray:
        """(n, 2) x, y in the source CRS -> (n, 2) in the destination CRS."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if self.identity or len(xy) == 0:
            return xy.copy()
        try:
            with osr.ExceptionMgr(useExceptions=True):
                out = np.asarray(self._ct.TransformPoints(xy.tolist()), dtype=float)[:, :2]
        except RuntimeError as exc:
            raise QgsProcessingException(
                f"{self.what} cannot be transformed from {_label(self.source)} to {_label(self.destination)}: {exc}") from exc
        if not np.isfinite(out).all():
            raise QgsProcessingException(
                f"{self.what} has coordinates that cannot be transformed from {_label(self.source)} to "
                f"{_label(self.destination)}; the layer's CRS probably does not match its coordinates.")
        require_transformed(self.source, self.destination, xy[0], out[0])
        return out

    def point(self, x: float, y: float) -> tuple[float, float]:
        (px, py), = self.points([(x, y)])
        return float(px), float(py)

    def rectangle(self, rect: QgsRectangle) -> QgsRectangle:
        """The bounding box of the rectangle's transformed outline (each edge sampled at EDGE_POINTS)."""
        if self.identity:
            return QgsRectangle(rect)
        t = np.linspace(0.0, 1.0, EDGE_POINTS)
        x0, x1, y0, y1 = rect.xMinimum(), rect.xMaximum(), rect.yMinimum(), rect.yMaximum()
        xs = np.concatenate([x0 + (x1 - x0) * t, np.full_like(t, x1), x1 - (x1 - x0) * t, np.full_like(t, x0)])
        ys = np.concatenate([np.full_like(t, y0), y0 + (y1 - y0) * t, np.full_like(t, y1), y1 - (y1 - y0) * t])
        out = self.points(np.column_stack([xs, ys]))
        return QgsRectangle(out[:, 0].min(), out[:, 1].min(), out[:, 0].max(), out[:, 1].max())

    def geometry(self, geometry):
        """A Shapely geometry, vertex by vertex."""
        return geometry if self.identity else shapely.transform(geometry, self.points)


def require_transformed(source, destination, before, after):
    """Raise when a transform between degrees and metres handed the coordinates back unchanged.

    That is what QGIS's broken transform cache does (see the module docstring);
    it can never be right, since degrees and metres differ by orders of magnitude.
    """
    if source.isGeographic() != destination.isGeographic() and math.isclose(before[0], after[0]) and math.isclose(before[1], after[1]):
        raise QgsProcessingException(
            f"The coordinate transform from {_label(source)} to {_label(destination)} returned the coordinates unchanged "
            f"({before[0]:.6g}, {before[1]:.6g}). This is a problem in the PROJ setup, not in the layers; restarting QGIS "
            "usually clears it.")
