"""Vineyard polygons from a QGIS feature source into the core, and geometries back out."""

from __future__ import annotations

import shapely
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from qgis.core import QgsGeometry, QgsWkbTypes

from ..core.export import BLOCK_FIELDS
from ..core.vineyards import Block
from .transform import Transform

BLOCK_OUTPUT_FIELDS = BLOCK_FIELDS


def read_blocks(source, crs, context, feedback) -> tuple[list[Block], list]:
    """(blocks in crs, their source features). Block ids are feature ids.

    Curved geometries are straightened, Z/M dropped, and invalid polygons
    repaired; anything that is not an area after that is skipped with a warning.
    Coordinates are transformed with GDAL's OSR, as the rasters are (transform.py).
    """
    transform = Transform(source.sourceCrs(), crs, "The vineyard polygon layer")
    blocks, features = [], []
    for feature in source.getFeatures():
        if feedback.isCanceled():
            break
        geometry = feature.geometry()
        if geometry is None or geometry.isEmpty():
            feedback.pushWarning(f"Vineyard polygon {feature.id()} has no geometry; skipped.")
            continue
        geometry = QgsGeometry(geometry)
        if QgsWkbTypes.isCurvedType(geometry.wkbType()):
            geometry.convertToStraightSegment()
        polygon = polygonal(transform.geometry(shapely.force_2d(shapely.from_wkb(bytes(geometry.asWkb())))))
        if polygon is None:
            feedback.pushWarning(f"Vineyard polygon {feature.id()} has no area after repair; skipped.")
            continue
        blocks.append(Block(id=int(feature.id()), polygon=polygon, source="layer"))
        features.append(feature)
    return blocks, features


def polygonal(geometry):
    """The polygon parts of a geometry after repair, as a Polygon or MultiPolygon, or None."""
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    parts, stack = [], list(shapely.get_parts(geometry))
    while stack:  # repair can nest multipolygons inside a collection
        part = stack.pop(0)
        if isinstance(part, Polygon):
            if part.area > 0:
                parts.append(part)
        elif isinstance(part, (MultiPolygon, GeometryCollection)):
            stack.extend(shapely.get_parts(part))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def to_qgs_geometry(geometry) -> QgsGeometry:
    out = QgsGeometry()
    out.fromWkb(shapely.to_wkb(geometry))
    return out
