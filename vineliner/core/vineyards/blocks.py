"""Vineyard block polygons (PROJECT.md §4).

A block is one planting block with its own row orientation. Its polygon
limits image support during scoring and growth, and clips the final rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from ..geo import Grid


@dataclass(frozen=True)
class Block:
    id: int
    polygon: BaseGeometry  # Polygon or MultiPolygon, in the raster CRS
    source: str  # "file" (supplied/hand-drawn fixture) or "classified"
    attributes: dict = field(default_factory=dict)  # the input feature's attributes, carried to the output

    @property
    def area_m2(self) -> float:
        return float(self.polygon.area)


def make_blocks(ids, polygons, source: str, records=None) -> list[Block]:
    blocks = []
    records = records if records is not None else [{} for _ in ids]
    for block_id, polygon, attrs in zip(ids, polygons, records):
        if not polygon.is_valid:
            polygon = shapely.make_valid(polygon)
        blocks.append(Block(id=int(block_id), polygon=polygon, source=source, attributes=dict(attrs)))
    return blocks


def block_mask(block: Block, grid: Grid) -> np.ndarray:
    """Pixels whose centre lies inside the block polygon (boundary included)."""
    rows, cols = np.indices(grid.shape)
    x, y = grid.array_to_world(rows, cols)
    shapely.prepare(block.polygon)
    return shapely.intersects_xy(block.polygon, x, y)


def pixel_window(block: Block, grid: Grid, pad_px: int) -> tuple[int, int, int, int]:
    """(row0, col0, height, width) of the block's bounding box plus pad_px on every side.

    The window may extend beyond the raster; callers pad the arrays to match.
    """
    xmin, ymin, xmax, ymax = block.polygon.bounds
    r_top, c_left = grid.world_to_array(xmin, ymax)
    r_bottom, c_right = grid.world_to_array(xmax, ymin)
    row0 = int(np.floor(r_top)) - pad_px
    col0 = int(np.floor(c_left)) - pad_px
    row1 = int(np.ceil(r_bottom)) + pad_px + 1
    col1 = int(np.ceil(c_right)) + pad_px + 1
    return row0, col0, row1 - row0, col1 - col0
