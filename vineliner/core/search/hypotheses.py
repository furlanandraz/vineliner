"""Hypothesis grids for the exhaustive local search (PROJECT.md §7).

Translations are whole-pixel offsets on each world axis:
    offset_x = i * px,  offset_y = i * py,  -s/2 <= i * p < s/2,
where s is the box side (the tested spacing d, or the nominal d0). Offsets
here are world-axis integers: positive i_y is northward. Converting to array
rows (which run southward) happens once, through Grid or in local.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely

from ..geo import Grid

OFFSET_TOLERANCE = 1e-9  # in pixel units


def translation_offsets(box_m: float, pixel_m: float, tol: float = OFFSET_TOLERANCE) -> np.ndarray:
    """Integers i with -box/2 <= i * pixel < box/2, i.e. ceil(-h) .. ceil(h) - 1 for h = box / (2 pixel)."""
    h = box_m / (2.0 * pixel_m)
    lo = math.ceil(-h - tol)
    hi = math.ceil(h - tol) - 1
    return np.arange(lo, hi + 1, dtype=np.int64)


def centre_outward_order(ox: np.ndarray, oy: np.ndarray, px: float, py: float) -> np.ndarray:
    """Natural flat indices of the (len(oy), len(ox)) grid, sorted centre outward.

    Sort key: distance from the box centre, then counterclockwise angle from +x
    in [0, 2*pi). Deterministic. Ordering never permits skipping hypotheses.
    """
    gy, gx = np.meshgrid(oy * py, ox * px, indexing="ij")
    dist2 = np.round((gx**2 + gy**2).ravel(), 9)
    angle = np.mod(np.arctan2(gy, gx).ravel(), 2 * np.pi)
    return np.lexsort((angle, dist2))


@dataclass
class SearchSpace:
    thetas: tuple[float, ...]
    spacings: tuple[float, ...]
    px: float
    py: float
    box: str  # "tested_d" or "nominal_d0"
    nominal_m: float
    _cache: dict = field(default_factory=dict, repr=False)

    def box_side(self, j: int) -> float:
        return self.spacings[j] if self.box == "tested_d" else self.nominal_m

    def offsets(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """(ox, oy): world-axis integer offsets along easting and northing for spacing j."""
        s = self.box_side(j)
        return translation_offsets(s, self.px), translation_offsets(s, self.py)

    def order(self, j: int) -> np.ndarray:
        if j not in self._cache:
            ox, oy = self.offsets(j)
            self._cache[j] = centre_outward_order(ox, oy, self.px, self.py)
        return self._cache[j]

    def n_translations(self, j: int) -> int:
        ox, oy = self.offsets(j)
        return len(ox) * len(oy)

    def hypotheses_per_centre(self) -> int:
        return sum(len(self.thetas) * self.n_translations(j) for j in range(len(self.spacings)))

    def max_translation_px(self) -> int:
        r = 0
        for j in range(len(self.spacings)):
            ox, oy = self.offsets(j)
            r = max(r, int(np.abs(ox).max()), int(np.abs(oy).max()))
        return r

    def describe(self) -> dict:
        return {
            "rotations": len(self.thetas),
            "rotation_range_deg": [self.thetas[0], self.thetas[-1]],
            "spacings": len(self.spacings),
            "spacing_range_m": [self.spacings[0], self.spacings[-1]],
            "translation_box": self.box,
            "translation_increment_m": [self.px, self.py],
            "translations_per_spacing": [self.n_translations(j) for j in range(len(self.spacings))],
            "hypotheses_per_centre": self.hypotheses_per_centre(),
            "offset_convention": "ox east-positive, oy north-positive, in whole pixels",
        }


def search_centres(polygon, grid: Grid, valid: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Search-box centres over a block, on pixel centres (PROJECT.md §7).

    A square lattice with the given spacing, anchored at whole multiples of the
    spacing in world coordinates, is snapped to the nearest pixel centre. Only
    pixels whose centre is inside the polygon and valid are kept.
    Returns (rows, cols) in grid coordinates, sorted by row then column.
    """
    xmin, ymin, xmax, ymax = polygon.bounds
    xs = np.arange(math.ceil(xmin / spacing_m) * spacing_m, xmax + 1e-9, spacing_m)
    ys = np.arange(math.ceil(ymin / spacing_m) * spacing_m, ymax + 1e-9, spacing_m)
    gx, gy = np.meshgrid(xs, ys)
    rows, cols = grid.nearest_pixel(gx.ravel(), gy.ravel())
    inside_raster = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    rows, cols = rows[inside_raster], cols[inside_raster]
    cx, cy = grid.array_to_world(rows, cols)
    keep = shapely.intersects_xy(polygon, cx, cy) & valid[rows, cols]
    pairs = np.unique(np.stack([rows[keep], cols[keep]], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]
