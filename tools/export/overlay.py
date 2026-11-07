"""Visual overlay of imagery, blocks, rows and review marks (PROJECT.md §13).

Rows are coloured by confidence: green high, amber medium, red low (review).
Bridged parts are white, dangling ends magenta squares, crossings red squares,
block outlines blue.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely

from ..geo import Grid
from .png import write_png

COLOURS = {
    "block": (0, 140, 255),
    "high": (40, 230, 40),
    "medium": (255, 190, 0),
    "low": (255, 40, 40),
    "bridge": (255, 255, 255),
    "dangling": (255, 0, 255),
    "crossing": (255, 0, 0),
}


def draw_lines(image: np.ndarray, grid: Grid, geometries, colour, step_px: float = 0.5) -> None:
    """Burn line geometries into image by sampling every step_px pixels."""
    px = min(grid.pixel_size)
    for geom in geometries:
        for part in shapely.get_parts(geom):
            if part.is_empty or part.geom_type not in ("LineString", "LinearRing"):
                continue
            n = max(2, int(np.ceil(part.length / (step_px * px))) + 1)
            pts = shapely.get_coordinates(shapely.line_interpolate_point(part, np.linspace(0, part.length, n)))
            _paint(image, grid, pts[:, 0], pts[:, 1], colour, size=0)


def draw_points(image: np.ndarray, grid: Grid, geometries, colour, size: int = 2) -> None:
    pts = np.array([g.representative_point().coords[0] for g in geometries if not g.is_empty])
    if pts.size:
        _paint(image, grid, pts[:, 0], pts[:, 1], colour, size=size)


def _paint(image, grid: Grid, x, y, colour, size: int) -> None:
    rows, cols = grid.nearest_pixel(x, y)
    for dr in range(-size, size + 1):
        for dc in range(-size, size + 1):
            r, c = rows + dr, cols + dc
            ok = (r >= 0) & (r < image.shape[0]) & (c >= 0) & (c < image.shape[1])
            image[r[ok], c[ok]] = colour


def write_overlay(path: str | Path, rgb: np.ndarray, grid: Grid, blocks, sections, reviews, issues) -> Path:
    """rgb: (3, rows, cols) uint8."""
    image = np.ascontiguousarray(np.moveaxis(rgb, 0, -1)).copy()
    draw_lines(image, grid, [b.polygon.boundary for b in blocks], COLOURS["block"])
    conf = {r["id"]: r["conf"] for r in reviews}
    for level in ("high", "medium", "low"):
        draw_lines(image, grid, [s.geometry for s in sections if conf.get(s.id) == level], COLOURS[level])
    draw_lines(image, grid, [s.bridged for s in sections if s.bridged is not None], COLOURS["bridge"])
    draw_points(image, grid, [f.geometry for f in issues if f.check == "completion"], COLOURS["dangling"], size=3)
    draw_points(image, grid, [f.geometry for f in issues if f.check == "cross_row"], COLOURS["crossing"], size=3)
    return write_png(path, image)
