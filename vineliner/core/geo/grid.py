"""The analysis pixel grid and the single world <-> array conversion (PROJECT.md §3, §6.1).

Conventions used everywhere in the core:

- World coordinates are projected easting x and northing y, in metres.
- Array coordinates are (row, col). Integer values are pixel centres, so pixel
  (r, c) covers rows r - 0.5 .. r + 0.5 and cols c - 0.5 .. c + 0.5.
- Rows increase southward. A northward world vector has a negative row
  component. This module is the only place that applies that sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """North-up affine pixel grid.

    The affine follows GDAL/rasterio order, for pixel-corner coordinates:
        x = a * col + b * row + c
        y = d * col + e * row + f
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    width: int
    height: int
    crs_wkt: str | None = None

    @classmethod
    def from_affine(cls, affine, width: int, height: int, crs_wkt: str | None = None) -> Grid:
        a, b, c, d, e, f = (float(v) for v in tuple(affine)[:6])
        return cls(a, b, c, d, e, f, int(width), int(height), crs_wkt)

    @property
    def affine(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.d, self.e, self.f)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_size(self) -> tuple[float, float]:
        """(px, py): pixel width along easting and height along northing, both positive."""
        return (self.a, -self.e)

    @property
    def is_north_up(self) -> bool:
        return self.b == 0 and self.d == 0 and self.a > 0 and self.e < 0

    def require_north_up(self) -> None:
        if not self.is_north_up:
            raise ValueError(
                f"grid affine {self.affine} has rotation/shear or flipped axes; "
                "the first version needs a north-up raster (PROJECT.md §3)"
            )

    def require_square_pixels(self, rel_tol: float = 1e-9) -> float:
        px, py = self.pixel_size
        if abs(px - py) > rel_tol * max(px, py):
            raise ValueError(f"pixels are {px} x {py} m; non-square pixels are not supported yet")
        return px

    def world_to_array(self, x, y):
        """World (x, y) -> fractional (row, col), integers at pixel centres."""
        col = (np.asarray(x, dtype=float) - self.c) / self.a - 0.5
        row = (np.asarray(y, dtype=float) - self.f) / self.e - 0.5
        return row, col

    def array_to_world(self, row, col):
        """Fractional (row, col) -> world (x, y). (0, 0) is the centre of the top-left pixel."""
        x = self.c + (np.asarray(col, dtype=float) + 0.5) * self.a
        y = self.f + (np.asarray(row, dtype=float) + 0.5) * self.e
        return x, y

    def nearest_pixel(self, x, y):
        """World (x, y) -> integer (row, col) of the pixel containing the point."""
        row, col = self.world_to_array(x, y)
        return np.floor(row + 0.5).astype(np.int64), np.floor(col + 0.5).astype(np.int64)

    def world_vector_to_array(self, dx, dy):
        """World displacement (dx, dy) in metres -> array displacement (drow, dcol)."""
        return np.asarray(dy, dtype=float) / self.e, np.asarray(dx, dtype=float) / self.a

    def array_vector_to_world(self, drow, dcol):
        """Array displacement (drow, dcol) -> world displacement (dx, dy) in metres."""
        return np.asarray(dcol, dtype=float) * self.a, np.asarray(drow, dtype=float) * self.e

    def window(self, row0: int, col0: int, height: int, width: int) -> Grid:
        """Sub-grid whose pixel (0, 0) is this grid's pixel (row0, col0). It may extend outside."""
        x0 = self.c + col0 * self.a + row0 * self.b
        y0 = self.f + col0 * self.d + row0 * self.e
        return Grid(self.a, self.b, x0, self.d, self.e, y0, int(width), int(height), self.crs_wkt)

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the pixel footprints."""
        x0, x1 = self.c, self.c + self.width * self.a
        y0, y1 = self.f, self.f + self.height * self.e
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def same_grid(self, other: Grid, tol: float = 1e-9) -> bool:
        return (
            self.width == other.width
            and self.height == other.height
            and all(abs(p - q) <= tol * max(1.0, abs(p)) for p, q in zip(self.affine, other.affine))
        )
