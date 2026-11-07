"""Synthetic striped vineyard with known rows, shared by the tests."""

from __future__ import annotations

import numpy as np
from tools.geo import Grid, across


SYNTHETIC_THETA = 30.0
SYNTHETIC_SPACING = 1.5
SYNTHETIC_PIXEL = 0.2
SYNTHETIC_ORIGIN = (500000.0, 100060.0)  # upper-left corner
SYNTHETIC_SIZE = 300  # pixels per side -> 60 m


def synthetic_grid(crs_wkt: str | None = None) -> Grid:
    x0, y0 = SYNTHETIC_ORIGIN
    return Grid(SYNTHETIC_PIXEL, 0.0, x0, 0.0, -SYNTHETIC_PIXEL, y0, SYNTHETIC_SIZE, SYNTHETIC_SIZE, crs_wkt)


def stripe_evidence(grid: Grid, theta=SYNTHETIC_THETA, spacing=SYNTHETIC_SPACING, half_width=0.25, phase=0.0) -> np.ndarray:
    """V = 1 within half_width of a row line, else 0. Row lines: (p - origin) . n = phase + i * spacing."""
    rows, cols = np.indices(grid.shape)
    x, y = grid.array_to_world(rows, cols)
    n = across(theta)
    s = (x - grid.c) * n[0] + (y - grid.f) * n[1] - phase
    dist = np.abs(s - spacing * np.round(s / spacing))
    return (dist <= half_width).astype(np.float32)
