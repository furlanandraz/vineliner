"""Five-line template geometry (PROJECT.md §6).

For grid centre p, translation Delta, spacing d and angle theta, line k is

    q(k, t) = p + Delta + k*d*n + t*u,   -L/2 <= t <= L/2

Each line is sampled as a band: the segment buffered by w/2 across the row,
with flat ends. w is a sampling width for the kernel, not a physical row width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo import across, along


@dataclass(frozen=True)
class Template:
    indices: tuple[int, ...]  # e.g. (-2, -1, 0, 1, 2)
    weights: tuple[float, ...]  # e.g. (0.05, 0.20, 0.50, 0.20, 0.05)
    line_length_m: float
    band_width_m: float

    @property
    def centre_position(self) -> int:
        """Position of line 0 in indices/weights."""
        return self.indices.index(0)

    def band_centre(self, theta_deg: float, spacing_m: float, k: int, origin=(0.0, 0.0)) -> np.ndarray:
        return np.asarray(origin, dtype=float) + k * spacing_m * across(theta_deg)

    def band_corners(self, theta_deg: float, spacing_m: float, k: int, origin=(0.0, 0.0)) -> np.ndarray:
        """(4, 2) world corners of band k, counterclockwise."""
        c = self.band_centre(theta_deg, spacing_m, k, origin)
        hu = along(theta_deg) * (self.line_length_m / 2.0)
        hn = across(theta_deg) * (self.band_width_m / 2.0)
        return np.array([c - hu - hn, c + hu - hn, c + hu + hn, c - hu + hn])

    def line_endpoints(self, theta_deg: float, spacing_m: float, k: int, origin=(0.0, 0.0)) -> np.ndarray:
        """(2, 2) world endpoints of line k."""
        c = self.band_centre(theta_deg, spacing_m, k, origin)
        hu = along(theta_deg) * (self.line_length_m / 2.0)
        return np.array([c - hu, c + hu])
