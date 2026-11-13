"""Directional persistence across translations (PROJECT.md §9).

Diagnostics only. Nothing here rejects a hypothesis; the statistics are stored
so persistence rules can be chosen later from evidence.

For the winning (theta, d) at a centre, the translation surface S(dx, dy) is
read along two lines through the winning translation:
  along  - cells on the line in the row direction u (should stay high on a row)
  across - cells on the line in the across direction n (should fall off)
A cell is on a line when its perpendicular distance to it is at most
0.5 * p * max(|u_x|, |u_y|), which picks one cell per step along the line.
"""

from __future__ import annotations

import numpy as np

from ..geo import across, along, axial_difference

RIDGE_FIELDS = ("along_mean", "along_n", "across_mean", "across_n", "box_mean", "box_min", "box_max")


def ridge_statistics(surface, ox, oy, px, py, theta_deg, iy, ix) -> tuple[float, ...]:
    """Statistics of one (Ty, Tx) translation surface, in RIDGE_FIELDS order."""
    gy, gx = np.meshgrid((oy - oy[iy]) * py, (ox - ox[ix]) * px, indexing="ij")
    u, n = along(theta_deg), across(theta_deg)
    a = gx * u[0] + gy * u[1]
    b = gx * n[0] + gy * n[1]
    half = 0.5 * max(px, py) * max(abs(u[0]), abs(u[1])) * (1 + 1e-9)
    finite = np.isfinite(surface)
    on_along = (np.abs(b) <= half) & finite
    on_across = (np.abs(a) <= half) & finite
    values = surface[finite]

    def mean(mask):
        return float(surface[mask].mean()) if mask.any() else float("nan")

    return (
        mean(on_along),
        float(on_along.sum()),
        mean(on_across),
        float(on_across.sum()),
        float(values.mean()) if values.size else float("nan"),
        float(values.min()) if values.size else float("nan"),
        float(values.max()) if values.size else float("nan"),
    )


def competing_orientations(profile, thetas, best_t: int, accept_min: float, min_separation_deg: float = 10.0, wrap: bool = True):
    """Other accepted local maxima of the per-rotation best score.

    Returns (count, theta_deg, score) of the strongest competitor, or
    (0, nan, nan) when none reaches accept_min at least min_separation_deg away.
    """
    p = np.where(np.isfinite(profile), profile, -np.inf)
    if wrap:
        left, right = np.roll(p, 1), np.roll(p, -1)
    else:
        left = np.concatenate([[-np.inf], p[:-1]])
        right = np.concatenate([p[1:], [-np.inf]])
    peaks = (p >= left) & (p >= right) & (p >= accept_min)
    far = axial_difference(np.asarray(thetas), thetas[best_t]) >= min_separation_deg
    candidates = np.flatnonzero(peaks & far)
    if candidates.size == 0:
        return 0, float("nan"), float("nan")
    top = candidates[np.argmax(p[candidates])]
    return int(candidates.size), float(thetas[top]), float(p[top])
