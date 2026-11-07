"""Row-direction angle convention (PROJECT.md §6).

theta is measured counterclockwise from the positive easting axis, in degrees.
    u = (cos theta, sin theta)    along-row direction
    n = (-sin theta, cos theta)   across-row direction (u rotated +90 deg)
Angles are axial: theta and theta + 180 describe the same row orientation.
"""

from __future__ import annotations

import numpy as np


def along(theta_deg):
    """Unit along-row vector(s) u, shape (..., 2)."""
    t = np.radians(np.asarray(theta_deg, dtype=float))
    return np.stack([np.cos(t), np.sin(t)], axis=-1)


def across(theta_deg):
    """Unit across-row vector(s) n, shape (..., 2)."""
    t = np.radians(np.asarray(theta_deg, dtype=float))
    return np.stack([-np.sin(t), np.cos(t)], axis=-1)


def axial(theta_deg):
    """Wrap to [0, 180)."""
    return np.mod(np.asarray(theta_deg, dtype=float), 180.0)


def axial_difference(a_deg, b_deg):
    """Smallest angle between two axial directions, in [0, 90]."""
    d = np.mod(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float), 180.0)
    return np.minimum(d, 180.0 - d)


def axial_mean(theta_deg, weights=None) -> float:
    """Mean of axial directions by angle doubling, in [0, 180)."""
    t = np.radians(2.0 * np.asarray(theta_deg, dtype=float))
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
    angle = np.degrees(np.arctan2((w * np.sin(t)).sum(), (w * np.cos(t)).sum())) / 2.0
    return float(np.mod(angle, 180.0))


def heading(vector) -> float:
    """Direction of a 2-D vector in degrees, in [0, 360)."""
    v = np.asarray(vector, dtype=float)
    return float(np.mod(np.degrees(np.arctan2(v[1], v[0])), 360.0))
