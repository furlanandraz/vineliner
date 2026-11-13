"""Row smoothing: a maximum direction change between vertices ([smoothing]).

The turn at a vertex is the angle between the segment arriving at it and the
segment leaving it. While any interior vertex turns more than max_turn_deg,
the vertex with the largest turn is removed. The two segments around it become
one, which is bridged if either of them was. The two end vertices are never
removed, so rows keep ending where they met the boundary.

This removes spikes and lines that double back (for example where two row
pieces were joined), and caps jitter. With vertices about 1 m apart, 20 deg
corresponds to a curve radius of about 2.9 m, so curves within the 5-10 m
radius assumed in PROJECT.md §2 are kept.
"""

from __future__ import annotations

import numpy as np


MIN_SEGMENT_M = 0.01  # shorter segments have no meaningful direction


def clean_line(points, max_turn_deg: float | None = None) -> np.ndarray:
    """Final tidy-up of a clipped, snapped row: drop vertices within MIN_SEGMENT_M of
    their neighbour (never the ends), then apply the turn limit if one is given."""
    c = np.asarray(points, dtype=float)
    if len(c) <= 2:
        return c
    keep = [c[0]]
    for p in c[1:-1]:
        if np.hypot(*(p - keep[-1])) >= MIN_SEGMENT_M:
            keep.append(p)
    while len(keep) > 1 and np.hypot(*(c[-1] - keep[-1])) < MIN_SEGMENT_M:
        keep.pop()
    keep.append(c[-1])
    if len(keep) < 2:
        keep = [c[0], c[-1]]
    out = np.array(keep)
    if max_turn_deg is not None and len(out) > 2:
        out, _ = limit_turns(out, max_turn_deg)
    return out


def turn_angles(points: np.ndarray) -> np.ndarray:
    """Direction change in degrees at each interior vertex, in [0, 180]."""
    v = np.diff(np.asarray(points, dtype=float), axis=0)
    a = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return np.abs((np.diff(a) + 180.0) % 360.0 - 180.0)


def limit_turns(points, max_turn_deg: float, support=None):
    """Return (points, support) with no interior turn above max_turn_deg.

    points: (n, 2) vertices. support: optional per-segment flags (n - 1), True
    where the segment had image support; it is kept aligned with the points.
    """
    pts = [np.asarray(p, dtype=float) for p in points]
    sup = list(support) if support is not None else None
    # Repeated vertices have no direction; drop them first.
    keep = [0] + [i for i in range(1, len(pts)) if np.any(np.abs(pts[i] - pts[i - 1]) > 1e-9)]
    if len(keep) < len(pts):
        if sup is not None:  # a kept segment covers the original segments between its two vertices
            sup = [all(sup[keep[k]:keep[k + 1]]) for k in range(len(keep) - 1)]
        pts = [pts[i] for i in keep]
    while len(pts) > 2:
        turns = turn_angles(np.array(pts))
        worst = int(np.argmax(turns))
        if turns[worst] <= max_turn_deg + 1e-9:
            break
        i = worst + 1  # turns[k] belongs to vertex k + 1
        del pts[i]
        if sup is not None:
            sup[i - 1:i + 1] = [sup[i - 1] and sup[i]]
    return np.array(pts), sup
