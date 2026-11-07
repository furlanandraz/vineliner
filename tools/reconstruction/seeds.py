"""Seed segments from accepted local detections (PROJECT.md §10, steps 1-2).

A centre whose selected hypothesis scores S >= accept_min yields one seed per
configured template line k whose band was available. A seed is a straight
segment of the template line length, with the evidence that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo import Grid, across, along
from ..search import SearchSpace, Selection


@dataclass(frozen=True)
class Seed:
    id: int
    block_id: int
    centre_id: int
    line: int  # template index k
    theta_deg: float
    spacing_m: float
    spacing_index: int
    score: float  # S of the selected hypothesis
    band_mean: float  # m_k of this line
    band_coverage: float
    x: float  # midpoint, world
    y: float
    length_m: float

    @property
    def point(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def endpoints(self) -> np.ndarray:
        h = along(self.theta_deg) * self.length_m / 2.0
        return np.array([self.point - h, self.point + h])


def seeds_from_selection(
    selection: Selection,
    rows: np.ndarray,
    cols: np.ndarray,
    centre_ids: np.ndarray,
    grid: Grid,
    space: SearchSpace,
    block_id: int,
    indices: tuple[int, ...],
    lines: tuple[int, ...],
    line_length_m: float,
    accept_min: float,
    min_valid_coverage: float,
    min_band_mean: float = 0.0,
    first_id: int = 0,
) -> list[Seed]:
    """rows/cols are the centres' pixels in grid coordinates.

    A line also needs its own band mean m_k >= min_band_mean. The template
    score can pass on the strength of line 0 while a neighbouring line sits
    on inter-row soil; such a line is not a seed.
    """
    seeds: list[Seed] = []
    cx, cy = grid.array_to_world(rows, cols)
    for i in np.flatnonzero(selection.accepted(accept_min)):
        j, t = int(selection.spacing[i]), int(selection.theta[i])
        theta, d = space.thetas[t], space.spacings[j]
        base = np.array([cx[i] + selection.offset[i, 0] * space.px, cy[i] + selection.offset[i, 1] * space.py])
        n = across(theta)
        for k in lines:
            b = indices.index(k)
            if selection.coverage[i, b] < min_valid_coverage - 1e-6 or selection.means[i, b] < min_band_mean:
                continue
            p = base + k * d * n
            seeds.append(Seed(
                id=first_id + len(seeds),
                block_id=block_id,
                centre_id=int(centre_ids[i]),
                line=k,
                theta_deg=float(theta),
                spacing_m=float(d),
                spacing_index=j,
                score=float(selection.score[i]),
                band_mean=float(selection.means[i, b]),
                band_coverage=float(selection.coverage[i, b]),
                x=float(p[0]),
                y=float(p[1]),
                length_m=line_length_m,
            ))
    return seeds
