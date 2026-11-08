"""Centroid pre-test: a cheaper search narrowed by one box near each block's centroid
(PROJECT.md §7, §17 items 23-24).

For each block:

1. If the polygon's centroid lies inside the polygon, a search box on the
   centroid pixel is searched in full: every translation (phase), all
   rotations and all spacings.
2. If that box is weak (its best score is below accept_min, or its pixel is not
   valid), rings of neighbouring boxes around it are tested the same way, ring
   by ring, spaced like the grid centres (ring 1 = the 8 boxes around the
   centroid, ring 2 = the 16 around those, ...). The first ring with an
   accepted box wins; within it, the highest score.
3. The winning box gives the block's pre-test direction theta_pt and spacing
   d_pt. No assumption is made that the centre has the best data; the
   pre-test only narrows the grid search.
4. The grid search then still scores every translation at every centre, but
   only rotations within theta_window_deg of theta_pt and spacings within
   spacing_window_frac * d_pt of d_pt.

The block gets the full grid search when the centroid is outside the polygon,
or when no box up to neighbour_rings is accepted. The reason is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from ..geo import Grid, axial_difference
from .local import LocalSearch


@dataclass
class Pretest:
    block_id: int
    mode: str  # "pretest" (grid search narrowed) or "full"
    reason: str
    centroid: tuple[float, float] | None = None
    x: float | None = None  # centre of the box that set the pre-test
    y: float | None = None
    ring: int | None = None  # 0 = the centroid box, 1.. = neighbouring rings
    boxes_tested: int = 0
    theta_deg: float | None = None
    spacing_m: float | None = None
    score: float | None = None
    theta_idx: np.ndarray | None = None  # rotations the grid search uses
    spacing_idx: np.ndarray | None = None  # spacings the grid search uses

    def describe(self, thetas, spacings) -> dict:
        out = {
            "mode": self.mode,
            "reason": self.reason,
            "centroid": list(self.centroid) if self.centroid else None,
            "box": [self.x, self.y] if self.x is not None else None,
            "ring": self.ring,
            "boxes_tested": self.boxes_tested,
            "theta_deg": self.theta_deg,
            "spacing_m": self.spacing_m,
            "score": self.score,
        }
        if self.theta_idx is not None:
            t = np.asarray(thetas)[self.theta_idx]
            d = np.asarray(spacings)[self.spacing_idx]
            out["rotations_searched"] = [float(t.min()), float(t.max()), int(t.size)]
            out["spacings_searched_m"] = [float(d.min()), float(d.max()), int(d.size)]
        return out


def ring_offsets(ring: int) -> list[tuple[int, int]]:
    """(drow, dcol) steps on the perimeter of the square ring at Chebyshev distance `ring`."""
    if ring == 0:
        return [(0, 0)]
    return [(dr, dc) for dr in range(-ring, ring + 1) for dc in range(-ring, ring + 1) if max(abs(dr), abs(dc)) == ring]


def run_pretest(
    block,
    grid: Grid,
    valid_block: np.ndarray,
    search: LocalSearch,
    mv: np.ndarray,
    m: np.ndarray,
    origin: tuple[int, int],
    accept_min: float,
    theta_window_deg: float,
    spacing_window_frac: float,
    box_spacing_m: float,
    neighbour_rings: int,
) -> Pretest:
    """valid_block: valid & inside-block mask on grid; mv/m: the block window arrays starting at origin (row0, col0)."""
    polygon = block.polygon
    c = polygon.centroid
    if c.is_empty or not polygon.contains(c):
        return Pretest(block.id, "full", "centroid outside the polygon")
    centroid = (float(c.x), float(c.y))
    crow, ccol = (int(v) for v in grid.nearest_pixel(c.x, c.y))
    step = max(1, int(round(box_spacing_m / min(grid.pixel_size))))
    full = search if search.is_full else search.restricted(None, None)
    thetas = np.asarray(full.space.thetas)
    spacings = np.asarray(full.space.spacings)
    tested, best_seen = 0, None

    for ring in range(neighbour_rings + 1):
        rows, cols = [], []
        for dr, dc in ring_offsets(ring):
            r, k = crow + dr * step, ccol + dc * step
            if not (0 <= r < grid.height and 0 <= k < grid.width) or not valid_block[r, k]:
                continue
            x, y = grid.array_to_world(r, k)
            if shapely.intersects_xy(polygon, float(x), float(y)):
                rows.append(r)
                cols.append(k)
        if not rows:
            continue
        rows, cols = np.array(rows), np.array(cols)
        res = full.run(mv, m, rows - origin[0], cols - origin[1])
        tested += rows.size
        scores = np.nan_to_num(res.best_score, nan=-1.0)
        top = int(np.argmax(scores))
        best_seen = max(best_seen or -1.0, float(scores[top]))
        if scores[top] < accept_min:
            continue
        theta_pt = float(thetas[res.best_theta[top]])
        d_pt = float(spacings[res.best_spacing[top]])
        x, y = (float(v) for v in grid.array_to_world(rows[top], cols[top]))
        reason = "centroid box accepted" if ring == 0 else f"centroid box weak; neighbouring box in ring {ring} accepted"
        return Pretest(
            block.id, "pretest", reason, centroid, x, y, ring, tested, theta_pt, d_pt, float(scores[top]),
            np.flatnonzero(axial_difference(thetas, theta_pt) <= theta_window_deg + 1e-9),
            np.flatnonzero(np.abs(spacings - d_pt) <= spacing_window_frac * d_pt + 1e-9),
        )

    seen = f"best S {best_seen:.3f}" if best_seen is not None and best_seen >= 0 else "no valid box"
    return Pretest(
        block.id, "full", f"no box within {neighbour_rings} ring(s) of the centroid reached accept_min {accept_min} ({seen})",
        centroid, boxes_tested=tested,
    )


def window_edge_share(theta_deg_selected: np.ndarray, pretest: Pretest, thetas, step_deg: float) -> float:
    """Share of selected directions on the edge of the pre-test rotation window.

    A large share suggests the true direction lies outside the window, i.e. the
    pre-test was off; widen theta_window_deg or turn the pre-test off.
    """
    if pretest.mode != "pretest" or theta_deg_selected.size == 0:
        return 0.0
    window = np.asarray(thetas)[pretest.theta_idx]
    edge = axial_difference(window, pretest.theta_deg).max()
    return float(np.mean(axial_difference(theta_deg_selected, pretest.theta_deg) >= edge - step_deg / 2))
