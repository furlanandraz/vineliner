"""Choosing one hypothesis per centre (peak selection; PROJECT.md §9, §15).

Every hypothesis is scored either way; these rules only decide which scored
hypothesis a centre reports and seeds from.

  best                   the unconstrained winner: highest S over all hypotheses.
  orientation_consensus  the best hypothesis whose rotation lies within
                         window_deg of the local consensus direction.

The consensus at a centre is the score-weighted axial mean of the
unconstrained winning directions of accepted centres within radius_m, the
centre itself included. Rationale: rows in one block are locally near-parallel
(§2), while a 2 m template separates neighbouring rotations only weakly, so
isolated winners at other angles (e.g. diagonals through the vine-plant
lattice) are usually noise. The unconstrained winner is still reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..geo import axial_difference, axial_mean
from .local import CentreResults


@dataclass
class Selection:
    method: str
    score: np.ndarray  # (N,) NaN when nothing scored
    theta: np.ndarray  # (N,) rotation index
    spacing: np.ndarray  # (N,) spacing index
    offset: np.ndarray  # (N, 2) world-axis offsets (ox, oy)
    means: np.ndarray  # (N, n_bands)
    coverage: np.ndarray  # (N, n_bands)
    consensus_deg: np.ndarray  # (N,) NaN without a consensus

    def accepted(self, accept_min: float) -> np.ndarray:
        return np.nan_to_num(self.score, nan=-1.0) >= accept_min


def select_best(results: CentreResults) -> Selection:
    return Selection(
        method="best",
        score=results.best_score.copy(),
        theta=results.best_theta.copy(),
        spacing=results.best_spacing.copy(),
        offset=results.best_offset.copy(),
        means=results.best_means.copy(),
        coverage=results.best_coverage.copy(),
        consensus_deg=np.full(len(results), np.nan),
    )


def select_orientation_consensus(results: CentreResults, x: np.ndarray, y: np.ndarray, thetas, accept_min: float, radius_m: float, window_deg: float) -> Selection:
    sel = select_best(results)
    sel.method = "orientation_consensus"
    thetas = np.asarray(thetas, dtype=float)
    raw_ok = np.nan_to_num(results.best_score, nan=-1.0) >= accept_min
    if not raw_ok.any():
        return sel
    xy = np.column_stack([x, y])
    acc_xy = xy[raw_ok]
    acc_theta = thetas[results.best_theta[raw_ok]]
    acc_w = results.best_score[raw_ok]
    tree = cKDTree(acc_xy)
    for i, neighbours in enumerate(tree.query_ball_point(xy, r=radius_m)):
        if not neighbours:
            continue
        cons = axial_mean(acc_theta[neighbours], weights=acc_w[neighbours])
        sel.consensus_deg[i] = cons
        diff = axial_difference(thetas, cons)
        profile = np.where(np.isfinite(results.theta_profile[i]) & (diff <= window_deg + 1e-9), results.theta_profile[i], -np.inf)
        if not np.isfinite(profile).any():
            sel.score[i] = np.nan
            continue
        best = np.flatnonzero(profile == profile.max())
        t = int(best[np.argmin(diff[best])])  # ties: closest to the consensus
        sel.score[i] = profile[t]
        sel.theta[i] = t
        sel.spacing[i] = results.theta_spacing[i, t]
        sel.offset[i] = results.theta_offset[i, t]
        sel.means[i] = results.theta_means[i, t]
        sel.coverage[i] = results.theta_coverage[i, t]
    return sel
