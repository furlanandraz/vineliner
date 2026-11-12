"""Overlap scoring (PROJECT.md §6.1, §8).

With kernel K_k, validity mask M and row evidence V, for one hypothesis:

    num_k = sum K_k * M * V,   den_k = sum K_k * M,   tot_k = sum K_k
    coverage_k = den_k / tot_k
    m_k = num_k / den_k        (normalisation = "renormalise_valid")
    m_k = num_k / tot_k        (normalisation = "full_kernel")

A band is available when coverage_k >= min_valid_coverage. Near a block edge
some bands fall outside and are unavailable. Their weights are mirrored: the
weight of an unavailable band k moves to band -k on the other side of the
centre, if that one is available. Weights are then rescaled to sum to 1, which
only matters when a band and its mirror are both unavailable:

    w'_k = (w_k + w_-k * unavailable_-k) * available_k
    S = sum_k w'_k * m_k / sum_k w'_k

With the agreed weights and one side outside, that gives 0.50 centre,
0.40 inner, 0.10 outer: the centre keeps its share and the remaining side keeps
the neighbour support that holds a template in its lane. With full support S =
0.50 m_0 + 0.20 (m_-1 + m_+1) + 0.05 (m_-2 + m_+2), the agreed score. S is NaN
when the centre band is unavailable. S is an overlap/matching score, not a
probability.
"""

from __future__ import annotations

import numpy as np

from ..config.settings import ScoringSettings
from ..geo import Grid, axial_difference

_COVERAGE_TOLERANCE = 1e-6


def band_statistics(num, den, tot, normalisation: str, min_valid_coverage: float):
    """Return (means, coverage, available); arrays broadcast with the band axis last."""
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    tot = np.asarray(tot, dtype=np.float32)
    coverage = den / tot
    if normalisation == "renormalise_valid":
        means = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    elif normalisation == "full_kernel":
        means = num / tot
    else:
        raise ValueError(f"unknown normalisation {normalisation!r}")
    available = coverage >= min_valid_coverage - _COVERAGE_TOLERANCE
    return means, coverage, available


def mirrored_weights(weights, available):
    """Effective band weights (band axis last): unavailable weights move to the mirror band.

    The template's bands are symmetric about the centre, so band k's mirror is
    the band at the reversed position.
    """
    w = np.asarray(weights, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    lost = w * ~available
    return (w + lost[..., ::-1]) * available


def base_score(means, available, weights, centre_position: int):
    """Combine band means (band axis last) into S. NaN where the centre band is unavailable."""
    wa = mirrored_weights(weights, available)
    weighted = (wa * means).sum(axis=-1)
    total = wa.sum(axis=-1)
    score = np.divide(weighted, total, out=np.zeros_like(weighted), where=total > 0)
    score = np.where(available[..., centre_position], score, np.nan)
    return score.astype(np.float32)


def score_hypothesis(kernels, mv: np.ndarray, m: np.ndarray, row: int, col: int, weights, centre_position: int, scoring: ScoringSettings):
    """Score one hypothesis whose template centre is pixel (row, col) of mv/m.

    mv = M * V and m = M as float arrays; the caller pads them so every kernel
    offset stays inside. Returns (S, means, coverage).
    """
    num = np.array([np.dot(k.weight, mv[row + k.drow, col + k.dcol]) for k in kernels], dtype=np.float32)
    den = np.array([np.dot(k.weight, m[row + k.drow, col + k.dcol]) for k in kernels], dtype=np.float32)
    tot = np.array([k.total for k in kernels], dtype=np.float32)
    means, coverage, available = band_statistics(num, den, tot, scoring.normalisation, scoring.min_valid_coverage)
    score = base_score(means, available, weights, centre_position)
    return float(score), means, coverage


class PointScorer:
    """Template score at any world point, for row growth (PROJECT.md §10, step 4).

    Uses the same kernels and scoring as the search. The template centre is
    the pixel containing the point and theta snaps to the nearest searched
    rotation, so position resolution is one pixel, as in the search.
    """

    def __init__(self, bank, grid: Grid, mv: np.ndarray, m: np.ndarray, weights, centre_position: int, scoring: ScoringSettings):
        self.bank = bank
        self.grid = grid
        self.mv = mv
        self.m = m
        self.weights = weights
        self.centre_position = centre_position
        self.scoring = scoring
        self.thetas = np.asarray(bank.thetas)

    def theta_index(self, theta_deg: float) -> int:
        return int(np.argmin(axial_difference(self.thetas, theta_deg)))

    def __call__(self, x: float, y: float, theta_deg: float, spacing_index: int) -> float:
        row, col = self.grid.nearest_pixel(x, y)
        row, col = int(row), int(col)
        R = self.bank.radius
        if row < R or col < R or row >= self.mv.shape[0] - R or col >= self.mv.shape[1] - R:
            return float("nan")
        kernels = self.bank.get(self.theta_index(theta_deg), spacing_index)
        score, _, _ = score_hypothesis(kernels, self.mv, self.m, row, col, self.weights, self.centre_position, self.scoring)
        return score
