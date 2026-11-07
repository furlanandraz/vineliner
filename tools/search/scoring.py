"""Overlap scoring (PROJECT.md §6.1, §8).

With kernel K_k, validity mask M and row evidence V, for one hypothesis:

    num_k = sum K_k * M * V,   den_k = sum K_k * M,   tot_k = sum K_k
    coverage_k = den_k / tot_k
    m_k = num_k / den_k        (normalisation = "renormalise_valid")
    m_k = num_k / tot_k        (normalisation = "full_kernel")

A band is available when coverage_k >= min_valid_coverage. The base score is

    S = sum_k w_k * m_k * available_k                       ("zero" policy)
    S = sum_k w_k * m_k * available_k / sum_k w_k * available_k   ("renormalise")

With full support both give the agreed S = 0.50 m_0 + 0.20 (m_-1 + m_+1)
+ 0.05 (m_-2 + m_+2). S is NaN when the centre band is unavailable. S is an
overlap/matching score, not a probability.
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


def base_score(means, available, weights, centre_position: int, policy: str):
    """Combine band means (band axis last) into S. NaN where the centre band is unavailable."""
    w = np.asarray(weights, dtype=np.float32)
    wa = w * available
    weighted = (wa * means).sum(axis=-1)
    if policy == "zero":
        score = weighted
    elif policy == "renormalise":
        total = wa.sum(axis=-1)
        score = np.divide(weighted, total, out=np.zeros_like(weighted), where=total > 0)
    else:
        raise ValueError(f"unknown unavailable-band policy {policy!r}")
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
    score = base_score(means, available, weights, centre_position, scoring.unavailable_bands)
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
