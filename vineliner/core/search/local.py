"""Exhaustive local search at search-box centres (PROJECT.md §7-8).

At every centre, every (translation, rotation, spacing) hypothesis is scored.
Translations are whole-pixel shifts, so one kernel serves all of them
(PROJECT.md §6.1). For each spacing j the kernels for every rotation, band and
translation are packed into one sparse matrix A_j over a fixed pixel window:

    row  = (rank * n_theta + theta) * n_bands + band
    col  = window pixel the kernel weight falls on
    [num | den] = A_j @ [windows of M*V | windows of M]

so a batch of centres is scored with one sparse-dense product per spacing.
The result equals the direct sums of §6.1 within float32 tolerance.

A search can be restricted to a subset of rotations and spacings (for the
centroid pre-test, pretest.py). Every translation is still scored. Results
always use global rotation and spacing indices; unsearched rotations and
spacings stay NaN in the profiles.

Ties: the winning hypothesis is the highest S; ties go to the smaller spacing
index, then the lower centre-outward rank, then the lower rotation index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

from ..config.settings import ScoringSettings
from ..template import KernelBank
from .hypotheses import SearchSpace
from .persistence import RIDGE_FIELDS, competing_orientations, ridge_statistics
from .scoring import band_statistics, base_score


@dataclass
class CentreResults:
    """Per-centre search outcome. Row/col index the arrays passed to LocalSearch.run."""

    rows: np.ndarray
    cols: np.ndarray
    best_score: np.ndarray  # NaN when no hypothesis had an available centre band
    best_spacing: np.ndarray  # spacing index j
    best_theta: np.ndarray  # rotation index t
    best_offset: np.ndarray  # (N, 2) world-axis whole-pixel offsets (ox, oy)
    best_rank: np.ndarray  # centre-outward rank of the winning translation
    best_means: np.ndarray  # (N, n_bands) m_k of the winner
    best_coverage: np.ndarray  # (N, n_bands) coverage_k of the winner
    theta_profile: np.ndarray  # (N, n_theta) best S per rotation
    spacing_profile: np.ndarray  # (N, n_spacing) best S per spacing
    ridge: np.ndarray  # (N, len(RIDGE_FIELDS)) persistence statistics of the winner
    competing: np.ndarray  # (N, 3) count, theta_deg, score of the strongest other orientation
    # Best hypothesis per rotation, for selection rules that restrict theta (selection.py).
    theta_spacing: np.ndarray  # (N, n_theta) spacing index
    theta_offset: np.ndarray  # (N, n_theta, 2) world-axis offsets (ox, oy)
    theta_means: np.ndarray  # (N, n_theta, n_bands)
    theta_coverage: np.ndarray  # (N, n_theta, n_bands)

    def __len__(self) -> int:
        return len(self.rows)


class LocalSearch:
    def __init__(
        self,
        bank: KernelBank,
        space: SearchSpace,
        scoring: ScoringSettings,
        weights,
        centre_position: int,
        theta_idx=None,
        spacing_idx=None,
    ):
        """theta_idx / spacing_idx: global indices to search (default: all)."""
        self.bank = bank
        self.space = space
        self.scoring = scoring
        self.weights = np.asarray(weights, dtype=np.float32)
        self.centre_position = centre_position
        self.n_theta_all = len(space.thetas)
        self.n_spacing_all = len(space.spacings)
        self.theta_idx = np.arange(self.n_theta_all) if theta_idx is None else np.unique(np.asarray(theta_idx, dtype=np.int64))
        self.spacing_idx = np.arange(self.n_spacing_all) if spacing_idx is None else np.unique(np.asarray(spacing_idx, dtype=np.int64))
        if self.theta_idx.size == 0 or self.spacing_idx.size == 0:
            raise ValueError("a search needs at least one rotation and one spacing")
        self.n_theta = len(self.theta_idx)  # rotations searched
        self.n_bands = bank.n_bands
        self.radius = bank.radius + space.max_translation_px()
        self.size = 2 * self.radius + 1
        thetas = np.asarray(space.thetas)
        step = thetas[1] - thetas[0] if len(thetas) > 1 else 180.0
        self.wrap_theta = abs((thetas[-1] + step) - (thetas[0] + 180.0)) < 1e-6
        self._local_theta = np.full(self.n_theta_all, -1, dtype=np.int64)
        self._local_theta[self.theta_idx] = np.arange(self.n_theta)
        self._matrices = {int(j): self._matrix(int(j)) for j in self.spacing_idx}
        self._totals = {int(j): bank.totals(int(j))[self.theta_idx].astype(np.float32) for j in self.spacing_idx}

    def restricted(self, theta_idx, spacing_idx) -> LocalSearch:
        """A search over a subset of rotations and spacings, sharing kernels and scoring."""
        return LocalSearch(self.bank, self.space, self.scoring, self.weights, self.centre_position, theta_idx, spacing_idx)

    @property
    def is_full(self) -> bool:
        return self.n_theta == self.n_theta_all and len(self.spacing_idx) == self.n_spacing_all

    @property
    def nnz(self) -> int:
        return sum(m.nnz for m in self._matrices.values())

    @property
    def hypotheses_per_centre(self) -> int:
        return sum(self.n_theta * self.space.n_translations(int(j)) for j in self.spacing_idx)

    def _matrix(self, j: int) -> sparse.csr_matrix:
        drow, dcol, weight, seg = self.bank.packed(j, self.theta_idx)
        ox, oy = self.space.offsets(j)
        iy, ix = np.divmod(self.space.order(j), len(ox))
        t_row = -oy[iy]  # array rows run southward
        t_col = ox[ix]
        n_t = t_row.size
        per_t = self.n_theta * self.n_bands
        R, W = self.radius, self.size
        rows = (np.arange(n_t)[:, None] * per_t + seg[None, :]).ravel()
        cols = ((R + t_row[:, None] + drow[None, :]) * W + (R + t_col[:, None] + dcol[None, :])).ravel()
        data = np.broadcast_to(weight, (n_t, weight.size)).ravel()
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_t * per_t, W * W), dtype=np.float32)

    def run(self, mv: np.ndarray, m: np.ndarray, rows: np.ndarray, cols: np.ndarray, batch_size: int = 64, surface_sink=None, progress=None) -> CentreResults:
        """Score every searched hypothesis at every centre (rows, cols) of mv = M*V and m = M.

        The arrays must extend at least self.radius pixels beyond every centre.
        surface_sink(batch_index, centre_slice, surfaces, best_spacing, theta_idx)
        receives, per searched spacing j, the surface S_j[centre, theta, iy, ix] in
        natural offset order (theta along theta_idx), plus each centre's winning
        spacing index.
        """
        R = self.radius
        if rows.size and (rows.min() < R or cols.min() < R or rows.max() >= mv.shape[0] - R or cols.max() >= mv.shape[1] - R):
            raise ValueError("centres too close to the array edge; pad the arrays by LocalSearch.radius")
        N, Th, K = len(rows), self.n_theta, self.n_bands
        ThA, JA = self.n_theta_all, self.n_spacing_all
        out = CentreResults(
            rows=rows,
            cols=cols,
            best_score=np.full(N, np.nan, np.float32),
            best_spacing=np.full(N, -1, np.int64),
            best_theta=np.full(N, -1, np.int64),
            best_offset=np.zeros((N, 2), np.int64),
            best_rank=np.full(N, -1, np.int64),
            best_means=np.full((N, K), np.nan, np.float32),
            best_coverage=np.full((N, K), np.nan, np.float32),
            theta_profile=np.full((N, ThA), -np.inf, np.float32),
            spacing_profile=np.full((N, JA), np.nan, np.float32),
            ridge=np.full((N, len(RIDGE_FIELDS)), np.nan, np.float32),
            competing=np.full((N, 3), np.nan, np.float32),
            theta_spacing=np.full((N, ThA), -1, np.int16),
            theta_offset=np.zeros((N, ThA, 2), np.int16),
            theta_means=np.full((N, ThA, K), np.nan, np.float32),
            theta_coverage=np.full((N, ThA, K), np.nan, np.float32),
        )
        win_mv = sliding_window_view(mv, (self.size, self.size))
        win_m = sliding_window_view(m, (self.size, self.size))
        best_val = np.full(N, -np.inf, np.float32)
        thetas = np.asarray(self.space.thetas)
        tglob = self.theta_idx

        for batch_index, b0 in enumerate(range(0, N, batch_size)):
            sl = slice(b0, min(N, b0 + batch_size))
            C = sl.stop - sl.start
            r, c = rows[sl] - R, cols[sl] - R
            X = np.empty((self.size * self.size, 2 * C), np.float32)
            X[:, :C] = win_mv[r, c].reshape(C, -1).T
            X[:, C:] = win_m[r, c].reshape(C, -1).T
            surfaces = {}
            cidx = np.arange(C)
            for j in (int(j) for j in self.spacing_idx):
                prod = self._matrices[j] @ X
                T = prod.shape[0] // (Th * K)
                num = np.moveaxis(prod[:, :C].reshape(T, Th, K, C), 2, 3)
                den = np.moveaxis(prod[:, C:].reshape(T, Th, K, C), 2, 3)
                means, cov, avail = band_statistics(num, den, self._totals[j][None, :, None, :], self.scoring.normalisation, self.scoring.min_valid_coverage)
                S = base_score(means, avail, self.weights, self.centre_position)  # (T, Th, C)
                Sf = np.where(np.isnan(S), -np.inf, S)
                flat = Sf.reshape(T * Th, C)
                idx = flat.argmax(axis=0)
                val = flat[idx, cidx]
                t_rank, t_local = np.divmod(idx, Th)
                ox, oy = self.space.offsets(j)
                better = val > best_val[sl]
                gi = np.flatnonzero(better)
                if gi.size:
                    g = sl.start + gi
                    best_val[g] = val[gi]
                    out.best_spacing[g] = j
                    out.best_theta[g] = tglob[t_local[gi]]
                    out.best_rank[g] = t_rank[gi]
                    out.best_means[g] = means[t_rank[gi], t_local[gi], gi]
                    out.best_coverage[g] = cov[t_rank[gi], t_local[gi], gi]
                    nat_iy, nat_ix = np.divmod(self.space.order(j)[t_rank[gi]], len(ox))
                    out.best_offset[g, 0] = ox[nat_ix]
                    out.best_offset[g, 1] = oy[nat_iy]
                # Best translation per rotation; strict > keeps the smaller spacing on ties.
                t_best = Sf.argmax(axis=0)  # (Th, C), lowest rank first on ties
                v_best = np.take_along_axis(Sf, t_best[None], axis=0)[0]
                cur = out.theta_profile[sl][:, tglob].T  # (Th, C)
                upd_t, upd_c = np.nonzero(v_best > cur)
                if upd_t.size:
                    tr = t_best[upd_t, upd_c]
                    g = sl.start + upd_c
                    gt = tglob[upd_t]
                    out.theta_profile[g, gt] = v_best[upd_t, upd_c]
                    out.theta_spacing[g, gt] = j
                    nat_iy, nat_ix = np.divmod(self.space.order(j)[tr], len(ox))
                    out.theta_offset[g, gt, 0] = ox[nat_ix]
                    out.theta_offset[g, gt, 1] = oy[nat_iy]
                    out.theta_means[g, gt] = means[tr, upd_t, upd_c]
                    out.theta_coverage[g, gt] = cov[tr, upd_t, upd_c]
                out.spacing_profile[sl, j] = np.where(np.isfinite(val), val, np.nan)
                natural = np.empty_like(S)
                natural[self.space.order(j)] = S
                surfaces[j] = natural.reshape(len(oy), len(ox), Th, C).transpose(3, 2, 0, 1)

            for i in range(C):
                g = sl.start + i
                if not np.isfinite(best_val[g]):
                    continue
                j, t = int(out.best_spacing[g]), int(out.best_theta[g])
                ox, oy = self.space.offsets(j)
                ix = int(np.flatnonzero(ox == out.best_offset[g, 0])[0])
                iy = int(np.flatnonzero(oy == out.best_offset[g, 1])[0])
                out.ridge[g] = ridge_statistics(surfaces[j][i, self._local_theta[t]], ox, oy, self.space.px, self.space.py, thetas[t], iy, ix)
                out.competing[g] = competing_orientations(out.theta_profile[g], thetas, t, self.scoring.accept_min, wrap=self.wrap_theta)
            if surface_sink is not None:
                surface_sink(batch_index, sl, surfaces, out.best_spacing[sl], tglob)
            if progress is not None:
                progress(sl.stop, N)

        out.best_score = np.where(np.isfinite(best_val), best_val, np.nan).astype(np.float32)
        out.theta_profile = np.where(np.isfinite(out.theta_profile), out.theta_profile, np.nan)
        return out
