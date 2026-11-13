"""Consolidate seeds of the same physical row into chains (PROJECT.md §10, step 1).

Two seeds are linked when, measured in their mean direction:
  - their directions differ by at most link_angle_deg,
  - their midpoints are at most link_lateral_frac * d apart across the row,
  - and at most link_along_m apart along the row.
Linked seeds form connected components. Each component becomes a chain:
seed midpoints are averaged in bins of vertex_spacing_m along the chain's
mean direction (score-weighted), giving one polyline per row piece.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..geo import across, along, axial_difference, axial_mean
from .seeds import Seed


@dataclass
class Chain:
    id: int
    block_id: int
    seed_ids: list[int]
    theta_deg: float
    spacing_m: float
    vertices: np.ndarray  # (n, 2) world, ordered along the chain direction
    max_lateral_spread_m: float  # widest across-row spread of seeds in one bin
    flags: list[str] = field(default_factory=list)


def link_seeds(seeds: list[Seed], angle_deg: float, lateral_frac: float, along_m: float) -> np.ndarray:
    """Component label per seed."""
    n = len(seeds)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    P = np.array([[s.x, s.y] for s in seeds])
    theta = np.array([s.theta_deg for s in seeds])
    d = np.array([s.spacing_m for s in seeds])
    radius = float(np.hypot(along_m, lateral_frac * d.max()))
    pairs = cKDTree(P).query_pairs(r=radius, output_type="ndarray")
    if pairs.size == 0:
        return np.arange(n)
    i, j = pairs[:, 0], pairs[:, 1]
    # Mean of two axial directions by angle doubling.
    t2 = np.radians(2 * theta)
    mean = np.degrees(np.arctan2(np.sin(t2[i]) + np.sin(t2[j]), np.cos(t2[i]) + np.cos(t2[j]))) / 2
    u, nv = along(mean), across(mean)
    delta = P[j] - P[i]
    lateral = np.abs((delta * nv).sum(axis=1))
    along_dist = np.abs((delta * u).sum(axis=1))
    ok = (
        (axial_difference(theta[i], theta[j]) <= angle_deg)
        & (lateral <= lateral_frac * 0.5 * (d[i] + d[j]))
        & (along_dist <= along_m)
    )
    graph = sparse.coo_matrix((np.ones(ok.sum()), (i[ok], j[ok])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def build_chains(seeds: list[Seed], labels: np.ndarray, vertex_spacing_m: float, smoothing_m: float = 0.0, first_id: int = 0) -> list[Chain]:
    """One chain per component. smoothing_m > 0 averages each vertex's across-row
    position with the vertices within smoothing_m / 2 along the chain (weighted
    by seed score), which removes bin-to-bin wobble from disagreeing seeds."""
    chains: list[Chain] = []
    for label in np.unique(labels):
        members = [seeds[i] for i in np.flatnonzero(labels == label)]
        P = np.array([[s.x, s.y] for s in members])
        w = np.array([s.score for s in members])
        theta = axial_mean([s.theta_deg for s in members], weights=w)
        u, n = along(theta), across(theta)
        a = P @ u
        b = P @ n
        half = members[0].length_m / 2.0
        spread = 0.0
        if a.max() - a.min() < vertex_spacing_m:
            centre = (P * w[:, None]).sum(axis=0) / w.sum()
            vertices = np.array([centre - u * half, centre + u * half])
            spread = float(b.max() - b.min())
        else:
            bins = np.floor((a - a.min()) / vertex_spacing_m).astype(int)
            pts, bin_w = [], []
            for k in np.unique(bins):
                sel = bins == k
                ws = w[sel]
                pts.append(((a[sel] * ws).sum() / ws.sum(), (b[sel] * ws).sum() / ws.sum()))
                bin_w.append(ws.sum())
                spread = max(spread, float(b[sel].max() - b[sel].min()))
            ab = np.array(pts)
            if smoothing_m > 0 and len(ab) > 2:
                bw = np.array(bin_w)
                near = np.abs(ab[:, None, 0] - ab[None, :, 0]) <= smoothing_m / 2
                ab[:, 1] = (near * bw[None, :] * ab[None, :, 1]).sum(axis=1) / (near * bw[None, :]).sum(axis=1)
            vertices = ab[:, :1] * u + ab[:, 1:] * n
            # Seeds extend half a line length beyond their midpoints.
            t0 = _unit(vertices[0] - vertices[1])
            t1 = _unit(vertices[-1] - vertices[-2])
            vertices = np.vstack([vertices[0] + t0 * half, vertices, vertices[-1] + t1 * half])
        d = float(np.median([s.spacing_m for s in members]))
        chain = Chain(
            id=first_id + len(chains),
            block_id=members[0].block_id,
            seed_ids=[s.id for s in members],
            theta_deg=theta,
            spacing_m=d,
            vertices=vertices,
            max_lateral_spread_m=spread,
        )
        if spread > 0.5 * d:
            chain.flags.append("lateral_spread")  # seeds of neighbouring rows may have been linked
        chains.append(chain)
    return chains


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.hypot(v[0], v[1])
    return v / norm if norm > 0 else v
