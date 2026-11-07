"""Geometry checks on final row sections (PROJECT.md §11), reported, never failed.

Each check produces issues (located on the map) and per-section measurements.
review.py turns the measurements into attributes and a review flag, so that
uncertainty is exposed for a human rather than rows being rejected.

  self_intersection  a section crosses itself
  cross_row          two sections of the block intersect
  spacing            at 1 m samples, the nearest neighbouring section along the
                     local normal is closer than spacing_min_m or farther than
                     spacing_max_m. Rays reach 2 * spacing_max_m, so a missing
                     row shows up as a too-wide gap; no section within reach
                     means an outer row, which is not an issue.
  containment        a section extends outside the polygon by more than tolerance_m
  completion         an endpoint is farther than tolerance_m from the boundary (dangling)

Passing every check does not prove the image was read correctly: missing rows
can pass, and an empty result is not a successful extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ..config.settings import ValidationSettings
from ..reconstruction import RowSection

CHECKS = ("self_intersection", "cross_row", "spacing", "containment", "completion")
RAY_REACH_FACTOR = 2.0


@dataclass
class Issue:
    check: str
    block_id: int
    section_ids: tuple[int, ...]
    geometry: BaseGeometry
    value: float | None = None
    detail: str = ""


@dataclass
class SectionStats:
    """Check measurements for one section."""

    spacing_samples: int = 0  # samples that found a neighbour
    spacing_narrow: int = 0
    spacing_wide: int = 0
    spacing_values: list[float] = field(default_factory=list)
    narrow_partners: dict[int, int] = field(default_factory=dict)  # section id -> narrow samples against it
    crossings: int = 0
    self_intersection: bool = False
    outside_m: float = 0.0
    dangling: list[str] = field(default_factory=list)  # "start"/"end" sides that dangle

    @property
    def spacing_out(self) -> int:
        return self.spacing_narrow + self.spacing_wide


def meets_boundary_twice(line, polygon, tol: float) -> bool:
    """True when the line meets the block boundary at exactly two points: its two ends.

    Both endpoints must lie within tol of the boundary, and the rest of the line
    must not touch the boundary. This is the test behind [export] drop_hanging_rows.
    """
    boundary = polygon.boundary
    coords = line.coords
    if boundary.distance(Point(coords[0])) > tol or boundary.distance(Point(coords[-1])) > tol:
        return False
    if line.length <= 2 * tol:
        return False
    inner = shapely.ops.substring(line, tol, line.length - tol)
    return not inner.intersects(boundary)


def split_hanging(sections: list[RowSection], polygon, tol: float) -> tuple[list[RowSection], list[RowSection]]:
    """(rows meeting the boundary exactly twice, all other rows)."""
    kept, hanging = [], []
    for s in sections:
        (kept if meets_boundary_twice(s.geometry, polygon, tol) else hanging).append(s)
    return kept, hanging


def check_block(sections: list[RowSection], polygon, settings: ValidationSettings, block_id: int):
    """Return (issues, stats by section id, summary)."""
    issues: list[Issue] = []
    stats = {s.id: SectionStats() for s in sections}
    tol = settings.tolerance_m
    geoms = [s.geometry for s in sections]
    ids = [s.id for s in sections]

    for s in sections:
        if not s.geometry.is_simple:
            stats[s.id].self_intersection = True
            issues.append(Issue("self_intersection", block_id, (s.id,), s.geometry))

    if geoms:
        tree = shapely.STRtree(geoms)
        pairs = tree.query(geoms, predicate="intersects")
        for i, j in zip(*pairs):
            if i < j:
                hit = geoms[i].intersection(geoms[j])
                stats[ids[i]].crossings += 1
                stats[ids[j]].crossings += 1
                issues.append(Issue("cross_row", block_id, (ids[i], ids[j]), hit.representative_point() if not hit.is_empty else geoms[i]))
        issues += _spacing(sections, tree, settings, block_id, stats)

    grown = polygon.buffer(tol)
    for s in sections:
        if not grown.covers(s.geometry):
            outside = s.geometry.difference(polygon)
            stats[s.id].outside_m = float(outside.length)
            issues.append(Issue("containment", block_id, (s.id,), outside, value=float(outside.length)))

    boundary = polygon.boundary
    for s in sections:
        for side, coord, state in (("start", s.geometry.coords[0], s.start_state), ("end", s.geometry.coords[-1], s.end_state)):
            p = Point(coord)
            dist = boundary.distance(p)
            if dist > tol:
                stats[s.id].dangling.append(side)
                issues.append(Issue("completion", block_id, (s.id,), p, value=float(dist), detail=f"{side} end stops inside the block ({state})"))

    summary = {
        "block_id": block_id,
        "sections": len(sections),
        "total_length_m": float(sum(g.length for g in geoms)),
        "issues": {c: sum(1 for f in issues if f.check == c) for c in CHECKS},
        "sections_with_issue": {c: len({sid for f in issues if f.check == c for sid in f.section_ids}) for c in CHECKS},
        "empty": len(sections) == 0,
    }
    return issues, stats, summary


def _spacing(sections, tree, settings: ValidationSettings, block_id: int, stats) -> list[Issue]:
    reach = RAY_REACH_FACTOR * settings.spacing_max_m
    origins, ends, owners = [], [], []
    for idx, s in enumerate(sections):
        line = s.geometry
        length = line.length
        for d in np.arange(settings.sample_step_m / 2, length, settings.sample_step_m):
            p = np.array(line.interpolate(d).coords[0])
            a = np.array(line.interpolate(max(d - 0.5, 0.0)).coords[0])
            b = np.array(line.interpolate(min(d + 0.5, length)).coords[0])
            t = b - a
            norm = np.hypot(*t)
            if norm == 0:
                continue
            n = np.array([-t[1], t[0]]) / norm
            for sign in (1.0, -1.0):
                origins.append(p)
                ends.append(p + sign * reach * n)
                owners.append(idx)
    if not origins:
        return []
    origins = np.array(origins)
    rays = shapely.linestrings(np.stack([origins, np.array(ends)], axis=1))
    owners = np.array(owners)
    ray_idx, geom_idx = tree.query(rays, predicate="intersects")
    keep = geom_idx != owners[ray_idx]
    ray_idx, geom_idx = ray_idx[keep], geom_idx[keep]
    nearest = np.full(len(origins), np.inf)
    nearest_geom = np.full(len(origins), -1)
    if ray_idx.size:
        hits = shapely.intersection(rays[ray_idx], tree.geometries.take(geom_idx))
        dist = shapely.distance(shapely.points(origins[ray_idx]), hits)
        order = np.lexsort((dist, ray_idx))  # per ray, nearest hit first
        first = order[np.r_[True, ray_idx[order][1:] != ray_idx[order][:-1]]]
        nearest[ray_idx[first]] = dist[first]
        nearest_geom[ray_idx[first]] = geom_idx[first]
    issues = []
    for r in np.flatnonzero(np.isfinite(nearest)):
        dist = float(nearest[r])
        sec = sections[owners[r]]
        st = stats[sec.id]
        st.spacing_samples += 1
        st.spacing_values.append(dist)
        if dist < settings.spacing_min_m or dist > settings.spacing_max_m:
            kind = "narrow" if dist < settings.spacing_min_m else "wide"
            if kind == "narrow":
                st.spacing_narrow += 1
                partner = sections[int(nearest_geom[r])].id
                st.narrow_partners[partner] = st.narrow_partners.get(partner, 0) + 1
            else:
                st.spacing_wide += 1
            issues.append(Issue("spacing", block_id, (sec.id,), Point(origins[r]), value=dist, detail=kind))
    return issues
