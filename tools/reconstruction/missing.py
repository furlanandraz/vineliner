"""Missing-row estimate (PROJECT.md §10, §17 item 27).

With [missing_row_estimate] enabled, rows the image did not show are added
between neighbouring detected rows that stand too far apart. It runs on
complete rows only, so hanging rows are deleted first ([export]
drop_hanging_rows is forced on, see config/settings.py).

1. Row width d, per block: the median distance between neighbouring rows,
   measured along the row normals every sample_step_m. A distance that spans
   several row intervals (rows missing in between) is first divided by the
   number of intervals, judged with the search's row spacing, so missing rows
   do not inflate the median. The raw median is reported too.
2. Neighbours: for each row and side, the row hit most often by normal rays
   reaching (max_missing_rows + 2.5) * d, so a gap one row wider than allowed is
   still seen and reported. A pair's gap is the median distance.
3. A pair further apart than d * (1 + gap_tolerance_frac) has
   n = round(gap / d) - 1 missing rows. They are added when
   1 <= n <= max_missing_rows and the resulting spacing gap / (n + 1) lies
   within whole_row_tolerance_frac of d (the whole-number check). Otherwise the
   gap is reported, not filled. Rounding never leaves the new spacing more than
   25 % off, so whole_row_tolerance_frac >= 0.25 accepts every gap.
4. Each added row is interpolated between the pair: stations every
   station_factor * d along the first row, the matching nearest point on the
   second, and the new vertex at fraction i / (n + 1) between them, so added
   rows follow the neighbours' curvature. The line is extended to the block
   boundary and clipped by it; an added row must meet the boundary at exactly
   its two ends, like the rows it sits between.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point

from .rows import RowSection, snap_to_boundary
from .smoothing import clean_line, limit_turns

MIN_SAMPLES = 3  # normal rays that must agree before two rows count as neighbours


@dataclass
class RowGap:
    a: int  # section ids, a < b
    b: int
    gap_m: float
    samples: int
    missing: int
    filled: bool
    reason: str
    x: float  # a point between the two rows, for the issues layer
    y: float


def row_width(sections: list[RowSection], stats: dict, search_spacing_m: float) -> tuple[float | None, dict]:
    """Median row width from the measured neighbour distances (step 1)."""
    values = np.array([v for s in sections for v in stats[s.id].spacing_values], dtype=float)
    info = {"search_spacing_m": round(search_spacing_m, 3), "samples": int(values.size)}
    if values.size == 0:
        return None, info | {"row_width_m": None, "raw_median_m": None}
    intervals = np.maximum(1.0, np.round(values / search_spacing_m))
    width = float(np.median(values / intervals))
    return width, info | {"row_width_m": round(width, 3), "raw_median_m": round(float(np.median(values)), 3)}


def find_gaps(
    sections: list[RowSection],
    width: float,
    gap_tolerance_frac: float,
    whole_row_tolerance_frac: float,
    max_missing_rows: int,
    sample_step_m: float,
) -> list[RowGap]:
    """Neighbouring pairs further apart than allowed (steps 2-3)."""
    if len(sections) < 2:
        return []
    reach = (max_missing_rows + 2.5) * width
    origins, ends, owners, sides = [], [], [], []
    for idx, s in enumerate(sections):
        line, length = s.geometry, s.geometry.length
        for d in np.arange(sample_step_m / 2, length, sample_step_m):
            p = np.array(line.interpolate(d).coords[0])
            t = np.array(line.interpolate(min(d + 0.5, length)).coords[0]) - np.array(line.interpolate(max(d - 0.5, 0.0)).coords[0])
            norm = np.hypot(*t)
            if norm == 0:
                continue
            n = np.array([-t[1], t[0]]) / norm
            for sign in (1, -1):
                origins.append(p)
                ends.append(p + sign * reach * n)
                owners.append(idx)
                sides.append(sign)
    if not origins:
        return []
    origins = np.array(origins)
    owners, sides = np.array(owners), np.array(sides)
    rays = shapely.linestrings(np.stack([origins, np.array(ends)], axis=1))
    tree = shapely.STRtree([s.geometry for s in sections])
    ray_idx, geom_idx = tree.query(rays, predicate="intersects")
    keep = geom_idx != owners[ray_idx]
    ray_idx, geom_idx = ray_idx[keep], geom_idx[keep]
    if ray_idx.size == 0:
        return []
    dist = shapely.distance(shapely.points(origins[ray_idx]), shapely.intersection(rays[ray_idx], tree.geometries.take(geom_idx)))
    order = np.lexsort((dist, ray_idx))
    first = order[np.r_[True, ray_idx[order][1:] != ray_idx[order][:-1]]]  # nearest hit per ray
    hit_ray, hit_geom, hit_dist = ray_idx[first], geom_idx[first], dist[first]

    # Neighbour on each side of each row: the row its rays hit most often.
    pair_dists: dict[tuple[int, int], list[float]] = {}
    for owner in np.unique(owners[hit_ray]):
        for sign in (1, -1):
            sel = (owners[hit_ray] == owner) & (sides[hit_ray] == sign)
            if not sel.any():
                continue
            partners, counts = np.unique(hit_geom[sel], return_counts=True)
            partner = int(partners[np.argmax(counts)])
            key = tuple(sorted((int(owner), partner)))
            pair_dists.setdefault(key, []).extend(hit_dist[sel & (hit_geom == partner)].tolist())

    gaps = []
    for (ia, ib), dists in sorted(pair_dists.items()):
        if len(dists) < MIN_SAMPLES:
            continue
        gap = float(np.median(dists))
        if gap <= width * (1 + gap_tolerance_frac):
            continue
        missing = int(np.floor(gap / width + 0.5)) - 1
        a, b = sections[ia], sections[ib]
        mid = a.geometry.interpolate(0.5, normalized=True)
        other = b.geometry.interpolate(b.geometry.project(mid))
        x, y = (mid.x + other.x) / 2, (mid.y + other.y) / 2
        if missing < 1:
            reason = f"gap {gap:.2f} m is wider than allowed but holds no whole row"
        elif missing > max_missing_rows:
            reason = f"gap {gap:.2f} m would hold {missing} rows, more than max_missing_rows = {max_missing_rows}"
        elif abs(gap / (missing + 1) - width) > whole_row_tolerance_frac * width + 1e-9:
            reason = f"gap {gap:.2f} m is not a whole number of {width:.2f} m rows"
        else:
            gaps.append(RowGap(min(a.id, b.id), max(a.id, b.id), gap, len(dists), missing, True, f"{missing} row(s) fit", x, y))
            continue
        gaps.append(RowGap(min(a.id, b.id), max(a.id, b.id), gap, len(dists), max(missing, 0), False, reason, x, y))
    return gaps


def interpolate_rows(
    a: LineString, b: LineString, missing: int, width: float, station_factor: float, polygon, tol: float,
    max_turn_deg: float | None = None,
) -> list[LineString | None]:
    """The rows between a and b (step 4). None where an interpolated row would not span the block.

    max_turn_deg: the [smoothing] limit, applied to the interpolated vertices as to detected rows.
    """
    step = station_factor * width
    stations = np.unique(np.concatenate([np.arange(0.0, a.length, step), [a.length]]))
    pa, pb = [], []
    for s in stations:
        p = np.array(a.interpolate(s).coords[0])
        t = b.project(Point(p))
        q = np.array(b.interpolate(t).coords[0])
        if t <= 1e-9 or t >= b.length - 1e-9:
            # Clamped to an end of b: keep it only if it still lies roughly across from a.
            tangent = np.array(a.interpolate(min(s + 0.5, a.length)).coords[0]) - np.array(a.interpolate(max(s - 0.5, 0.0)).coords[0])
            across = q - p
            if np.hypot(*across) > 0 and abs(np.dot(across, tangent)) > 0.5 * np.hypot(*across) * np.hypot(*tangent):
                continue
        pa.append(p)
        pb.append(q)
    if len(pa) < 2:
        return [None] * missing
    pa, pb = np.array(pa), np.array(pb)
    reach = 3.0 * (missing + 1) * width
    out = []
    for i in range(1, missing + 1):
        coords = pa + (i / (missing + 1)) * (pb - pa)
        if max_turn_deg is not None:
            coords, _ = limit_turns(coords, max_turn_deg)
        out.append(_span_block(coords, polygon, reach, tol, max_turn_deg))
    return out


def _span_block(coords: np.ndarray, polygon, reach: float, tol: float, max_turn_deg: float | None = None) -> LineString | None:
    """Extend a line along its end directions, clip it to the block, snap its ends onto the boundary."""
    def unit(v):
        n = np.hypot(*v)
        return v / n if n > 0 else v

    start = coords[0] + unit(coords[0] - coords[1]) * reach
    end = coords[-1] + unit(coords[-1] - coords[-2]) * reach
    line = LineString(np.vstack([start, coords, end]))
    middle = LineString(coords).interpolate(0.5, normalized=True)
    parts = [g for g in shapely.get_parts(line.intersection(polygon)) if g.geom_type == "LineString" and g.length > 0]
    if not parts:
        return None
    merged = shapely.line_merge(MultiLineString(parts))
    part = min(shapely.get_parts(merged), key=lambda g: g.distance(middle))
    c = [tuple(v) for v in part.coords]
    c[0] = snap_to_boundary(c[0], None, polygon, tol)
    c[-1] = snap_to_boundary(c[-1], None, polygon, tol)
    return LineString(clean_line(c, max_turn_deg))


def fill_gaps(
    gaps: list[RowGap],
    sections: list[RowSection],
    width: float,
    station_factor: float,
    polygon,
    tol: float,
    spacings,
    first_id: int,
    meets_boundary_twice,
    max_turn_deg: float | None = None,
) -> list[RowSection]:
    """RowSections for every row that fits into a fillable gap."""
    by_id = {s.id: s for s in sections}
    grid = np.asarray(spacings)
    added: list[RowSection] = []
    for gap in gaps:
        if not gap.filled:
            continue
        lines = interpolate_rows(by_id[gap.a].geometry, by_id[gap.b].geometry, gap.missing, width, station_factor, polygon, tol, max_turn_deg)
        complete = [g for g in lines if g is not None and meets_boundary_twice(g, polygon, tol)]
        if len(complete) < len(lines):
            gap.filled = False
            gap.reason = "an interpolated row would not span the block from boundary to boundary"
            continue
        for k, line in enumerate(complete, start=1):
            added.append(RowSection(
                id=first_id + len(added),
                row_id=-1,
                block_id=by_id[gap.a].block_id,
                geometry=line,
                start_state="boundary",
                end_state="boundary",
                spacing_m=width,
                spacing_index=int(np.argmin(np.abs(grid - width))),
                seed_ids=[],
                n_joins=0,
                flags=["estimated", f"between rows {gap.a} and {gap.b} ({k} of {gap.missing})"],
            ))
    return added


def gap_point(gap: RowGap) -> Point:
    return Point(gap.x, gap.y)
