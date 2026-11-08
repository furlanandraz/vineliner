"""Row numbering per block (PROJECT.md §10, §17 item 31).

1. Block direction theta_b: the length-weighted axial mean of the rows'
   directions; n_b is the across direction (theta_b + 90 deg).
2. Transects: lines along n_b through every row's midpoint (so every row is on
   one), plus one every transect_interval_m along the block.
3. On each transect the rows it crosses are sorted along n_b. Between two
   consecutive crossings, the number of row intervals is
   k = max(1, round(gap * cos(phi) / d)), with phi the rows' local deviation
   from theta_b and d the block's row width. A missing row therefore shows up
   as k = 2: its number is skipped.
4. Each consecutive pair (a, b) on any transect gives a relation
   no(b) - no(a) = k. Per pair the most common k wins. A maximum spanning tree,
   weighted by how many transects support each relation, links all rows, and
   numbers are assigned along it.
5. Every relation is checked against the numbers. A row in a relation they
   break (for example a row that drifts from one lane into the next), or in a
   separate group not linked to the rest, gets row_no_ok = 0.
6. Numbers start at 1 at the row furthest against n_b and increase along n_b,
   i.e. to the left when facing along theta_b (theta between east and west via
   north): northward for rows running east-west, westward for rows running
   north-south. Two sections of one row in a concave block share a number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from ..geo import across, along, axial_difference, axial_mean


@dataclass
class Numbering:
    numbers: dict[int, int] = field(default_factory=dict)  # section id -> row number
    ok: dict[int, bool] = field(default_factory=dict)  # section id -> consistent with its neighbours
    direction_deg: float | None = None
    missing: list[int] = field(default_factory=list)  # numbers no row has: missing rows
    transects: int = 0

    def describe(self) -> dict:
        return {
            "direction_deg": None if self.direction_deg is None else round(self.direction_deg, 1),
            "rows_numbered": len(self.numbers),
            "first_row_no": min(self.numbers.values()) if self.numbers else None,
            "last_row_no": max(self.numbers.values()) if self.numbers else None,
            "missing_row_nos": self.missing,
            "rows_uncertain": sum(not v for v in self.ok.values()),
            "transects": self.transects,
        }


def _direction(line: LineString) -> tuple[float, float]:
    """(axial direction in degrees, length) of a polyline."""
    c = np.asarray(line.coords)
    v = np.diff(c, axis=0)
    lengths = np.hypot(v[:, 0], v[:, 1])
    ok = lengths > 0
    if not ok.any():
        return 0.0, 0.0
    return axial_mean(np.degrees(np.arctan2(v[ok, 1], v[ok, 0])), weights=lengths[ok]), float(lengths.sum())


def _local_direction(line: LineString, p: Point) -> float:
    s = line.project(p)
    a = np.array(line.interpolate(max(s - 0.5, 0.0)).coords[0])
    b = np.array(line.interpolate(min(s + 0.5, line.length)).coords[0])
    return float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180.0)


def number_rows(sections, width: float, transect_interval_m: float) -> Numbering:
    out = Numbering()
    sections = [s for s in sections if s.geometry.length > 0]
    if not sections:
        return out
    dirs = [_direction(s.geometry) for s in sections]
    theta = axial_mean([d for d, _ in dirs], weights=[w for _, w in dirs])
    out.direction_deg = theta
    u, n = along(theta), across(theta)
    geoms = [s.geometry for s in sections]
    tree = shapely.STRtree(geoms)
    coords = np.vstack([np.asarray(g.coords) for g in geoms])
    centre = coords.mean(axis=0)
    reach = float(np.hypot(*(coords.max(axis=0) - coords.min(axis=0)))) + 1.0

    origins = [np.array(s.geometry.interpolate(0.5, normalized=True).coords[0]) for s in sections]
    a_min, a_max = float((coords @ u).min()), float((coords @ u).max())
    for a in np.arange(a_min + transect_interval_m / 2, a_max, transect_interval_m):
        origins.append(centre + (a - centre @ u) * u)
    out.transects = len(origins)

    relations: dict[tuple[int, int], list[int]] = defaultdict(list)  # (lo, hi) -> observed no(hi) - no(lo)
    for o in origins:
        transect = LineString([o - reach * n, o + reach * n])
        crossings = []
        for k in tree.query(transect, predicate="intersects"):
            hit = geoms[int(k)].intersection(transect)
            pts = [g for g in shapely.get_parts(hit) if g.geom_type == "Point"]
            if not pts:
                continue
            pos = float(np.median([np.dot(np.array(p.coords[0]) - o, n) for p in pts]))
            mid = pts[len(pts) // 2]
            crossings.append((pos, int(k), axial_difference(_local_direction(geoms[int(k)], mid), theta)))
        crossings.sort()
        for (p0, i0, f0), (p1, i1, f1) in zip(crossings[:-1], crossings[1:]):
            if i0 == i1:
                continue
            across_gap = (p1 - p0) * np.cos(np.radians(0.5 * (f0 + f1)))  # measured along the rows' normal
            k = max(1, int(np.floor(across_gap / width + 0.5)))
            lo, hi = (i0, i1) if i0 < i1 else (i1, i0)
            relations[(lo, hi)].append(k if lo == i0 else -k)

    # Most common difference per pair, then a maximum spanning tree by support.
    consensus = {}
    for key, values in relations.items():
        counts = Counter(values)
        best = max(counts.items(), key=lambda kv: (kv[1], -abs(kv[0])))
        consensus[key] = (best[0], best[1], len(set(values)) > 1)
    parent = list(range(len(sections)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree_edges = defaultdict(list)
    for (lo, hi), (diff, support, _) in sorted(consensus.items(), key=lambda kv: (-kv[1][1], abs(kv[1][0]))):
        ra, rb = find(lo), find(hi)
        if ra != rb:
            parent[ra] = rb
            tree_edges[lo].append((hi, diff))
            tree_edges[hi].append((lo, -diff))

    number = {}
    components = []
    for start in range(len(sections)):
        if start in number:
            continue
        number[start] = 0
        stack, members = [start], [start]
        while stack:
            x = stack.pop()
            for y, diff in tree_edges[x]:
                if y not in number:
                    number[y] = number[x] + diff
                    stack.append(y)
                    members.append(y)
        components.append(members)

    ok = {i: True for i in range(len(sections))}
    for (lo, hi), (diff, _, disagree) in consensus.items():
        if number[hi] - number[lo] != diff or disagree:
            ok[lo] = ok[hi] = False

    # Place smaller groups next to the main one by their position across the block; they are uncertain.
    components.sort(key=len, reverse=True)
    main = components[0]
    pos = {i: float(np.dot(origins[i] - centre, n)) for i in range(len(sections))}
    ref = min(main, key=lambda i: pos[i])
    for comp in components[1:]:
        anchor = min(comp, key=lambda i: pos[i])
        shift = number[ref] + int(np.floor((pos[anchor] - pos[ref]) / width + 0.5)) - number[anchor]
        for i in comp:
            number[i] += shift
            ok[i] = False

    low = min(number.values())
    out.numbers = {sections[i].id: number[i] - low + 1 for i in number}
    out.ok = {sections[i].id: ok[i] for i in ok}
    used = set(out.numbers.values())
    out.missing = [k for k in range(1, max(used) + 1) if k not in used]
    return out
