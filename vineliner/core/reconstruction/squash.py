"""Squash detected hanging pieces into estimated rows (PROJECT.md §10, §17 item 30).

After the missing-row estimate, hanging pieces that were removed often lie on
an estimated row: the image showed the row, only broken. Each such piece is
squashed into that row:

1. A piece belongs to the estimated row whose buffer (merge_buffer_frac x the
   block's row width on each side) holds at least merge_min_inside_frac of the
   piece's length, and whose direction differs by at most merge_max_angle_deg.
   If several qualify, the one holding the largest share wins. With a buffer of
   half a row width, a piece that drifted is still caught, while a neighbouring
   row, a full width away, is not.
2. Each estimated row with pieces is rebuilt along its own course: where a
   piece covers it (the piece's extent projected onto the row), the row follows
   the piece; elsewhere it keeps the estimate. Where pieces overlap, the one
   with better image evidence (mean template score) wins.
3. The rebuilt row keeps the estimate's two boundary ends, is cleaned and
   smoothed, and must still meet the boundary at exactly its two ends;
   otherwise the plain estimate is kept. detected_m records how much of it now
   comes from the image.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, Point

from ..geo import axial, axial_difference
from .rows import RowSection
from .smoothing import clean_line

MIN_COVER_M = 0.5  # a piece covering less of the row than this is ignored


def _direction(line: LineString) -> float:
    c = np.asarray(line.coords)
    v = np.diff(c, axis=0)
    lengths = np.hypot(v[:, 0], v[:, 1])
    t2 = 2 * np.arctan2(v[:, 1], v[:, 0])  # angle doubling: directions are axial
    return float(axial(np.degrees(np.arctan2((lengths * np.sin(t2)).sum(), (lengths * np.cos(t2)).sum())) / 2))


def assign_pieces(estimated: list[RowSection], pieces: list[RowSection], width: float, buffer_frac: float,
                  min_inside_frac: float, max_angle_deg: float) -> dict[int, list[RowSection]]:
    """Estimated row id -> the pieces that belong to it (step 1)."""
    if not estimated or not pieces:
        return {}
    buffers = [e.geometry.buffer(buffer_frac * width, cap_style="flat") for e in estimated]
    directions = [_direction(e.geometry) for e in estimated]
    out: dict[int, list[RowSection]] = {}
    for p in pieces:
        if p.geometry.length == 0:
            continue
        p_dir = _direction(p.geometry)
        best, best_inside = None, -1.0
        for e, buf, e_dir in zip(estimated, buffers, directions):
            inside = p.geometry.intersection(buf).length / p.geometry.length
            if inside >= min_inside_frac and axial_difference(p_dir, e_dir) <= max_angle_deg and inside > best_inside:
                best, best_inside = e, inside
        if best is not None:
            out.setdefault(best.id, []).append(p)
    return out


def rebuild(estimate: RowSection, pieces: list[RowSection], evidence: dict[int, float | None], polygon, tol: float,
            max_turn_deg: float | None, meets_boundary_twice) -> tuple[RowSection, list[int]]:
    """The estimated row rebuilt from its pieces (steps 2-3); returns it and the ids of the pieces used."""
    E = estimate.geometry
    L = E.length
    intervals = []  # (from, to) along E, piece
    for p in pieces:
        s = [E.project(Point(c)) for c in p.geometry.coords]
        a, b = max(0.0, min(s)), min(L, max(s))
        if b - a >= MIN_COVER_M:
            intervals.append((a, b, p))
    if not intervals:
        return estimate, []
    # Elementary stretches along E; each goes to the covering piece with the best evidence, or stays estimated.
    cuts = sorted({0.0, L, *(a for a, _, _ in intervals), *(b for _, b, _ in intervals)})
    stretches = []
    for s0, s1 in zip(cuts[:-1], cuts[1:]):
        if s1 - s0 <= 1e-9:
            continue
        mid = 0.5 * (s0 + s1)
        covering = [p for a, b, p in intervals if a <= mid <= b]
        owner = max(covering, key=lambda p: evidence.get(p.id) if evidence.get(p.id) is not None else -1.0) if covering else None
        if stretches and stretches[-1][2] is owner:
            stretches[-1] = (stretches[-1][0], s1, owner)
        else:
            stretches.append((s0, s1, owner))
    # The row keeps the estimate's boundary ends: a zero-length estimate stretch at an end a piece covers.
    if stretches[0][2] is not None:
        stretches.insert(0, (0.0, 0.0, None))
    if stretches[-1][2] is not None:
        stretches.append((L, L, None))

    def at(owner, s):
        """The point of a stretch's line (a piece, or the estimate) level with arc length s on E."""
        p = E.interpolate(s)
        return p if owner is None else owner.geometry.interpolate(owner.geometry.project(p))

    # Where the row switches line, the two lines are offset sideways at the same point along the row.
    # A straight ramp of length offset / tan(angle) keeps the switch within the turn limit.
    ramp_angle = np.radians(0.75 * (max_turn_deg if max_turn_deg is not None else 20.0))
    ramps = []
    for (_, s, a), (_, _, b) in zip(stretches[:-1], stretches[1:]):
        ramps.append(max(0.5, at(a, s).distance(at(b, s)) / np.tan(ramp_angle)))

    coords: list[tuple[float, float]] = []
    detected = 0.0
    used = []
    for k, (s0, s1, owner) in enumerate(stretches):
        length = s1 - s0
        before = ramps[k - 1] if k > 0 else 0.0
        after = ramps[k] if k < len(ramps) else 0.0
        # A zero-length end stretch gives the whole ramp to its neighbour; otherwise it is split half and half.
        before = before if k > 0 and stretches[k - 1][1] - stretches[k - 1][0] <= 1e-9 else before / 2
        after = after if k < len(ramps) and stretches[k + 1][1] - stretches[k + 1][0] <= 1e-9 else after / 2
        a, b = s0 + min(before, 0.45 * length), s1 - min(after, 0.45 * length)
        if length <= 1e-9:
            coords.append(tuple(at(owner, s0).coords[0]))
            continue
        if owner is None:
            part = shapely.ops.substring(E, a, b)
        else:
            g = owner.geometry
            t0, t1 = g.project(E.interpolate(a)), g.project(E.interpolate(b))
            part = shapely.ops.substring(g, min(t0, t1), max(t0, t1))
            if t1 < t0:
                part = part.reverse()
            detected += b - a
            if owner.id not in used:
                used.append(owner.id)
        coords.extend(list(part.coords) if part.geom_type == "LineString" else [tuple(part.coords[0])])
    line = LineString(clean_line(coords, max_turn_deg))
    if not line.is_valid or line.length == 0 or not meets_boundary_twice(line, polygon, tol):
        return estimate, []
    merged = replace(
        estimate,
        geometry=line,
        detected_m=round(detected, 3),
        flags=estimate.flags + [f"{len(used)} detected piece(s) squashed in: {','.join(map(str, used))}"],
    )
    return merged, used


def squash_pieces(estimated: list[RowSection], pieces: list[RowSection], width: float, buffer_frac: float,
                  min_inside_frac: float, max_angle_deg: float, evidence: dict[int, float | None], polygon, tol: float,
                  max_turn_deg: float | None, meets_boundary_twice) -> tuple[list[RowSection], set[int]]:
    """Rebuild every estimated row that has pieces; return the rows and the ids of all pieces used."""
    owned = assign_pieces(estimated, pieces, width, buffer_frac, min_inside_frac, max_angle_deg)
    out, used_all = [], set()
    for e in estimated:
        if e.id in owned:
            e2, used = rebuild(e, owned[e.id], evidence, polygon, tol, max_turn_deg, meets_boundary_twice)
            out.append(e2)
            used_all.update(used)
        else:
            out.append(replace(e, detected_m=0.0))
    return out, used_all
