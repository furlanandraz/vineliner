"""Final row sections: clip, split, snap and label endpoints (PROJECT.md §10, step 8).

Each grown row is clipped to its block polygon. A row that leaves and
re-enters the polygon becomes several sections, each with its own two
endpoints. An endpoint within boundary_tol_m of the polygon boundary is
"boundary" and is snapped exactly onto the boundary: extended along the row
direction when it stops just short, otherwise projected. Those snapped points
become vertices of the output polygons, so rows and polygons share them
exactly. Any other endpoint keeps the state growth gave it (a dangling end).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point

from .chains import Chain
from .growth import Row
from .smoothing import clean_line


@dataclass
class RowSection:
    id: int
    row_id: int
    block_id: int
    geometry: LineString
    start_state: str
    end_state: str
    spacing_m: float
    spacing_index: int
    seed_ids: list[int]
    n_joins: int
    bridged: object = None  # line geometry of the bridged (unsupported) parts, or None
    flags: list[str] = field(default_factory=list)

    @property
    def length_m(self) -> float:
        return float(self.geometry.length)

    @property
    def n_seeds(self) -> int:
        return len(self.seed_ids)

    @property
    def bridged_m(self) -> float:
        return float(self.bridged.length) if self.bridged is not None else 0.0

    def boundary_points(self) -> list[tuple[float, float]]:
        """Endpoints that sit on the block boundary."""
        out = []
        if self.start_state == "boundary":
            out.append(self.geometry.coords[0])
        if self.end_state == "boundary":
            out.append(self.geometry.coords[-1])
        return out


def rows_from_chains(chains: list[Chain], spacings: tuple[float, ...], first_id: int = 0) -> list[Row]:
    rows = []
    grid = np.asarray(spacings)
    for chain in chains:
        rows.append(Row(
            id=first_id + len(rows),
            block_id=chain.block_id,
            points=[np.asarray(p, dtype=float) for p in chain.vertices],
            spacing_m=chain.spacing_m,
            spacing_index=int(np.argmin(np.abs(grid - chain.spacing_m))),
            seed_ids=list(chain.seed_ids),
            chain_ids=[chain.id],
            flags=list(chain.flags),
        ))
    return rows


def finalise_rows(rows: list[Row], polygon, boundary_tol_m: float, first_id: int = 0, max_turn_deg: float | None = None) -> list[RowSection]:
    """max_turn_deg: the [smoothing] limit, applied again to the final clipped and snapped geometry."""
    boundary = polygon.boundary
    sections: list[RowSection] = []
    for row in rows:
        coords = np.array(row.points)
        keep = np.ones(len(coords), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(coords, axis=0)) > 1e-9, axis=1)
        coords = coords[keep]
        if len(coords) < 2:
            continue
        line = LineString(coords)
        if line.length == 0 or not line.intersects(polygon):
            continue
        start, end = Point(coords[0]), Point(coords[-1])
        bridged_all = row.bridged_segments()
        bridged_zone = MultiLineString(bridged_all).buffer(1e-6) if bridged_all else None
        # The clip can mix lines with isolated touch points; keep the lines only.
        pieces = [g for g in shapely.get_parts(line.intersection(polygon)) if g.geom_type == "LineString" and g.length > 0]
        if not pieces:
            continue
        for part in shapely.get_parts(shapely.line_merge(MultiLineString(pieces))):
            if part.geom_type != "LineString" or part.length == 0:
                continue

            def state(p: Point) -> str:
                if boundary.distance(p) <= boundary_tol_m:
                    return "boundary"
                if p.distance(start) < 1e-6:
                    return row.start.state
                if p.distance(end) < 1e-6:
                    return row.end.state
                return "boundary"  # created by clipping, so it lies on the boundary

            s_state = state(Point(part.coords[0]))
            e_state = state(Point(part.coords[-1]))
            part = _snap_ends(part, polygon, s_state == "boundary", e_state == "boundary", boundary_tol_m)
            part = LineString(clean_line(part.coords, max_turn_deg))
            bridged = None
            if bridged_zone is not None:
                b = part.intersection(bridged_zone)
                lines = [g for g in shapely.get_parts(b) if g.geom_type == "LineString" and g.length > 1e-6]
                bridged = shapely.line_merge(MultiLineString(lines)) if lines else None
            sections.append(RowSection(
                id=first_id + len(sections),
                row_id=row.id,
                block_id=row.block_id,
                geometry=part,
                start_state=s_state,
                end_state=e_state,
                spacing_m=row.spacing_m,
                spacing_index=row.spacing_index,
                seed_ids=list(row.seed_ids),
                n_joins=len(row.joined_rows),
                bridged=bridged,
                flags=list(row.flags),
            ))
    return sections


def snap_to_boundary(point, inward, polygon, reach_m: float) -> tuple[float, float]:
    """A point exactly on the nearest polygon ring.

    If the point stops short of the boundary, the row is extended along
    `inward`'s opposite (the outward row direction) up to reach_m; without a
    hit there, the point is projected onto the nearest ring.
    """
    p = np.asarray(point, dtype=float)
    rings = [ring for poly in shapely.get_parts(polygon) for ring in [poly.exterior, *poly.interiors]]
    ring = min(rings, key=lambda r: r.distance(Point(p)))
    if ring.distance(Point(p)) > 1e-9 and inward is not None:
        outward = -np.asarray(inward, dtype=float)
        norm = np.hypot(*outward)
        if norm > 0:
            ray = LineString([p, p + outward / norm * reach_m])
            hits = [np.array(g.coords[0]) for g in shapely.get_parts(ray.intersection(ring)) if not g.is_empty]
            if hits:
                p = min(hits, key=lambda h: np.hypot(*(h - p)))
    snapped = ring.interpolate(ring.project(Point(p)))
    return (snapped.x, snapped.y)


def _snap_ends(line: LineString, polygon, snap_start: bool, snap_end: bool, tol: float) -> LineString:
    coords = [tuple(c) for c in line.coords]
    reach = 3 * tol
    if snap_start:
        inward = np.subtract(coords[1], coords[0])
        coords[0] = snap_to_boundary(coords[0], inward, polygon, reach)
    if snap_end:
        inward = np.subtract(coords[-2], coords[-1])
        coords[-1] = snap_to_boundary(coords[-1], inward, polygon, reach)
    return LineString(coords)
