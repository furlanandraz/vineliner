"""Block polygons with a vertex at every row end (PROJECT.md §13).

The output keeps each input polygon's shape. Every point where a final row
meets the block boundary is inserted as a vertex, at exactly the coordinate
the row ends on, so rows and polygons snap together for editing.
"""

from __future__ import annotations

import shapely
from shapely.geometry import MultiPolygon, Point, Polygon

_EPS = 1e-9


def insert_vertices(polygon, points, tol: float = 1e-6):
    """Return polygon with each point within tol of a ring inserted into that ring."""
    parts = []
    for poly in shapely.get_parts(polygon):
        rings = [poly.exterior, *poly.interiors]
        assigned: list[list] = [[] for _ in rings]
        for p in points:
            pt = Point(p)
            dists = [ring.distance(pt) for ring in rings]
            k = min(range(len(rings)), key=dists.__getitem__)
            if dists[k] <= tol:
                assigned[k].append(tuple(p))
        new_rings = [_insert(ring, pts) for ring, pts in zip(rings, assigned)]
        parts.append(Polygon(new_rings[0], new_rings[1:]))
    if polygon.geom_type == "MultiPolygon":
        return MultiPolygon(parts)
    return parts[0]


def _insert(ring, points):
    coords = list(ring.coords)  # closed: first == last
    if not points:
        return coords
    cum = [0.0]
    for a, b in zip(coords[:-1], coords[1:]):
        cum.append(cum[-1] + ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
    existing = set(coords)
    positions = sorted({(ring.project(Point(p)), p) for p in points if p not in existing})
    out, j = [], 0
    for i in range(len(coords) - 1):
        out.append(coords[i])
        while j < len(positions) and positions[j][0] < cum[i + 1] - _EPS:
            if positions[j][0] > cum[i] + _EPS:
                out.append(positions[j][1])
            j += 1
    out.append(coords[-1])
    return out
