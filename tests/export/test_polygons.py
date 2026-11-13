"""Block polygons get a vertex at every row end, at exactly the row's coordinate."""

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from vineliner.core.export import insert_vertices
from vineliner.core.reconstruction.rows import snap_to_boundary


def test_vertices_inserted_in_ring_order_and_shape_unchanged():
    poly = box(0, 0, 10, 5)
    pts = [(3.0, 0.0), (7.0, 0.0), (10.0, 2.5), (0.0, 4.0)]
    out = insert_vertices(poly, pts)
    coords = list(out.exterior.coords)
    for p in pts:
        assert p in coords
    assert out.is_valid
    assert abs(out.area - poly.area) < 1e-9
    assert out.symmetric_difference(poly).area < 1e-9


def test_existing_vertices_and_far_points_are_not_duplicated():
    poly = box(0, 0, 10, 5)
    out = insert_vertices(poly, [(0.0, 0.0), (5.0, 2.5)])  # a corner and an interior point
    assert len(out.exterior.coords) == len(poly.exterior.coords)


def test_multipolygon_and_holes():
    outer = Polygon([(0, 0), (20, 0), (20, 10), (0, 10)], [[(5, 3), (8, 3), (8, 6), (5, 6)]])
    multi = MultiPolygon([outer, box(30, 0, 40, 10)])
    out = insert_vertices(multi, [(6.5, 3.0), (35.0, 10.0)])
    assert out.geom_type == "MultiPolygon"
    assert (6.5, 3.0) in list(out.geoms[0].interiors[0].coords)
    assert (35.0, 10.0) in list(out.geoms[1].exterior.coords)


def test_snapping_extends_a_row_that_stops_short_of_the_boundary():
    poly = box(0, 0, 10, 10)
    end = (9.8, 5.0)  # 0.2 m short of the east edge, row running east
    snapped = snap_to_boundary(end, inward=np.array([-1.0, 0.0]), polygon=poly, reach_m=0.9)
    assert snapped == (10.0, 5.0)
    out = insert_vertices(poly, [snapped])
    assert snapped in list(out.exterior.coords)
    assert LineString([(1.0, 5.0), snapped]).touches(out) or Point(snapped).distance(out.boundary) < 1e-12
