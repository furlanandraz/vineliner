"""Row growth with a mock scorer standing in for the image (PROJECT.md §10)."""

import math

import numpy as np
import pytest
from shapely.geometry import box

from tools.config.settings import ReconstructionSettings
from tools.geo import axial_difference
from tools.reconstruction import Endpoint, Grower, Row

FIELD = box(0, 0, 50, 10)  # rows run east along y = 5


def settings(**overrides) -> ReconstructionSettings:
    values = dict(
        growth_step_m=1.0, min_curvature_radius_m=5.0, link_angle_deg=10.0, link_lateral_frac=0.25,
        link_along_m=4.0, chain_smoothing_m=0.0, growth_min_score=0.5, max_gap_m=3.0, bridge_tangent_m=10.0,
        join_distance_frac=0.35, join_angle_deg=20.0, max_row_length_m=1000.0,
    )
    values.update(overrides)
    return ReconstructionSettings(**values)


def row(row_id, *points, fixed=False):
    r = Row(id=row_id, block_id=0, points=[np.array(p, float) for p in points], spacing_m=1.5, spacing_index=0, seed_ids=[], chain_ids=[])
    if fixed:  # a row that does not grow
        r.start, r.end = Endpoint(state="boundary"), Endpoint(state="boundary")
    return r


def everywhere(score=0.9):
    return lambda x, y, theta, j: score


def grow(rows, scorer, polygon=FIELD, **kw):
    g = Grower(rows, polygon, scorer, settings(**kw), line_length_m=2.0)
    return g.run(), g


def xs(r):
    return [p[0] for p in r.points]


def test_max_turn_follows_the_minimum_curvature_radius():
    assert settings().max_turn_deg == pytest.approx(math.degrees(1 / 5))
    assert settings(min_curvature_radius_m=20.0).max_turn_deg == pytest.approx(math.degrees(1 / 20))


def test_supported_row_grows_to_the_boundary_at_both_ends():
    (r,), _ = grow([row(0, (20, 5), (22, 5))], everywhere())
    assert (r.start.state, r.end.state) == ("boundary", "boundary")
    assert min(xs(r)) == pytest.approx(0) and max(xs(r)) == pytest.approx(50)
    assert all(abs(p[1] - 5) < 1e-9 for p in r.points)  # equal scores everywhere: it goes straight
    assert all(r.seg_support)


def test_long_unsupported_stretch_stops_and_is_trimmed():
    scorer = lambda x, y, t, j: 0.9 if x < 30 else 0.1
    (r,), _ = grow([row(0, (20, 5), (22, 5))], scorer, max_gap_m=3.0)
    assert r.end.state == "unsupported"
    assert max(xs(r)) == pytest.approx(30)  # the unsupported tail beyond the last support is removed
    assert r.start.state == "boundary"


def test_short_gap_is_bridged_and_recorded():
    scorer = lambda x, y, t, j: 0.1 if 30 < x < 32 else 0.9
    (r,), _ = grow([row(0, (20, 5), (22, 5))], scorer, max_gap_m=3.0)
    assert r.end.state == "boundary" and max(xs(r)) == pytest.approx(50)
    bridged = sum(s.length for s in r.bridged_segments())
    assert bridged == pytest.approx(2.0)


def test_two_pieces_of_one_row_join():
    rows, g = grow([row(0, (10, 5), (12, 5)), row(1, (30, 5), (32, 5))], everywhere())
    assert len(rows) == 1 and len(g.joins) == 1
    (r,) = rows
    assert {r.id, *r.joined_rows} == {0, 1}  # whichever row reached the other absorbs it
    assert min(xs(r)) == pytest.approx(0) and max(xs(r)) == pytest.approx(50)
    steps = np.diff(xs(r))
    assert np.all(steps >= -1e-9) or np.all(steps <= 1e-9)  # one monotone line after the merge, either way round


def test_offset_pieces_do_not_join():
    # 0.6 m apart across the row: more than link_lateral_frac * d = 0.375 m.
    rows, g = grow([row(0, (10, 5), (12, 5)), row(1, (30, 5.6), (32, 5.6))], everywhere())
    assert g.joins == []
    assert len(rows) == 2


def test_crossing_another_row_is_a_conflict():
    crossing = row(1, (30, 1), (30, 9), fixed=True)
    rows, _ = grow([row(0, (20, 5), (22, 5)), crossing], everywhere())
    r = next(x for x in rows if x.id == 0)
    assert r.end.state == "conflict"
    assert max(xs(r)) < 30
    assert r.start.state == "boundary"


def test_growth_limit():
    (r,), _ = grow([row(0, (20, 5), (22, 5))], everywhere(), max_row_length_m=5.0)
    assert (r.start.state, r.end.state) == ("limit", "limit")
    assert max(xs(r)) - min(xs(r)) == pytest.approx(2 + 2 * 5)


def test_direction_follows_the_evidence_within_the_turn_limit():
    # The mock evidence prefers rows at 10 deg; the seed points due east.
    scorer = lambda x, y, theta, j: 0.9 if axial_difference(theta, 10.0) <= 1.0 else 0.3
    (r,), _ = grow([row(0, (5, 20), (7, 20))], scorer, polygon=box(0, 0, 60, 40))
    last = r.points[-1] - r.points[-2]
    assert axial_difference(math.degrees(math.atan2(last[1], last[0])), 10.0) <= 1.0
    assert r.end.state == "boundary"
