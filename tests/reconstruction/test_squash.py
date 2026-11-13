"""Squashing detected hanging pieces into estimated rows."""

import numpy as np
import pytest
from shapely.geometry import LineString, box

from vineliner.core.config import ConfigError, RunSettings, plugin_config
from vineliner.core.reconstruction import RowSection, assign_pieces, squash_pieces, turn_angles
from vineliner.core.validation import mark_estimated, meets_boundary_twice

FIELD = box(0, 0, 30, 10)
WIDTH = 1.5


def section(i, coords, estimated=False):
    flags = ["estimated", "between rows 1 and 2 (1 of 1)"] if estimated else []
    return RowSection(i, i, 0, LineString(coords), "boundary", "boundary", WIDTH, 0, [], 0, flags=flags,
                      detected_m=0.0 if estimated else None)


ESTIMATE = section(100, [(0, 4.0), (30, 4.0)], estimated=True)


def squash(pieces, evidence=None, **kw):
    args = dict(width=WIDTH, buffer_frac=0.5, min_inside_frac=0.8, max_angle_deg=10.0)
    args.update(kw)
    return squash_pieces([ESTIMATE], pieces, evidence=evidence or {}, polygon=FIELD, tol=0.1, max_turn_deg=20.0,
                         meets_boundary_twice=meets_boundary_twice, **args)


def test_assignment_catches_drift_but_not_the_neighbouring_row():
    drift = section(1, [(2, 4.3), (12, 4.3)])  # 0.3 m off: same row
    neighbour = section(2, [(2, 5.5), (12, 5.5)])  # a full row width away
    angled = section(3, [(2, 3.0), (12, 3.0 + 10 * np.tan(np.radians(30)))])
    half_out = section(4, [(2, 4.3), (6, 4.3), (10, 6.0)])  # less than 80 % inside
    owned = assign_pieces([ESTIMATE], [drift, neighbour, angled, half_out], WIDTH, 0.5, 0.8, 10.0)
    assert [p.id for p in owned[100]] == [1]


def test_rebuilt_row_follows_the_pieces_and_keeps_its_boundary_ends():
    pieces = [section(1, [(2, 4.3), (12, 4.3)]), section(2, [(18, 3.9), (28, 3.9)])]
    (row,), used = squash(pieces)
    assert used == {1, 2}
    c = np.asarray(row.geometry.coords)
    assert tuple(c[0]) == (0, 4.0) and tuple(c[-1]) == (30, 4.0)
    assert row.geometry.distance(LineString([(4, 4.3), (10, 4.3)])) < 1e-6  # follows piece 1 in its middle
    assert row.geometry.distance(LineString([(20, 3.9), (26, 3.9)])) < 1e-6
    # 20 m of pieces, less the ramps onto and off them (offset / tan(15 deg), half from each side).
    assert 17.0 < row.detected_m < 20.0
    assert meets_boundary_twice(row.geometry, FIELD, 0.1) and turn_angles(c).max() <= 20.0 + 1e-6
    assert any("squashed in" in f for f in row.flags)


def test_overlapping_pieces_give_the_overlap_to_the_better_evidence():
    weak = section(1, [(2, 4.3), (14, 4.3)])
    strong = section(2, [(10, 3.8), (25, 3.8)])
    (row,), _ = squash([weak, strong], evidence={1: 0.4, 2: 0.8})
    assert row.geometry.distance(LineString([(11, 3.8), (13, 3.8)])) < 1e-6  # the overlap follows the strong piece
    assert 19.0 < row.detected_m < 23.0  # 2..25 covered by one piece or the other, less the ramps


def test_rows_without_pieces_stay_estimates():
    (row,), used = squash([section(1, [(2, 5.5), (12, 5.5)])])  # only a neighbour-row piece
    assert used == set() and row.detected_m == 0.0 and row.geometry.equals(ESTIMATE.geometry)


def test_merged_rows_are_labelled_by_their_detected_share():
    sec = ESTIMATE
    mostly = mark_estimated({"estimated": 0, "review": 0, "conf": "high", "issues": "", "det_pct": 80.0}, sec)
    assert (mostly["conf"], mostly["review"]) == ("merged", 0) and "80% from detected pieces" in mostly["issues"]
    little = mark_estimated({"estimated": 0, "review": 0, "conf": "high", "issues": "", "det_pct": 20.0}, sec)
    assert (little["conf"], little["review"]) == ("merged", 1)
    pure = mark_estimated({"estimated": 0, "review": 0, "conf": "high", "issues": "", "det_pct": 0.0}, sec)
    assert (pure["conf"], pure["review"]) == ("estimated", 1)


def test_squash_settings():
    m = RunSettings.from_config(plugin_config(1.2, 1.6)).missing_row_estimate
    assert (m.merge_pieces, m.merge_buffer_frac, m.merge_min_inside_frac, m.merge_max_angle_deg) == (True, 0.5, 0.8, 10.0)
    bad = plugin_config(1.2, 1.6, {"missing_row_estimate": {"merge_buffer_frac": 1.5}})
    with pytest.raises(ConfigError, match="merge_buffer_frac"):
        RunSettings.from_config(bad)
