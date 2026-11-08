"""Maximum direction change between row vertices ([smoothing])."""

import numpy as np
import pytest
from shapely.geometry import box

from tools.config import CONFIG_DIR, ConfigError, RunSettings, load_config, merge
from tools.config.settings import ReconstructionSettings
from tools.reconstruction import Grower, Row, clean_line, limit_turns, turn_angles


def test_turn_angles():
    assert turn_angles(np.array([(0, 0), (1, 0), (2, 0)])) == pytest.approx([0.0])
    assert turn_angles(np.array([(0, 0), (1, 0), (1, 1)])) == pytest.approx([90.0])
    assert turn_angles(np.array([(0, 0), (1, 0), (0, 0.001)])) == pytest.approx([180.0], abs=0.1)


def test_a_line_that_doubles_back_is_straightened():
    zig = [(0, 0), (5, 0), (4.7, 0.05), (10, 0.05)]  # steps 0.3 m back, as after a join
    pts, _ = limit_turns(zig, 20.0)
    assert np.all(turn_angles(pts) <= 20.0)
    assert tuple(pts[0]) == (0, 0) and tuple(pts[-1]) == (10, 0.05)  # ends never move
    assert np.all(np.diff(pts[:, 0]) > 0)


def test_curves_within_the_limit_are_kept():
    radius = 5.0  # the tightest curve assumed in PROJECT.md §2
    a = np.arange(0, 1.2, 1.0 / radius)  # vertices 1 m apart: 11.5 deg per vertex
    curve = np.column_stack([radius * np.sin(a), radius * (1 - np.cos(a))])
    pts, _ = limit_turns(curve, 20.0)
    assert len(pts) == len(curve)


def test_jitter_above_the_limit_is_removed():
    jitter = [(x, 0.4 if x % 2 else 0.0) for x in range(11)]  # 0.4 m zigzag every metre: ~44 deg turns
    pts, _ = limit_turns(jitter, 20.0)
    assert np.all(turn_angles(pts) <= 20.0) and len(pts) < len(jitter)


def test_support_flags_stay_aligned():
    pts, sup = limit_turns([(0, 0), (1, 0), (2, 1), (3, 0), (4, 0)], 20.0, [True, False, True, True])
    assert len(sup) == len(pts) - 1
    assert False in sup  # merging a bridged segment keeps it bridged
    assert np.all(turn_angles(pts) <= 20.0)


def test_repeated_vertices_are_dropped():
    pts, sup = limit_turns([(0, 0), (1, 0), (1, 0), (2, 0)], 20.0, [True, True, False])
    assert len(pts) == 3 and sup == [True, False]


def test_two_point_lines_are_left_alone():
    pts, _ = limit_turns([(0, 0), (3, 4)], 1.0)
    assert len(pts) == 2


def test_clean_line_drops_slivers_next_to_the_ends():
    # A vertex 0.4 mm from the end: its tiny segment points anywhere and fakes a sharp turn.
    line = [(0, 0), (5, 0), (10, 0), (10.0003, 0.0003)]
    out = clean_line(line, 20.0)
    assert tuple(out[0]) == (0, 0) and tuple(out[-1]) == (10.0003, 0.0003)  # the ends are kept
    assert len(out) == 3 and np.all(turn_angles(out) <= 20.0)
    start = clean_line([(0, 0), (0.002, 0.004), (5, 0), (10, 0)])
    assert len(start) == 3 and tuple(start[0]) == (0, 0)


def test_join_does_not_step_back():
    s = ReconstructionSettings(
        growth_step_m=1.0, min_curvature_radius_m=5.0, link_angle_deg=10.0, link_lateral_frac=0.25,
        link_along_m=4.0, chain_smoothing_m=0.0, growth_min_score=0.5, max_gap_m=3.0, bridge_tangent_m=10.0,
        join_distance_frac=0.35, join_angle_deg=20.0, max_row_length_m=1000.0,
    )
    a = Row(0, 0, [np.array(p, float) for p in [(10, 5), (15, 5)]], 1.5, 0, [], [])
    b = Row(1, 0, [np.array(p, float) for p in [(14.7, 5.05), (25, 5.05)]], 1.5, 0, [], [])  # starts 0.3 m behind a's tip
    (r,) = Grower([a, b], box(0, 0, 50, 10), lambda *args: 0.9, s, 2.0).run()
    x = np.array([p[0] for p in r.points])
    steps = np.diff(x)
    assert np.all(steps >= -1e-9) or np.all(steps <= 1e-9)  # monotone: no step back at the join
    assert len(r.seg_support) == len(r.points) - 1


def test_smoothing_settings():
    s = RunSettings.from_config(load_config(CONFIG_DIR / "centroid-pretest.toml")).smoothing
    assert (s.enabled, s.max_turn_deg) == (True, 20.0)
    with pytest.raises(ConfigError, match="smoothing"):
        RunSettings.from_config(merge(load_config(CONFIG_DIR / "centroid-pretest.toml"), {"smoothing": {"max_turn_deg": 0.0}}))
