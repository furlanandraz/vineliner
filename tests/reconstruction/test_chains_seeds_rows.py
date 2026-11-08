"""Seeds from a selection, chains from seeds, and final row sections (PROJECT.md §10)."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from tools.geo import Grid
from tools.reconstruction import Endpoint, Row, Seed, build_chains, finalise_rows, link_seeds, rows_from_chains, seeds_from_selection
from tools.search import SearchSpace, Selection


def seed(i, x, y, theta=0.0, score=0.8):
    return Seed(i, 0, i, 0, theta, 1.5, 0, score, 0.7, 1.0, x, y, 2.0)


def two_rows():
    return [seed(i, x, y) for i, (x, y) in enumerate((x, y) for y in (0.0, 1.5) for x in range(0, 20, 2))]


# -- linking -------------------------------------------------------------------------


def test_seeds_on_neighbouring_rows_are_not_linked():
    labels = link_seeds(two_rows(), angle_deg=10, lateral_frac=0.25, along_m=4)
    assert len(set(labels)) == 2
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1


def test_link_rules_angle_and_along_gap():
    seeds = [seed(0, 0, 0), seed(1, 2, 0, theta=30.0), seed(2, 9, 0)]
    labels = link_seeds(seeds, angle_deg=10, lateral_frac=0.25, along_m=4)
    assert len(set(labels)) == 3  # 30 deg apart, and 9 m apart along the row


def test_no_seeds():
    assert link_seeds([], 10, 0.25, 4).size == 0


# -- chains --------------------------------------------------------------------------


def test_chain_runs_along_its_seeds_and_extends_half_a_line():
    seeds = [seed(i, x, 0.0) for i, x in enumerate(range(0, 20, 2))]
    (chain,) = build_chains(seeds, np.zeros(len(seeds), int), vertex_spacing_m=1.0)
    xs = chain.vertices[:, 0]
    assert np.all(np.diff(xs) > 0)
    assert xs[0] == pytest.approx(-1.0) and xs[-1] == pytest.approx(19.0)  # +- L/2
    assert chain.theta_deg == pytest.approx(0.0, abs=1e-9) and chain.flags == []


def test_chain_smoothing_reduces_wobble():
    seeds = [seed(i, x, 0.1 if i % 2 else -0.1) for i, x in enumerate(range(0, 30, 2))]
    labels = np.zeros(len(seeds), int)
    rough = build_chains(seeds, labels, 1.0, smoothing_m=0.0)[0]
    smooth = build_chains(seeds, labels, 1.0, smoothing_m=4.0)[0]
    assert np.std(smooth.vertices[1:-1, 1]) < 0.5 * np.std(rough.vertices[1:-1, 1])


def test_two_rows_forced_into_one_chain_are_flagged():
    labels = link_seeds(two_rows(), angle_deg=10, lateral_frac=1.5, along_m=4)  # far too loose
    (chain,) = build_chains(two_rows(), labels, 1.0)
    assert "lateral_spread" in chain.flags


# -- seeds from a selection ----------------------------------------------------------


def test_seed_positions_and_filters():
    grid = Grid(0.2, 0, 1000.0, 0, -0.2, 2000.0, 100, 100)
    space = SearchSpace((0.0, 90.0), (1.5,), 0.2, 0.2, "nominal_d0", 1.5)
    sel = Selection(
        method="best",
        score=np.array([0.8, 0.3]),  # the second centre is not accepted
        theta=np.array([1, 1]),  # 90 deg: rows run north, n points west
        spacing=np.array([0, 0]),
        offset=np.array([[1, 0], [0, 0]]),  # one pixel east
        means=np.array([[0.9, 0.9, 0.9, 0.2, 0.9], [0.9] * 5]),  # line +1 has weak evidence
        coverage=np.array([[1, 0.3, 1, 1, 1], [1] * 5], float),  # line -1 is mostly outside
        consensus_deg=np.full(2, np.nan),
    )
    rows, cols = np.array([50, 60]), np.array([50, 60])
    seeds = seeds_from_selection(sel, rows, cols, np.array([7, 8]), grid, space, 3, (-2, -1, 0, 1, 2), (-1, 0, 1),
                                 2.0, accept_min=0.5, min_valid_coverage=0.5, min_band_mean=0.4)
    assert [s.line for s in seeds] == [0]  # -1 fails coverage, +1 fails evidence, centre 8 not accepted
    cx, cy = grid.array_to_world(50, 50)
    assert (seeds[0].x, seeds[0].y) == pytest.approx((cx + 0.2, cy))
    assert (seeds[0].centre_id, seeds[0].block_id, seeds[0].theta_deg) == (7, 3, 90.0)
    # Without filters, line +1 sits one spacing along n = (-1, 0), i.e. 1.5 m west.
    all_lines = seeds_from_selection(sel, rows, cols, np.array([7, 8]), grid, space, 3, (-2, -1, 0, 1, 2), (0, 1),
                                     2.0, accept_min=0.5, min_valid_coverage=0.0)
    plus = next(s for s in all_lines if s.line == 1)
    assert (plus.x, plus.y) == pytest.approx((cx + 0.2 - 1.5, cy))


# -- final sections --------------------------------------------------------------------


def grown(*points, start="boundary", end="boundary", support=None):
    r = Row(0, 0, [np.array(p, float) for p in points], 1.5, 0, [1, 2], [0])
    r.start, r.end = Endpoint(state=start), Endpoint(state=end)
    if support is not None:
        r.seg_support = list(support)
    return r


def test_row_is_clipped_and_ends_snap_onto_the_boundary():
    poly = box(0, 0, 20, 10)
    (s,) = finalise_rows([grown((-3, 5), (25, 5))], poly, boundary_tol_m=0.3)
    assert s.geometry.coords[0] == pytest.approx((0, 5)) and s.geometry.coords[-1] == pytest.approx((20, 5))
    assert (s.start_state, s.end_state) == ("boundary", "boundary")
    assert s.boundary_points() == [s.geometry.coords[0], s.geometry.coords[-1]]


def test_end_just_short_of_the_boundary_is_extended_onto_it():
    (s,) = finalise_rows([grown((0, 5), (19.8, 5), end="unsupported")], box(0, 0, 20, 10), boundary_tol_m=0.3)
    assert s.end_state == "boundary" and s.geometry.coords[-1] == pytest.approx((20, 5))


def test_dangling_end_keeps_its_growth_state():
    (s,) = finalise_rows([grown((0, 5), (12, 5), end="unsupported")], box(0, 0, 20, 10), boundary_tol_m=0.3)
    assert s.end_state == "unsupported" and s.boundary_points() == [s.geometry.coords[0]]


def test_row_through_a_concave_polygon_splits_into_sections():
    u = Polygon([(0, 0), (30, 0), (30, 10), (20, 10), (20, 3), (10, 3), (10, 10), (0, 10)])
    sections = finalise_rows([grown((-1, 6), (31, 6))], u, boundary_tol_m=0.3)
    assert len(sections) == 2
    assert all(s.start_state == s.end_state == "boundary" for s in sections)


def test_bridged_parts_are_carried_to_the_section():
    r = grown((0, 5), (5, 5), (8, 5), (20, 5), support=[True, False, True])
    (s,) = finalise_rows([r], box(0, 0, 20, 10), boundary_tol_m=0.3)
    assert s.bridged_m == pytest.approx(3.0)
    assert s.n_seeds == 2


def test_rows_from_chains_pick_the_nearest_spacing_index():
    seeds = [seed(i, x, 0.0) for i, x in enumerate(range(0, 10, 2))]
    chains = build_chains(seeds, np.zeros(len(seeds), int), 1.0)
    (r,) = rows_from_chains(chains, (1.2, 1.4, 1.6))
    assert r.spacing_index == 1 and r.seed_ids == [s.id for s in seeds] and all(r.seg_support)
    assert isinstance(r.line(), LineString)
