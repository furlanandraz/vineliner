"""Centroid pre-test, its neighbouring-box backup, and the narrowed search."""

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from synthetic import SYNTHETIC_PIXEL, SYNTHETIC_SPACING, SYNTHETIC_THETA, stripe_evidence, synthetic_grid
from tools.config.settings import ScoringSettings
from tools.search import LocalSearch, SearchSpace, ring_offsets, run_pretest
from tools.template import KernelBank, Template

W = (0.05, 0.20, 0.50, 0.20, 0.05)


@pytest.fixture(scope="module")
def setup():
    grid = synthetic_grid()
    V = stripe_evidence(grid)
    M = np.ones_like(V)
    template = Template((-2, -1, 0, 1, 2), W, 2.0, 0.6)
    thetas = tuple(float(t) for t in range(0, 180, 2))  # every 2 deg keeps the test fast
    spacings = (1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70)
    bank = KernelBank(template, thetas, spacings, SYNTHETIC_PIXEL, SYNTHETIC_PIXEL, "exact")
    space = SearchSpace(thetas, spacings, SYNTHETIC_PIXEL, SYNTHETIC_PIXEL, "nominal_d0", 1.5)
    scoring = ScoringSettings(0.5, "renormalise_valid", 0.5, "renormalise")
    return grid, V * M, M, LocalSearch(bank, space, scoring, W, 2), space


def pretest(setup, polygon, mv=None, rings=3, theta_window=6.0, spacing_frac=0.05):
    grid, mv0, m, full, space = setup
    block = SimpleNamespace(id=1, polygon=polygon)
    return run_pretest(block, grid, np.ones(grid.shape, bool), full, mv0 if mv is None else mv, m, (0, 0),
                       0.5, theta_window, spacing_frac, box_spacing_m=2.0, neighbour_rings=rings)


def square(grid):
    return box(grid.c + 10, grid.f - 50, grid.c + 50, grid.f - 10)


def test_ring_offsets():
    assert ring_offsets(0) == [(0, 0)]
    for r in (1, 2, 3):
        cells = ring_offsets(r)
        assert len(cells) == 8 * r and all(max(abs(a), abs(b)) == r for a, b in cells)


def test_restricted_search_matches_the_full_search_on_its_subset(setup):
    grid, mv, m, full, space = setup
    rows, cols = np.array([150, 120, 180]), np.array([150, 170, 130])
    a = full.run(mv, m, rows, cols)
    # Fewer rotations, all spacings: the per-rotation best is unchanged where searched.
    theta_idx = np.arange(10, 21)
    b = full.restricted(theta_idx, None)
    assert b.hypotheses_per_centre == 11 * len(space.spacings) * 49
    rb = b.run(mv, m, rows, cols)
    assert np.allclose(a.theta_profile[:, theta_idx], rb.theta_profile[:, theta_idx])
    assert np.isnan(np.delete(rb.theta_profile, theta_idx, axis=1)).all()
    assert set(rb.best_theta) <= set(theta_idx)
    # Fewer spacings, all rotations: the per-spacing best is unchanged where searched.
    spacing_idx = np.array([2, 3, 4])
    rc = full.restricted(None, spacing_idx).run(mv, m, rows, cols)
    assert np.allclose(a.spacing_profile[:, spacing_idx], rc.spacing_profile[:, spacing_idx])
    assert np.isnan(np.delete(rc.spacing_profile, spacing_idx, axis=1)).all()
    assert set(rc.best_spacing) <= set(spacing_idx)


def test_centroid_box_finds_the_rows_and_narrows_the_search(setup):
    grid, _, _, _, space = setup
    pt = pretest(setup, square(grid))
    assert (pt.mode, pt.ring, pt.boxes_tested) == ("pretest", 0, 1)
    assert abs(pt.theta_deg - SYNTHETIC_THETA) <= 2
    assert abs(pt.spacing_m - SYNTHETIC_SPACING) <= 0.05
    thetas = np.asarray(space.thetas)[pt.theta_idx]
    assert len(thetas) == 7 and np.all(np.abs(thetas - pt.theta_deg) <= 6)
    spacings = np.asarray(space.spacings)[pt.spacing_idx]
    assert np.all(np.abs(spacings - pt.spacing_m) <= 0.05 * pt.spacing_m + 1e-9)


def test_weak_centroid_box_is_backed_up_by_a_neighbouring_box(setup):
    grid, mv, _, _, _ = setup
    poly = square(grid)
    c = poly.centroid
    r, k = (int(v) for v in grid.nearest_pixel(c.x, c.y))
    blank = mv.copy()
    blank[r - 8:r + 9, k - 8:k + 9] = 0  # no rows within 1.6 m of the centroid
    pt = pretest(setup, poly, mv=blank)
    assert pt.mode == "pretest" and pt.ring >= 1 and pt.boxes_tested > 1
    assert "ring" in pt.reason
    assert abs(pt.theta_deg - SYNTHETIC_THETA) <= 2
    assert np.hypot(pt.x - c.x, pt.y - c.y) <= 2.0 * pt.ring * np.sqrt(2) + 0.2


def test_no_accepted_box_falls_back_to_the_full_search(setup):
    grid, mv, _, _, _ = setup
    pt = pretest(setup, square(grid), mv=np.zeros_like(mv))
    assert pt.mode == "full" and "accept_min" in pt.reason and pt.boxes_tested == 1 + 8 + 16 + 24
    pt0 = pretest(setup, square(grid), mv=np.zeros_like(mv), rings=0)
    assert pt0.mode == "full" and pt0.boxes_tested == 1


def test_centroid_outside_the_polygon_falls_back_to_the_full_search(setup):
    grid = setup[0]
    x0, y0 = grid.c, grid.f
    # A U shape: its centroid lies in the notch, outside the polygon.
    u = Polygon([(x0 + 10, y0 - 50), (x0 + 50, y0 - 50), (x0 + 50, y0 - 10), (x0 + 40, y0 - 10),
                 (x0 + 40, y0 - 40), (x0 + 20, y0 - 40), (x0 + 20, y0 - 10), (x0 + 10, y0 - 10)])
    assert not u.contains(u.centroid)
    pt = pretest(setup, u)
    assert (pt.mode, pt.reason, pt.boxes_tested) == ("full", "centroid outside the polygon", 0)
