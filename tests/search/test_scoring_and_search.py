"""Scoring rules (PROJECT.md §8) and the batched local search against direct sums."""

import numpy as np
import pytest

from synthetic import SYNTHETIC_PIXEL, SYNTHETIC_SPACING, SYNTHETIC_THETA, stripe_evidence, synthetic_grid
from tools.config.settings import ScoringSettings
from tools.search import LocalSearch, SearchSpace, base_score, score_hypothesis
from tools.template import KernelBank, Template

W = (0.05, 0.20, 0.50, 0.20, 0.05)
ALL = np.ones(5, bool)


def test_agreed_equal_totals():
    centre_only = np.array([0, 0, 1, 0, 0], float)
    neighbours_only = np.array([1, 1, 0, 1, 1], float)
    for policy in ("zero", "renormalise"):
        assert base_score(centre_only, ALL, W, 2, policy) == pytest.approx(0.50)
        assert base_score(neighbours_only, ALL, W, 2, policy) == pytest.approx(0.50)


def test_unavailable_bands():
    means = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    avail = np.array([False, True, True, True, True])
    assert base_score(means, avail, W, 2, "zero") == pytest.approx(0.95)
    assert base_score(means, avail, W, 2, "renormalise") == pytest.approx(1.0)
    no_centre = np.array([True, True, False, True, True])
    assert np.isnan(base_score(means, no_centre, W, 2, "zero"))


@pytest.fixture(scope="module")
def setup():
    grid = synthetic_grid()
    V = stripe_evidence(grid)
    M = np.ones_like(V)
    M[:, :40] = 0  # an invalid strip, so coverage < 1 near it
    template = Template((-2, -1, 0, 1, 2), W, 2.0, 0.6)
    thetas = tuple(float(t) for t in range(20, 41))
    spacings = (1.40, 1.45, 1.50, 1.55, 1.60)
    bank = KernelBank(template, thetas, spacings, SYNTHETIC_PIXEL, SYNTHETIC_PIXEL, "exact")
    space = SearchSpace(thetas, spacings, SYNTHETIC_PIXEL, SYNTHETIC_PIXEL, "nominal_d0", 1.5)
    scoring = ScoringSettings(0.5, "renormalise_valid", 0.5, "renormalise")
    search = LocalSearch(bank, space, scoring, W, 2)
    return grid, V * M, M, bank, space, scoring, search


def test_search_equals_direct_sums(setup):
    grid, mv, m, bank, space, scoring, search = setup
    rows, cols = np.array([150, 120, 60]), np.array([150, 200, 55])
    captured = {}
    res = search.run(mv, m, rows, cols, batch_size=2, surface_sink=lambda b, sl, s, best, tidx: captured.setdefault(b, (sl, s)))
    rng = np.random.default_rng(1)
    for _ in range(25):
        i = int(rng.integers(len(rows)))
        j, t = int(rng.integers(len(space.spacings))), int(rng.integers(len(space.thetas)))
        ox, oy = space.offsets(j)
        iy, ix = int(rng.integers(len(oy))), int(rng.integers(len(ox)))
        batch = i // 2
        sl, surfaces = captured[batch]
        got = surfaces[j][i - sl.start, t, iy, ix]
        want, _, _ = score_hypothesis(bank.get(t, j), mv, m, rows[i] - oy[iy], cols[i] + ox[ix], W, 2, scoring)
        if np.isnan(want):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(want, abs=1e-5)
    assert res.best_score.shape == (3,)


def test_search_recovers_known_rows(setup):
    grid, mv, m, bank, space, scoring, search = setup
    rows, cols = np.array([150, 100, 200]), np.array([150, 180, 220])
    res = search.run(mv, m, rows, cols)
    thetas = np.asarray(space.thetas)[res.best_theta]
    spacings = np.asarray(space.spacings)[res.best_spacing]
    assert np.all(np.abs(thetas - SYNTHETIC_THETA) <= 1)
    assert np.all(np.abs(spacings - SYNTHETIC_SPACING) <= 0.05)
    # 0.5 m stripes read through 0.6 m bands: the best band mean is about 0.5 / 0.6.
    assert np.all(res.best_score > 0.8)
