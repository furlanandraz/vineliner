"""Peak selection and persistence diagnostics, on hand-made search results."""


import numpy as np
import pytest

from vineliner.core.search import (
    RIDGE_FIELDS,
    CentreResults,
    competing_orientations,
    ridge_statistics,
    select_best,
    select_orientation_consensus,
)

THETAS = np.arange(180.0)
K = 5


def results(winners, profiles=None):
    """Mock CentreResults: centre i's raw winner is rotation winners[i]."""
    n = len(winners)
    prof = np.full((n, 180), 0.3, np.float32) if profiles is None else np.asarray(profiles, np.float32)
    for i, t in enumerate(winners):
        prof[i, t] = np.fmax(prof[i, t], 0.8)  # fmax: a NaN profile entry still gets the winner's 0.8
    theta_means = np.repeat(np.arange(180, dtype=np.float32)[None, :, None] / 180, n, axis=0).repeat(K, axis=2)
    return CentreResults(
        rows=np.arange(n), cols=np.arange(n),
        best_score=np.nanmax(prof, axis=1), best_spacing=np.zeros(n, int), best_theta=np.asarray(winners),
        best_offset=np.zeros((n, 2), int), best_rank=np.zeros(n, int),
        best_means=np.full((n, K), 0.7, np.float32), best_coverage=np.ones((n, K), np.float32),
        theta_profile=prof, spacing_profile=np.full((n, 1), 0.8, np.float32),
        ridge=np.full((n, len(RIDGE_FIELDS)), np.nan, np.float32), competing=np.full((n, 3), np.nan, np.float32),
        theta_spacing=np.zeros((n, 180), np.int16), theta_offset=np.zeros((n, 180, 2), np.int16),
        theta_means=theta_means, theta_coverage=np.ones((n, 180, K), np.float32),
    )


# -- selection -----------------------------------------------------------------------


def test_best_selection_copies_the_raw_winner():
    res = results([40, 41, 25])
    sel = select_best(res)
    assert sel.method == "best" and list(sel.theta) == [40, 41, 25]
    assert np.isnan(sel.consensus_deg).all()
    sel.theta[0] = 1
    assert res.best_theta[0] == 40  # a copy, not a view


def test_orientation_consensus_reorients_an_outlier():
    # A 3 x 3 patch of centres 2 m apart: eight agree on 40 deg, the middle one won at 25 deg.
    winners = [40] * 9
    winners[4] = 25
    profiles = np.full((9, 180), 0.3)
    profiles[4, 40] = 0.78  # nearly as good at 40 deg
    res = results(winners, profiles)
    xs, ys = np.meshgrid([0.0, 2.0, 4.0], [0.0, 2.0, 4.0])
    sel = select_orientation_consensus(res, xs.ravel(), ys.ravel(), THETAS, 0.5, radius_m=6.0, window_deg=5.0)
    assert sel.method == "orientation_consensus"
    assert sel.theta[4] == 40 and sel.score[4] == pytest.approx(0.78)
    assert sel.means[4] == pytest.approx(np.full(K, 40 / 180))  # taken from the 40 deg hypothesis
    assert np.all(sel.theta[[0, 1, 2, 3, 5, 6, 7, 8]] == 40)
    assert sel.consensus_deg[4] == pytest.approx(40, abs=2)


def test_consensus_without_a_hypothesis_in_the_window_is_not_accepted():
    profiles = np.full((2, 180), np.nan)
    res = results([40, 100], profiles)  # each centre only scored its own winner
    assert list(res.best_score) == [pytest.approx(0.8), pytest.approx(0.8)]  # both raw winners are accepted
    sel = select_orientation_consensus(res, np.array([0.0, 1.0]), np.array([0.0, 0.0]), THETAS, 0.5, 6.0, 5.0)
    assert sel.consensus_deg == pytest.approx([70.0, 70.0])
    # The consensus of 40 and 100 deg (axial mean 70) has neither within 5 deg.
    assert np.isnan(sel.score).all()
    assert not sel.accepted(0.5).any()


# -- persistence ---------------------------------------------------------------------


def test_ridge_along_the_row_scores_higher_than_across():
    ox = oy = np.arange(-3, 4)
    gy, gx = np.meshgrid(oy, ox, indexing="ij")
    surface = 1.0 - 0.2 * np.abs(gy)  # rows run east (theta 0): high along oy = 0
    stats = dict(zip(RIDGE_FIELDS, ridge_statistics(surface, ox, oy, 0.2, 0.2, 0.0, iy=3, ix=3)))
    assert stats["along_mean"] == pytest.approx(1.0) and stats["along_n"] == 7
    assert stats["across_mean"] < stats["along_mean"]
    assert stats["box_max"] == pytest.approx(1.0) and stats["box_min"] == pytest.approx(0.4)


def test_competing_orientations():
    profile = np.full(180, 0.3)
    profile[40], profile[100] = 0.9, 0.7
    assert competing_orientations(profile, THETAS, 40, 0.5) == (1, 100.0, pytest.approx(0.7))
    profile[100] = 0.45  # below accept_min
    assert competing_orientations(profile, THETAS, 40, 0.5)[0] == 0
    near = np.full(180, 0.3)
    near[40], near[45] = 0.9, 0.8  # a peak 5 deg away is not a competitor
    assert competing_orientations(near, THETAS, 40, 0.5)[0] == 0
    wrap = np.full(180, 0.3)
    wrap[40], wrap[179], wrap[0] = 0.9, 0.7, 0.65  # 179 and 0 are neighbours
    assert competing_orientations(wrap, THETAS, 40, 0.5)[1] == 179.0

