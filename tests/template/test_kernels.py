"""Template rasterisation (PROJECT.md §6.1)."""

import numpy as np
import pytest
import shapely

from vineliner.core.template import KernelBank, Template, rasterise_band

T = Template((-2, -1, 0, 1, 2), (0.05, 0.20, 0.50, 0.20, 0.05), line_length_m=2.0, band_width_m=0.2)
P = 0.2


def dense(kernel, size=40):
    out = np.zeros((2 * size + 1, 2 * size + 1))
    out[size + kernel.drow, size + kernel.dcol] = kernel.weight
    return out


@pytest.mark.parametrize("theta", [0, 17, 37, 45, 90, 133])
@pytest.mark.parametrize("k", [0, 1, -2])
def test_exact_total_equals_band_area(theta, k):
    kern = rasterise_band(T, theta, 1.45, k, P, P, "exact")
    assert kern.total == pytest.approx(2.0 * 0.2 / P**2, rel=1e-5)
    assert kern.weight.max() <= 1.0 and kern.weight.min() > 0


@pytest.mark.parametrize("theta", [0, 23, 45, 71])
def test_sampled_coverage_converges_to_exact(theta):
    exact = dense(rasterise_band(T, theta, 1.45, 1, P, P, "exact"))
    sampled = dense(rasterise_band(T, theta, 1.45, 1, P, P, "sampled", subpixel_samples=32))
    assert np.abs(exact - sampled).max() < 0.08  # an edge pixel can differ by ~2/s


def test_exact_matches_independent_shapely_clip():
    """Rebuild one kernel pixel by pixel with Shapely, independently of the vectorised code."""
    theta, d, k = 37.0, 1.45, -1
    kern = rasterise_band(T, theta, d, k, P, P, "exact")
    band = shapely.Polygon(T.band_corners(theta, d, k))
    for r, c, w in zip(kern.drow, kern.dcol, kern.weight):
        pixel = shapely.box((c - 0.5) * P, -(r + 0.5) * P, (c + 0.5) * P, -(r - 0.5) * P)
        assert w == pytest.approx(pixel.intersection(band).area / P**2, abs=1e-6)


def test_band_k_sits_k_spacings_across():
    # theta = 0: rows run east, n points north, so band +1 is d north = d/p rows up (negative rows).
    kern = rasterise_band(T, 0.0, 1.4, 1, P, P, "exact")
    centre_row = np.average(kern.drow, weights=kern.weight)
    assert centre_row == pytest.approx(-1.4 / P, abs=1e-6)


def test_binary_methods_are_binary():
    for method in ("centre", "all_touched"):
        kern = rasterise_band(T, 33, 1.45, 0, P, P, method)
        assert set(np.unique(kern.weight)) == {1.0}


def test_bank_shares_line_zero_across_spacings():
    bank = KernelBank(T, (0.0, 45.0), (1.4, 1.6), P, P, "exact")
    assert bank.get(1, 0)[2] is bank.get(1, 1)[2]
    assert bank.get(1, 0)[3] is not bank.get(1, 1)[3]
    assert bank.totals(0).shape == (2, 5)
