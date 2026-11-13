"""World <-> array conversion and the angle convention (PROJECT.md §6, §6.1)."""

import numpy as np
import pytest

from vineliner.core.geo import Grid, across, along, axial_difference, axial_mean

GRID = Grid(0.2, 0.0, 449350.5, 0.0, -0.2, 6426545.4, 2000, 2000)  # the Bordeaux grid


def test_pixel_centre_convention():
    x, y = GRID.array_to_world(0, 0)
    assert (x, y) == pytest.approx((449350.6, 6426545.3))
    row, col = GRID.world_to_array(x, y)
    assert (row, col) == pytest.approx((0.0, 0.0), abs=1e-6)  # float error at 6.4e6 m coordinates


def test_round_trip():
    rng = np.random.default_rng(0)
    r, c = rng.uniform(0, 2000, 50), rng.uniform(0, 2000, 50)
    x, y = GRID.array_to_world(r, c)
    r2, c2 = GRID.world_to_array(x, y)
    assert np.allclose(r, r2) and np.allclose(c, c2)


def test_rows_run_south():
    # Moving north by one pixel decreases the row index by one.
    drow, dcol = GRID.world_vector_to_array(0.0, 0.2)
    assert (drow, dcol) == pytest.approx((-1.0, 0.0))
    dx, dy = GRID.array_vector_to_world(1.0, 1.0)
    assert (dx, dy) == pytest.approx((0.2, -0.2))


def test_nearest_pixel():
    x, y = GRID.array_to_world(10, 20)
    assert tuple(map(int, GRID.nearest_pixel(x + 0.09, y - 0.09))) == (10, 20)
    assert tuple(map(int, GRID.nearest_pixel(x + 0.11, y))) == (10, 21)


def test_window_keeps_world_positions():
    sub = GRID.window(100, 200, 50, 60)
    assert sub.array_to_world(0, 0) == pytest.approx(GRID.array_to_world(100, 200))
    assert sub.shape == (50, 60)


def test_north_up_check():
    GRID.require_north_up()
    with pytest.raises(ValueError):
        Grid(0.2, 0.01, 0, 0, -0.2, 0, 10, 10).require_north_up()
    with pytest.raises(ValueError):
        Grid(0.2, 0, 0, 0, -0.25, 0, 10, 10).require_square_pixels()


def test_angle_convention():
    assert along(0) == pytest.approx([1, 0])
    assert along(90) == pytest.approx([0, 1], abs=1e-12)
    assert across(0) == pytest.approx([0, 1])  # u rotated +90 deg
    assert across(30) == pytest.approx([-0.5, np.sqrt(3) / 2])


def test_axial_helpers():
    assert axial_difference(179, 1) == pytest.approx(2)
    assert axial_difference(10, 100) == pytest.approx(90)
    assert axial_mean([179, 1]) in (pytest.approx(0, abs=1e-9), pytest.approx(180, abs=1e-9))
    assert axial_mean([40, 42], weights=[1, 1]) == pytest.approx(41)
