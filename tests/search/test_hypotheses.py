"""Hypothesis grids (PROJECT.md §7) and the count table there."""

import numpy as np
import pytest
from shapely.geometry import box

from vineliner.core.geo import Grid
from vineliner.core.search import SearchSpace, centre_outward_order, search_centres, translation_offsets


@pytest.mark.parametrize(
    ("box_m", "pixel_m", "expected"),
    [
        (2.0, 0.20, list(range(-5, 5))),  # the §7 example: -1.00 .. +0.80 m
        (2.2, 0.20, list(range(-5, 6))),
        (1.4, 0.20, list(range(-3, 4))),
        (2.2, 0.25, list(range(-4, 5))),
        (2.2, 0.50, list(range(-2, 3))),
        (2.2, 0.10, list(range(-11, 11))),
    ],
)
def test_translation_offsets(box_m, pixel_m, expected):
    assert translation_offsets(box_m, pixel_m).tolist() == expected


SPACINGS = tuple(round(1.80 + 0.02 * j, 6) for j in range(21))
THETAS = tuple(float(t) for t in range(180))


@pytest.mark.parametrize(
    ("pixel_m", "box", "nominal", "expected"),
    [
        (0.20, "nominal_d0", 2.0, 378_000),
        (0.20, "nominal_d0", 2.2, 457_380),
        (0.20, "tested_d", 2.0, 381_600),
        (0.25, "nominal_d0", 2.2, 306_180),
        (0.50, "nominal_d0", 2.2, 94_500),
        (0.10, "nominal_d0", 2.2, 1_829_520),
    ],
)
def test_hypotheses_per_centre_match_project_md_table(pixel_m, box, nominal, expected):
    assert SearchSpace(THETAS, SPACINGS, pixel_m, pixel_m, box, nominal).hypotheses_per_centre() == expected


def test_centre_outward_order_starts_at_centre_and_covers_everything():
    ox = oy = np.arange(-3, 4)
    order = centre_outward_order(ox, oy, 0.2, 0.2)
    iy, ix = np.divmod(order, len(ox))
    assert (ox[ix[0]], oy[iy[0]]) == (0, 0)
    d = np.hypot(ox[ix], oy[iy])
    assert np.all(np.diff(d) >= -1e-12)
    assert sorted(order.tolist()) == list(range(49))


def test_search_centres_are_pixel_centres_inside_the_polygon():
    grid = Grid(0.2, 0, 1000.0, 0, -0.2, 2000.0, 200, 200)
    poly = box(1005.0, 1965.0, 1030.0, 1995.0)
    valid = np.ones(grid.shape, bool)
    rows, cols = search_centres(poly, grid, valid, 2.0)
    x, y = grid.array_to_world(rows, cols)
    assert rows.size > 0
    assert np.all((x >= 1005) & (x <= 1030) & (y >= 1965) & (y <= 1995))
    # snapped lattice: neighbouring centres about 2 m apart
    assert len(np.unique(np.round(x, 6))) == pytest.approx(13, abs=1)
