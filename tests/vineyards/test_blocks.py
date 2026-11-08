"""Block polygons, pixel masks and windows."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from tools.geo import Grid
from tools.vineyards import block_mask, classify_blocks, make_blocks, pixel_window

GRID = Grid(1.0, 0, 0.0, 0, -1.0, 10.0, 10, 10)  # 1 m pixels; pixel (r, c) has its centre at (c + 0.5, 9.5 - r)


def test_mask_uses_pixel_centres_boundary_included():
    (b,) = make_blocks([0], [box(1.5, 1.5, 3.5, 3.5)], "file")
    mask = block_mask(b, GRID)
    assert mask.sum() == 9  # centres at 1.5, 2.5, 3.5 on both axes; the boundary counts as inside
    rows, cols = np.nonzero(mask)
    assert set(cols) == {1, 2, 3} and set(rows) == {6, 7, 8}


def test_window_covers_the_block_plus_padding():
    (b,) = make_blocks([0], [box(2.0, 3.0, 5.0, 6.0)], "file")
    row0, col0, h, w = pixel_window(b, GRID, pad_px=2)
    sub = GRID.window(row0, col0, h, w)
    xmin, ymin, xmax, ymax = sub.bounds()
    assert xmin <= 2.0 - 2 and xmax >= 5.0 + 2 and ymin <= 3.0 - 2 and ymax >= 6.0 + 2


def test_blocks_keep_ids_attributes_and_are_made_valid():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # self-intersecting
    assert not bowtie.is_valid
    blocks = make_blocks([5, 6], [bowtie, box(0, 0, 1, 1)], "file", records=[{"name": "x"}, {"name": "y"}])
    assert [b.id for b in blocks] == [5, 6]
    assert blocks[0].polygon.is_valid and blocks[0].area_m2 == pytest.approx(2.0)
    assert blocks[1].attributes == {"name": "y"} and blocks[1].source == "file"


def test_automatic_classification_is_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="source"):
        classify_blocks()
