"""Which evidence components the selected layers give, and the checks on them (pipeline/inputs.py)."""

import numpy as np
import pytest
from shapely.geometry import box

from vineliner.core.geo import Grid
from vineliner.core.pipeline import InputError, Raster, assemble_inputs, ndvi_from
from vineliner.core.vineyards import Block

GRID = Grid(0.2, 0, 1000.0, 0, -0.2, 2000.0, 50, 40)
BLOCKS = [Block(1, box(1001, 1993, 1009, 1999), "test")]


def layer(name, value=100.0, grid=GRID):
    bands = {"rgb": ("red", "green", "blue"), "nir": ("nir", "red", "green"), "ndvi": ("ndvi",)}[name]
    data = np.full((len(bands), grid.height, grid.width), value, np.float32)
    if name == "nir":
        data[0], data[1] = 300.0, 100.0  # NIR 300, red 100 -> NDVI 0.5
    if name == "ndvi":
        data[0] = 0.2
    return Raster(data, grid, bands)


def test_rgb_alone_gives_the_rgb_component():
    inputs = assemble_inputs({"rgb": layer("rgb")}, BLOCKS)
    assert inputs.components == ("rgb",) and inputs.ndvi_source is None and inputs.pixel_m == 0.2


def test_rgb_with_nir_r_g_also_gives_nir_and_computed_ndvi():
    inputs = assemble_inputs({"rgb": layer("rgb"), "nir": layer("nir")}, BLOCKS)
    assert inputs.components == ("rgb", "nir", "ndvi") and inputs.ndvi_source == "computed from NIR-R-G"
    assert np.allclose(inputs.layers["ndvi"].data[0], 0.5)
    assert [r.band_names[0] for r in inputs.band_rasters] == ["red", "nir"]  # gray bands: RGB first


def test_ndvi_alone_and_a_supplied_ndvi_is_preferred_to_a_computed_one():
    assert assemble_inputs({"ndvi": layer("ndvi")}, BLOCKS).components == ("ndvi",)
    both = assemble_inputs({"nir": layer("nir"), "ndvi": layer("ndvi"), "rgb": None}, BLOCKS)
    assert both.components == ("nir", "ndvi") and both.ndvi_source == "supplied"
    assert np.allclose(both.layers["ndvi"].data[0], 0.2)


def test_no_data_in_any_band_of_any_layer_is_invalid():
    rgb, nir = layer("rgb"), layer("nir")
    rgb.data[2, 0, 0] = np.nan  # one band of one layer
    nir.data[0, 1, 1] = np.nan  # NIR missing -> computed NDVI missing too
    inputs = assemble_inputs({"rgb": rgb, "nir": nir}, BLOCKS)
    assert not inputs.valid[0, 0] and not inputs.valid[1, 1] and inputs.valid.sum() == GRID.width * GRID.height - 2
    assert inputs.masks[1].sum() > 0


def test_ndvi_from_handles_zero_and_missing_sums():
    nir = np.array([3.0, 0.0, np.nan, 1.0], np.float32)
    red = np.array([1.0, 0.0, 1.0, 1.0], np.float32)
    out = ndvi_from(nir, red)
    assert out[0] == pytest.approx(0.5) and np.isnan(out[1]) and np.isnan(out[2]) and out[3] == 0.0


def test_polygons_with_no_data_under_them_are_refused():
    """What a layer with the wrong CRS in QGIS gives: polygons that land where no layer has data."""
    empty = layer("rgb", value=np.nan)
    with pytest.raises(InputError, match="all without data"):
        assemble_inputs({"rgb": empty}, BLOCKS)
    between_centres = [Block(1, box(1001.01, 1998.01, 1001.02, 1998.02), "test")]  # pixel centres at 1000.9, 1001.1
    with pytest.raises(InputError, match="no pixel centre"):
        assemble_inputs({"rgb": layer("rgb")}, between_centres)
    partial = layer("rgb")
    partial.data[:, 4:36, 4:46] = np.nan  # everything under BLOCKS[0] (rows 5-34, columns 5-44)
    with pytest.raises(InputError, match="no input layer has data"):
        assemble_inputs({"rgb": partial}, BLOCKS)
    elsewhere = BLOCKS + [Block(2, box(1001, 1992, 1003, 1993.5), "test")]  # one polygon with data is enough
    inputs = assemble_inputs({"rgb": partial}, elsewhere)
    assert (inputs.masks[2] & inputs.valid).any() and not (inputs.masks[1] & inputs.valid).any()


def test_bad_inputs_are_refused():
    with pytest.raises(InputError, match="at least one"):
        assemble_inputs({}, BLOCKS)
    with pytest.raises(InputError, match="no polygons"):
        assemble_inputs({"rgb": layer("rgb")}, [])
    other = Grid(0.2, 0, 1000.2, 0, -0.2, 2000.0, 50, 40)
    with pytest.raises(InputError, match="not on the RGB layer's grid"):
        assemble_inputs({"rgb": layer("rgb"), "ndvi": layer("ndvi", grid=other)}, BLOCKS)
    two_bands = Raster(np.zeros((2, 40, 50), np.float32), GRID, ("red", "green"))
    with pytest.raises(InputError, match="needs 3 band"):
        assemble_inputs({"rgb": two_bands}, BLOCKS)
    scaled = layer("ndvi")
    scaled.data[0] = 180.0  # an 8-bit rendering, not NDVI
    with pytest.raises(InputError, match="outside -1..1"):
        assemble_inputs({"ndvi": scaled}, BLOCKS)
    with pytest.raises(InputError, match="unknown"):
        assemble_inputs({"swir": layer("rgb")}, BLOCKS)
