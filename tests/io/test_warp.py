"""Reprojection onto one analysis grid (tools/io/warp.py), on small temporary rasters."""

import numpy as np
import pytest
import rasterio

from tools.io import SourceLayer, read_raster, target_grid, warp_to_grid

# A geographic raster near Cormons, as drone software writes it: EPSG:4326, about 0.08 x 0.11 m pixels.
LON0, LAT0, STEP = 13.4604, 45.9716, 1e-6
SIZE = 200  # pixels per side -> about 15 x 22 m


def _geographic(path, data, nodata):
    with rasterio.open(path, "w", driver="GTiff", width=SIZE, height=SIZE, count=data.shape[0], dtype=data.dtype.name,
                       crs="EPSG:4326", transform=rasterio.Affine(STEP, 0, LON0, 0, -STEP, LAT0), nodata=nodata) as dst:
        dst.write(data)
    return path


def test_grid_is_projected_north_up_and_covers_every_source(tmp_path):
    a = _geographic(tmp_path / "a.tif", np.ones((1, SIZE, SIZE), np.float32), None)
    grid = target_grid([a], "EPSG:6708", 0.5)
    assert grid.is_north_up and grid.pixel_size == (0.5, 0.5)
    assert grid.c % 0.5 == 0 and grid.f % 0.5 == 0  # corners on whole pixels
    with rasterio.open(a) as src:
        from rasterio.warp import transform_bounds

        xmin, ymin, xmax, ymax = transform_bounds(src.crs, "EPSG:6708", *src.bounds, densify_pts=21)
    gx0, gy0, gx1, gy1 = grid.bounds()
    assert gx0 <= xmin and gy0 <= ymin and gx1 >= xmax and gy1 >= ymax
    assert gx1 - xmax < 0.5 and gy1 - ymax < 0.5  # no more than a pixel of margin
    with pytest.raises(ValueError, match="not a projected CRS"):
        target_grid([a], "EPSG:4326", 0.5)


def test_average_resampling_keeps_values_and_float_nodata_becomes_nan(tmp_path):
    data = np.full((1, SIZE, SIZE), 0.8, np.float32)
    data[0, :, : SIZE // 2] = 0.0  # declared NoData: the left half
    src = _geographic(tmp_path / "ndvi_src.tif", data, 0.0)
    grid = target_grid([src], "EPSG:6708", 0.5)
    tags = warp_to_grid(SourceLayer("ndvi", src), tmp_path / "ndvi.tif", grid)
    out = read_raster(tmp_path / "ndvi.tif", ("ndvi",))
    band = out.data[0]
    assert out.grid.same_grid(grid) and np.isnan(out.nodata)
    assert np.nanmin(band) == pytest.approx(0.8) and np.nanmax(band) == pytest.approx(0.8)  # NoData not averaged in
    assert 0.3 < np.isnan(band).mean() < 0.8  # the NoData half and the corners outside the source
    assert tags["VINELINER_RESAMPLING"] == "average" and tags["VINELINER_SOURCE_CRS"] == "EPSG:4326"
    with rasterio.open(tmp_path / "ndvi.tif") as f:
        assert f.tags()["VINELINER_SOURCE"] == str(src)  # provenance travels with the file into report.json
        assert "RDN2008" in f.tags()["VINELINER_TRANSFORMATION"]


def test_band_subset_and_integer_nodata(tmp_path):
    rgba = np.zeros((4, SIZE, SIZE), np.uint8)
    rgba[0], rgba[1], rgba[2], rgba[3] = 50, 100, 150, 255
    rgba[:3, :20, :] = 0  # NoData strip at the top
    src = _geographic(tmp_path / "rgba.tif", rgba, 0)
    grid = target_grid([src], "EPSG:6708", 0.25)
    warp_to_grid(SourceLayer("rgb", src, (1, 2, 3)), tmp_path / "rgb.tif", grid)
    out = read_raster(tmp_path / "rgb.tif", ("red", "green", "blue"))
    assert out.data.dtype == np.uint8 and out.nodata == 0
    inside = out.data[0] > 0
    assert inside.any() and set(np.unique(out.data[2][inside])) == {150}
    with pytest.raises(ValueError, match="cannot keep bands"):
        warp_to_grid(SourceLayer("bad", src, (5,)), tmp_path / "bad.tif", grid)
    with pytest.raises(ValueError, match="unknown resampling"):
        warp_to_grid(SourceLayer("rgb", src), tmp_path / "x.tif", grid, resampling="cubic_magic")


def test_layers_from_different_grids_land_on_one_grid(tmp_path):
    fine = _geographic(tmp_path / "fine.tif", np.ones((1, SIZE, SIZE), np.float32), None)
    with rasterio.open(tmp_path / "coarse.tif", "w", driver="GTiff", width=SIZE // 2, height=SIZE // 2, count=1, dtype="float32",
                       crs="EPSG:4326", transform=rasterio.Affine(2 * STEP, 0, LON0, 0, -2 * STEP, LAT0)) as dst:
        dst.write(np.full((1, SIZE // 2, SIZE // 2), 2.0, np.float32))
    grid = target_grid([fine, tmp_path / "coarse.tif"], "EPSG:6708", 0.5)
    a = warp_to_grid(SourceLayer("a", fine), tmp_path / "a.tif", grid)
    b = warp_to_grid(SourceLayer("b", tmp_path / "coarse.tif"), tmp_path / "b.tif", grid)
    ra, rb = read_raster(tmp_path / "a.tif", ("a",)), read_raster(tmp_path / "b.tif", ("b",))
    assert ra.grid.same_grid(rb.grid) and ra.grid.crs_wkt == rb.grid.crs_wkt
    assert a["VINELINER_SOURCE_PIXEL"] != b["VINELINER_SOURCE_PIXEL"]
