"""File adapters on small temporary files (no orthophotos needed)."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from shapely.geometry import LineString, box

from tools.geo import Grid
from tools.io import describe_raster, read_polygons, read_raster, write_layer, write_raster

D96 = CRS.from_epsg(3794)
GRID = Grid(0.5, 0, 500000.0, 0, -0.5, 100010.0, 20, 20, D96.to_wkt())


def test_raster_round_trip(tmp_path):
    data = np.arange(3 * 20 * 20, dtype=np.uint8).reshape(3, 20, 20)
    path = write_raster(tmp_path / "x.tif", data, GRID, nodata=255)
    r = read_raster(path, ("red", "green", "blue"))
    assert np.array_equal(r.data, data) and r.grid.same_grid(GRID)
    assert r.nodata == 255 and np.array_equal(r.band("green"), data[1])
    meta = describe_raster(path)
    assert meta["count"] == 3 and meta["pixel_size"] == [0.5, 0.5] and meta["crs_units"] == "metre"


def test_bool_rasters_are_written_as_bytes(tmp_path):
    path = write_raster(tmp_path / "mask.tif", np.eye(20, dtype=bool), GRID)
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint8" and src.read(1).sum() == 20


def test_band_names_must_match_the_band_count(tmp_path):
    path = write_raster(tmp_path / "x.tif", np.zeros((3, 20, 20), np.uint8), GRID)
    with pytest.raises(ValueError, match="3 bands"):
        read_raster(path, ("red", "green"))


def _write(tmp_path, crs):
    path = tmp_path / "raw.tif"
    with rasterio.open(path, "w", driver="GTiff", width=4, height=4, count=1, dtype="uint8", crs=crs,
                       transform=rasterio.Affine(1, 0, 0, 0, -1, 4)) as dst:
        dst.write(np.zeros((1, 4, 4), np.uint8))
    return path


def test_missing_or_geographic_crs_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no CRS"):
        read_raster(_write(tmp_path, None), ("gray",))
    with pytest.raises(ValueError, match="not projected"):
        read_raster(_write(tmp_path, "EPSG:4326"), ("gray",))


def test_polygons_with_ids_attributes_and_explicit_reprojection(tmp_path):
    path = tmp_path / "blocks.gpkg"
    gpd.GeoDataFrame({"block_no": [4, 9], "name": ["a", "b"]}, geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
                     crs=D96).to_file(path)
    ids, geoms, prov, records = read_polygons(path, None, "block_no", D96.to_wkt())
    assert ids == [4, 9] and not prov["reprojected"] and records[1]["name"] == "b"
    assert geoms[0].area == pytest.approx(100)
    ids, geoms, prov, _ = read_polygons(path, None, None, CRS.from_epsg(3857).to_wkt())
    assert ids == [0, 1] and prov["reprojected"] and geoms[0].area != pytest.approx(100)


@pytest.mark.filterwarnings("ignore:'crs' was not provided")  # writing the no-CRS file is the point
def test_bad_polygon_inputs(tmp_path):
    dup = tmp_path / "dup.gpkg"
    gpd.GeoDataFrame({"k": [1, 1]}, geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)], crs=D96).to_file(dup)
    with pytest.raises(ValueError, match="duplicate"):
        read_polygons(dup, None, "k", D96.to_wkt())
    lines = tmp_path / "lines.gpkg"
    gpd.GeoDataFrame({"k": [1]}, geometry=[LineString([(0, 0), (1, 1)])], crs=D96).to_file(lines)
    with pytest.raises(ValueError, match="polygons"):
        read_polygons(lines, None, None, D96.to_wkt())
    nocrs = tmp_path / "nocrs.gpkg"
    gpd.GeoDataFrame({"k": [1]}, geometry=[box(0, 0, 1, 1)]).to_file(nocrs)
    with pytest.raises(ValueError, match="no CRS"):
        read_polygons(nocrs, None, None, D96.to_wkt())


def test_layers_shapefile_names_and_empty_layers(tmp_path):
    wkt = D96.to_wkt()
    with pytest.raises(ValueError, match="10 characters"):
        write_layer(tmp_path / "a.shp", [{"much_too_long_name": 1}], [box(0, 0, 1, 1)], wkt, driver="ESRI Shapefile")
    write_layer(tmp_path / "empty.shp", [], [], wkt, driver="ESRI Shapefile", fields=("id", "check"), geometry_type="Point")
    empty = gpd.read_file(tmp_path / "empty.shp")
    assert len(empty) == 0 and {"id", "check"} <= set(empty.columns)
    with pytest.raises(ValueError, match="empty layer"):
        write_layer(tmp_path / "x.gpkg", [], [], wkt, layer="x")
    write_layer(tmp_path / "x.gpkg", [{"a": 1}], [box(0, 0, 1, 1)], wkt, layer="first")
    write_layer(tmp_path / "x.gpkg", [{"b": 2}], [box(0, 0, 2, 2)], wkt, layer="second")
    assert gpd.read_file(tmp_path / "x.gpkg", layer="second").b.iloc[0] == 2
