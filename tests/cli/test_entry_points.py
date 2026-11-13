"""Entry files in src/, with the pipeline mocked out."""

import importlib.util
import json

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import box

from tools.config import CONFIG_DIR, PROJECT_ROOT
from tools.geo import Grid
from tools.io import write_raster

SRC = PROJECT_ROOT / "src"


def load(name):
    spec = importlib.util.spec_from_file_location(f"entry_{name}", SRC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_pipeline_cli():
    return load("run_pipeline")


def test_missing_open_choices_exit_with_code_2(run_pipeline_cli, capsys):
    code = run_pipeline_cli.main([str(CONFIG_DIR / "default.toml")])
    err = capsys.readouterr().err
    assert code == 2 and "Missing keys" in err and "[input] rgb" in err


def test_plan_merges_block_subset_into_the_config(run_pipeline_cli, monkeypatch, capsys):
    seen = {}

    def fake_plan(config):
        seen["config"] = config
        return {"centres": 12, "blocks": config["vineyards"]["blocks"]}

    monkeypatch.setattr(run_pipeline_cli, "plan_run", fake_plan)
    code = run_pipeline_cli.main([str(CONFIG_DIR / "centroid-pretest.toml"), "--plan", "--blocks", "0", "3"])
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"centres": 12, "blocks": [0, 3]}
    assert seen["config"]["centroid_pretest"]["enabled"] is True  # the named config was loaded


def test_run_passes_name_batch_size_and_merged_configs(run_pipeline_cli, monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(run_pipeline_cli, "run_pipeline", lambda config, **kw: calls.update(config=config, **kw))
    extra = tmp_path / "extra.toml"
    extra.write_text("[centres]\nspacing_m = 3.0\n")
    code = run_pipeline_cli.main([str(CONFIG_DIR / "full-search.toml"), str(extra), "--name", "trial", "--batch-size", "16"])
    assert code == 0
    assert calls["name"] == "trial" and calls["batch_size"] == 16
    assert calls["config"]["centres"]["spacing_m"] == 3.0  # later file wins
    assert calls["config"]["centroid_pretest"]["enabled"] is False  # full-search.toml, through its extends chain


def test_describe_inputs_reports_rasters_and_polygons(tmp_path, capsys):
    crs = CRS.from_epsg(3794)
    tif = write_raster(tmp_path / "x.tif", np.zeros((2, 8, 8), np.uint8), Grid(0.5, 0, 0, 0, -0.5, 4, 8, 8, crs.to_wkt()))
    gpkg = tmp_path / "b.gpkg"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 2, 2)], crs=crs).to_file(gpkg)
    assert load("describe_inputs").main([str(tif), str(gpkg)]) == 0
    raster, vector = json.loads(capsys.readouterr().out)
    assert raster["count"] == 2 and "crs_wkt" not in raster
    assert vector["layers"][0]["features"] == 1 and vector["layers"][0]["areas_m2"] == [4.0]


def test_layers_switch_off_every_other_layer(run_pipeline_cli, monkeypatch):
    calls = {}
    monkeypatch.setattr(run_pipeline_cli, "run_pipeline", lambda config, **kw: calls.update(config=config))
    assert run_pipeline_cli.main([str(CONFIG_DIR / "centroid-pretest.toml"), "--layers", "ndvi", "nir"]) == 0
    flags = {k: calls["config"]["input"][k] for k in ("use_rgb", "use_nir", "use_ndvi")}
    assert flags == {"use_rgb": False, "use_nir": True, "use_ndvi": True}


def test_layers_without_a_path_exit_with_code_2(run_pipeline_cli, tmp_path, capsys):
    only_rgb = tmp_path / "rgb.toml"
    only_rgb.write_text('[input]\nrgb = "x.tif"\nrgb_bands = ["red", "green", "blue"]\n')
    assert run_pipeline_cli.main([str(only_rgb), "--layers", "nir"]) == 2
    assert "no [input] path for nir" in capsys.readouterr().err


def test_prepare_inputs_puts_layers_on_one_grid_with_provenance(tmp_path, capsys):
    import rasterio

    src = tmp_path / "ndvi_4326.tif"
    with rasterio.open(src, "w", driver="GTiff", width=50, height=50, count=1, dtype="float32", crs="EPSG:4326",
                       transform=rasterio.Affine(1e-6, 0, 13.4604, 0, -1e-6, 45.9716), nodata=0.0) as dst:
        dst.write(np.full((1, 50, 50), 0.7, np.float32))
    rgb = tmp_path / "rgba_4326.tif"
    with rasterio.open(rgb, "w", driver="GTiff", width=80, height=80, count=4, dtype="uint8", crs="EPSG:4326",
                       transform=rasterio.Affine(6e-7, 0, 13.4604, 0, -6e-7, 45.9716), nodata=0) as dst:
        dst.write(np.full((4, 80, 80), 120, np.uint8))
    out = tmp_path / "area"
    code = load("prepare_inputs").main([str(out), "--crs", "EPSG:6708", "--pixel", "0.5",
                                        "--layer", "ndvi", str(src), "--layer", "rgb", str(rgb), "1", "2", "3"])
    assert code == 0 and "Grid:" in capsys.readouterr().out
    summary = json.loads((out / "prepare.json").read_text())
    assert summary["crs"] == "EPSG:6708" and summary["layers"]["rgb"]["count"] == 3  # alpha dropped
    assert summary["layers"]["ndvi"]["provenance"]["VINELINER_SOURCE_CRS"] == "EPSG:4326"
    with rasterio.open(out / "ndvi.tif") as a, rasterio.open(out / "rgb.tif") as b:
        assert a.transform == b.transform and a.shape == b.shape and a.crs.to_epsg() == 6708


def test_prepare_inputs_refuses_bad_layer_arguments(tmp_path):
    prepare = load("prepare_inputs")
    with pytest.raises(SystemExit):
        prepare.main([str(tmp_path), "--crs", "EPSG:6708", "--pixel", "0.5", "--layer", "rgb", "x.tif", "one"])
    with pytest.raises(SystemExit):
        prepare.main([str(tmp_path), "--crs", "EPSG:6708", "--pixel", "0.5", "--layer", "rgb"])
