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
