"""End-to-end run on a synthetic striped vineyard with known rows."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, Polygon

from synthetic import SYNTHETIC_SPACING, SYNTHETIC_THETA, stripe_evidence, synthetic_grid
from tools.config import load_config, merge
from tools.geo import axial_difference
from tools.pipeline import run_pipeline

CONFIG = {
    "input": {"rgb_bands": ["red", "green", "blue"], "nodata_value": 255},
    "evidence": {"method": "gray_stretch", "scale_region": "per_block"},
    "template": {"line_length_m": 2.0, "band_width_px": 3},
    "kernels": {"coverage": "exact"},
    # A narrow rotation window keeps the test fast; real runs search 0-179.
    "rotation": {"start_deg": 20, "stop_deg": 41, "step_deg": 1},
    "spacing": {"nominal_m": 1.5, "min_m": 1.4, "max_m": 1.6, "step_m": 0.05},
    "translation": {"box": "nominal_d0"},
    "centres": {"spacing_m": 3.0},
    "scoring": {"normalisation": "renormalise_valid", "min_valid_coverage": 0.5, "unavailable_bands": "renormalise"},
    "selection": {"method": "orientation_consensus", "consensus_radius_m": 6.0, "consensus_window_deg": 5.0},
    "seeds": {"lines": [-1, 0, 1], "min_band_mean": 0.4},
    "reconstruction": {
        "link_angle_deg": 10.0, "link_lateral_frac": 0.25, "link_along_m": 6.0, "chain_smoothing_m": 4.0,
        "growth_min_score": 0.4, "max_gap_m": 50.0, "bridge_tangent_m": 10.0, "join_distance_frac": 0.35,
        "join_angle_deg": 20.0, "max_row_length_m": 200.0,
    },
    "validation": {"spacing_min_m": 1.3, "spacing_max_m": 1.7, "tolerance_m": 0.3},
    "storage": {"surfaces": "winning_spacing"},
    "export": {"overlay": True},
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("synthetic"), {})


@pytest.fixture(scope="module")
def run_dropping(tmp_path_factory):
    """Without bridging the rows break at the faint patch and dangle; drop_hanging_rows removes them."""
    overrides = {"reconstruction": {"max_gap_m": 0.5}, "export": {"drop_hanging_rows": True}}
    return _run(tmp_path_factory.mktemp("synthetic_drop"), overrides)


def _run(tmp, overrides):
    from pyproj import CRS

    crs = CRS.from_epsg(3794)
    grid = synthetic_grid(crs.to_wkt())
    V = stripe_evidence(grid, half_width=0.3)
    V[_patch_mask(grid)] = 0  # a faint patch with no visible rows, to be bridged
    rng = np.random.default_rng(0)
    gray = 200 - 110 * V + rng.normal(0, 8, V.shape)  # dark rows on bright soil, with noise
    rgb = np.clip(np.stack([gray, gray, gray]), 0, 254).astype(np.uint8)
    tif = tmp / "rgb.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=grid.width, height=grid.height, count=3, dtype="uint8",
                       crs=crs, transform=rasterio.Affine(*grid.affine), nodata=255) as dst:
        dst.write(rgb)
    x0, y0 = grid.c, grid.f
    block = Polygon([(x0 + 8, y0 - 10), (x0 + 50, y0 - 6), (x0 + 54, y0 - 50), (x0 + 10, y0 - 52)])
    gpkg = tmp / "blocks.gpkg"
    gpd.GeoDataFrame({"id": [7]}, geometry=[block], crs=crs).to_file(gpkg)
    cfg = merge(load_config(), CONFIG)
    cfg = merge(cfg, {"input": {"rgb": str(tif)}, "vineyards": {"source": str(gpkg), "id_field": "id"}})
    cfg = merge(cfg, overrides)
    result = run_pipeline(cfg, run_dir=tmp / "run", log=lambda *_: None)
    return result, block


def test_outputs_exist(run):
    result, _ = run
    for name in ("report.json", "config.json", "evidence.tif", "valid.tif", "vineliner.gpkg", "overlay.png",
                 "shapefiles/rows.shp", "shapefiles/blocks.shp", "shapefiles/bridges.shp", "scores/manifest.json"):
        assert (result.run_dir / name).exists(), name


def test_every_true_row_is_found_once(run):
    result, block = run
    truth = _true_rows(block)
    used = set()
    for line in truth:
        mid = line.interpolate(0.5, normalized=True)
        matches = [s for s in result.sections if s.geometry.distance(mid) < 0.3]
        assert len(matches) == 1, f"row of {line.length:.1f} m matched {len(matches)} sections"
        s = matches[0]
        used.add(s.id)
        assert s.length_m == pytest.approx(line.length, rel=0.1, abs=0.5)
        assert s.geometry.hausdorff_distance(line) < 0.5  # well inside the ~1 m target
        assert (s.start_state, s.end_state) == ("boundary", "boundary")
    assert used == {s.id for s in result.sections}, "extra sections that match no true row"


def test_long_rows_have_the_true_direction(run):
    result, _ = run
    for s in result.sections:
        if s.length_m > 10:
            (x0, y0), (x1, y1) = s.geometry.coords[0], s.geometry.coords[-1]
            tolerance = 2.0 if s.length_m < 30 else 1.0  # end-to-end angle; short rows amplify end offsets
            assert axial_difference(np.degrees(np.arctan2(y1 - y0, x1 - x0)), SYNTHETIC_THETA) < tolerance


def test_only_corner_stubs_need_review(run):
    result, _ = run
    totals = result.report["totals"]
    assert totals["dangling_ends"] == 0 and totals["crossings"] == 0
    for r in result.reviews:
        if r["review"]:
            # Rows shorter than [review] min_length_m (4 m) at the block corners are flagged as short.
            assert r["length_m"] < 4.0 and r["issues"].startswith("short"), r["issues"]


def test_faint_patch_is_bridged_and_reported(run):
    result, _ = run
    patch = _patch_polygon(synthetic_grid())
    by_id = {r["id"]: r for r in result.reviews}
    crossing = [s for s in result.sections if s.geometry.crosses(patch)]
    assert len(crossing) >= 8
    for s in crossing:
        r = by_id[s.id]
        assert 4.0 <= r["bridged_m"] <= 8.0, r  # the patch is 6 m long along the rows
        assert r["conf"] == "medium" and "bridged" in r["issues"] and r["review"] == 0


def test_rows_shapefile_carries_review_attributes(run):
    from tools.validation import ROW_FIELDS

    result, _ = run
    rows = gpd.read_file(result.run_dir / "shapefiles" / "rows.shp")
    assert set(ROW_FIELDS) <= set(rows.columns)
    assert len(rows) == len(result.sections)
    assert (rows.conf == "high").sum() > len(rows) / 2
    assert (rows.block == 7).all()


def test_blocks_shapefile_has_a_vertex_at_every_row_end(run):
    result, block = run
    out = gpd.read_file(result.run_dir / "shapefiles" / "blocks.shp")
    assert len(out) == 1 and int(out["id"].iloc[0]) == 7  # input attribute kept
    poly = out.geometry.iloc[0]
    assert poly.symmetric_difference(block).area < 1e-6  # shape unchanged
    vertices = np.array([c for part in getattr(poly, "geoms", [poly]) for c in part.exterior.coords])
    rows = gpd.read_file(result.run_dir / "shapefiles" / "rows.shp")
    ends = [c for g in rows.geometry for c in (g.coords[0], g.coords[-1])]
    assert int(out["row_ends"].iloc[0]) == len(ends)
    for e in ends:
        assert np.min(np.hypot(*(vertices - np.array(e)).T)) < 1e-9


def test_hanging_rows_are_kept_by_default(run):
    result, _ = run
    assert result.report["totals"]["drop_hanging_rows"] is False
    assert not (result.run_dir / "shapefiles" / "removed.shp").exists()


def test_drop_hanging_rows_keeps_only_rows_meeting_the_boundary_twice(run_dropping):
    from tools.validation import meets_boundary_twice

    result, block = run_dropping
    totals = result.report["totals"]
    assert totals["removed_hanging"] >= 8  # the pieces either side of the faint patch
    assert len(result.sections) >= 20  # complete rows elsewhere are kept
    assert totals["dangling_ends"] == 0
    for s in result.sections:
        assert meets_boundary_twice(s.geometry, block, 0.3)
    removed = gpd.read_file(result.run_dir / "shapefiles" / "removed.shp")
    assert len(removed) == totals["removed_hanging"]
    assert (removed.dangling > 0).all() and (removed.review == 1).all()
    kept = gpd.read_file(result.run_dir / "shapefiles" / "rows.shp")
    assert (kept.dangling == 0).all()
    issues = gpd.read_file(result.run_dir / "shapefiles" / "issues.shp")
    assert "completion" not in set(issues.check)
    blocks = gpd.read_file(result.run_dir / "shapefiles" / "blocks.shp")
    assert int(blocks.removed.iloc[0]) == totals["removed_hanging"]


def test_report_lists_open_choices(run):
    result, _ = run
    chosen = result.report["open_choices_set_by_this_run"]
    assert chosen["translation.box"] == "nominal_d0"
    assert chosen["centres.spacing_m"] == 3.0


def _patch_polygon(grid):
    """A 6 m (along rows) x 16 m (across rows) patch in the middle of the block."""
    th = np.radians(SYNTHETIC_THETA)
    u, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    c = np.array([grid.c + 31.0, grid.f - 29.0])
    return Polygon([c - 3 * u - 8 * n, c + 3 * u - 8 * n, c + 3 * u + 8 * n, c - 3 * u + 8 * n])


def _patch_mask(grid) -> np.ndarray:
    import shapely

    rows, cols = np.indices(grid.shape)
    x, y = grid.array_to_world(rows, cols)
    return shapely.intersects_xy(_patch_polygon(grid), x, y)


def _true_rows(block) -> list[LineString]:
    """The synthetic row lines clipped to the block."""
    grid = synthetic_grid()
    th = np.radians(SYNTHETIC_THETA)
    u, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    origin = np.array([grid.c, grid.f])
    s = [(np.array(p) - origin) @ n for p in block.exterior.coords]
    rows = []
    for i in range(int(np.ceil(min(s) / SYNTHETIC_SPACING)), int(np.floor(max(s) / SYNTHETIC_SPACING)) + 1):
        c = origin + i * SYNTHETIC_SPACING * n
        clipped = LineString([c - 200 * u, c + 200 * u]).intersection(block)
        if clipped.length > 0:
            rows.append(clipped)
    return rows
