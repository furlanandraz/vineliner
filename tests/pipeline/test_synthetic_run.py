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
    "scoring": {"normalisation": "renormalise_valid", "min_valid_coverage": 0.5},
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


PRETEST = {"centroid_pretest": {"enabled": True, "theta_window_deg": 5.0, "spacing_window_frac": 0.05, "neighbour_rings": 3}}


@pytest.fixture(scope="module")
def run_pretest(tmp_path_factory):
    """Centroid pre-test on, without the faint patch (which covers the block centroid)."""
    return _run(tmp_path_factory.mktemp("synthetic_pretest"), PRETEST, patch=False)


@pytest.fixture(scope="module")
def run_pretest_bad_centre(tmp_path_factory):
    """Pre-test on, with the faint patch over the centroid: a neighbouring box has to stand in."""
    return _run(tmp_path_factory.mktemp("synthetic_pretest_patch"), PRETEST)


@pytest.fixture(scope="module")
def run_pretest_no_neighbours(tmp_path_factory):
    """As above but without neighbouring boxes: the block falls back to the full search."""
    no_rings = {"centroid_pretest": dict(PRETEST["centroid_pretest"], neighbour_rings=0)}
    return _run(tmp_path_factory.mktemp("synthetic_pretest_no_rings"), no_rings)


@pytest.fixture(scope="module")
def run_row_estimate(tmp_path_factory):
    """One row erased from the image; the missing-row estimate should put it back."""
    overrides = {"missing_row_estimate": {"enabled": True}}  # drop_hanging_rows follows automatically
    return _run(tmp_path_factory.mktemp("synthetic_row_estimate"), overrides, patch=False, erase_row=ERASED_ROW)


@pytest.fixture(scope="module")
def run_numbering(tmp_path_factory):
    """Erased row, estimate on, and row numbering."""
    overrides = {"missing_row_estimate": {"enabled": True}, "row_numbering": {"enabled": True}}
    return _run(tmp_path_factory.mktemp("synthetic_numbering"), overrides, patch=False, erase_row=ERASED_ROW)


ERASED_ROW = -28  # row index i of the line (p - origin) . n = i * spacing; the block holds rows -46 .. -9

# Single-layer runs on the multispectral synthetic scene (no faint patch). Switching RGB
# off needs nothing else: the remaining weights are rescaled over the layers in use.
NDVI_ONLY = {"input": {"use_rgb": False, "use_nir": False}, "evidence": {"ndvi_weight": 1.0, "ndvi_method": "stretch"}}
NIR_ONLY = {
    "input": {"use_rgb": False, "use_ndvi": False},
    "evidence": {"nir_weight": 1.0, "nir_method": "white_tophat", "nir_tophat_size_m": 1.2},
}
NAN_CORNER_PX = 20  # NIR and NDVI are NaN (their NoData) in this top-left corner, outside the block


@pytest.fixture(scope="module")
def run_ndvi_only(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("synthetic_ndvi_only"), NDVI_ONLY, patch=False, multispectral=True)


@pytest.fixture(scope="module")
def run_nir_only(tmp_path_factory):
    """NIR alone, with outputs under a configured runs_dir instead of an explicit run directory."""
    tmp = tmp_path_factory.mktemp("synthetic_nir_only")
    return _run(tmp, merge(NIR_ONLY, {"export": {"runs_dir": str(tmp / "runs" / "area")}}), patch=False,
                multispectral=True, run_dir=None)


def _run(tmp, overrides, patch=True, erase_row=None, multispectral=False, run_dir="run"):
    """multispectral: also write float NIR and NDVI layers (rows bright) on the same grid."""
    from pyproj import CRS

    crs = CRS.from_epsg(3794)
    grid = synthetic_grid(crs.to_wkt())
    V = stripe_evidence(grid, half_width=0.3)
    if patch:
        V[_patch_mask(grid)] = 0  # a faint patch with no visible rows, to be bridged
    if erase_row is not None:
        V[_row_mask(grid, erase_row)] = 0  # a whole row missing from the image
    rng = np.random.default_rng(0)
    gray = 200 - 110 * V + rng.normal(0, 8, V.shape)  # dark rows on bright soil, with noise
    rgb = np.clip(np.stack([gray, gray, gray]), 0, 254).astype(np.uint8)
    tif = tmp / "rgb.tif"
    _write_tif(tif, rgb, grid, crs, nodata=255)
    x0, y0 = grid.c, grid.f
    block = Polygon([(x0 + 8, y0 - 10), (x0 + 50, y0 - 6), (x0 + 54, y0 - 50), (x0 + 10, y0 - 52)])
    gpkg = tmp / "blocks.gpkg"
    gpd.GeoDataFrame({"id": [7]}, geometry=[block], crs=crs).to_file(gpkg)
    cfg = merge(load_config(), CONFIG)
    cfg = merge(cfg, {"input": {"rgb": str(tif)}, "vineyards": {"source": str(gpkg), "id_field": "id"}})
    if multispectral:
        nir = (1000 + 1200 * V + rng.normal(0, 60, V.shape)).astype(np.float32)  # reflectance-like, canopy bright
        ndvi = np.clip(0.3 + 0.5 * V + rng.normal(0, 0.04, V.shape), -1, 1).astype(np.float32)
        nir[:NAN_CORNER_PX, :NAN_CORNER_PX] = np.nan
        ndvi[:NAN_CORNER_PX, :NAN_CORNER_PX] = np.nan
        _write_tif(tmp / "nir.tif", nir[np.newaxis], grid, crs, nodata=float("nan"))
        _write_tif(tmp / "ndvi.tif", ndvi[np.newaxis], grid, crs, nodata=float("nan"))
        cfg = merge(cfg, {"input": {"nir": str(tmp / "nir.tif"), "nir_bands": ["nir"], "ndvi": str(tmp / "ndvi.tif")}})
    cfg = merge(cfg, {"kernels": {"cache": True, "cache_dir": str(tmp / "kernel_cache")}})  # test kernels stay out of the project
    cfg = merge(cfg, overrides)
    result = run_pipeline(cfg, run_dir=tmp / run_dir if run_dir else None, log=lambda *_: None)
    return result, block


def _write_tif(path, data, grid, crs, nodata):
    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height, count=data.shape[0],
                       dtype=data.dtype.name, crs=crs, transform=rasterio.Affine(*grid.affine), nodata=nodata) as dst:
        dst.write(data)


def test_outputs_exist(run):
    result, _ = run
    for name in ("report.json", "config.json", "evidence.tif", "valid.tif", "vineliner.gpkg", "overlay.png",
                 "shapefiles/rows.shp", "shapefiles/blocks.shp", "shapefiles/bridges.shp", "scores/manifest.json"):
        assert (result.run_dir / name).exists(), name


def test_every_true_row_is_found_once(run):
    _assert_every_true_row_found_once(*run)


def test_single_layer_runs_find_every_true_row(run_ndvi_only, run_nir_only):
    _assert_every_true_row_found_once(*run_ndvi_only)
    _assert_every_true_row_found_once(*run_nir_only)


def test_switched_off_layers_are_left_out_and_reported(run_ndvi_only):
    result, _ = run_ndvi_only
    inputs = result.report["inputs"]
    assert inputs["layers_in_use"] == ["ndvi"] and inputs["layers_switched_off"] == ["rgb", "nir"]
    assert [r["layer"] for r in inputs["rasters"]] == ["ndvi"]  # switched-off layers are not even read
    assert inputs["invalid_pixels"] == NAN_CORNER_PX**2  # NaN is the NDVI's NoData
    evidence = result.report["evidence"]
    assert evidence["weights"] == {"ndvi": 1.0} and "rgb" not in evidence
    assert any("switched off: rgb" in note for note in result.report["settings_notes"])
    assert (result.run_dir / "overlay.png").exists()  # drawn on the NDVI, as there is no RGB


def test_nir_layer_alone_uses_its_band_and_the_configured_runs_dir(run_nir_only):
    result, _ = run_nir_only
    assert result.run_dir.parent.name == "area" and result.run_dir.parent.parent.name == "runs"
    assert result.report["evidence"]["nir"]["method"] == "white_tophat"
    assert result.report["inputs"]["layers_in_use"] == ["nir"]


def _assert_every_true_row_found_once(result, block):
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


def test_weak_centroid_box_is_backed_up_by_a_neighbour(run_pretest_bad_centre):
    result, block = run_pretest_bad_centre
    pt = result.report["blocks"][0]["centroid_pretest"]
    assert result.report["blocks"][0]["search_mode"] == "pretest"
    assert pt["ring"] >= 1 and pt["boxes_tested"] > 1
    assert abs(pt["theta_deg"] - SYNTHETIC_THETA) <= 1
    assert len(result.sections) >= 30


def test_without_neighbours_a_weak_centroid_falls_back(run_pretest_no_neighbours):
    result, _ = run_pretest_no_neighbours
    report = result.report["blocks"][0]
    assert report["search_mode"] == "full" and "accept_min" in report["centroid_pretest"]["reason"]


def test_pretest_search_finds_the_same_rows_with_fewer_scores(run, run_pretest):
    full, _ = run
    pt, block = run_pretest
    report = pt.report["blocks"][0]
    assert report["search_mode"] == "pretest" and report["centroid_pretest"]["ring"] == 0
    assert abs(report["centroid_pretest"]["theta_deg"] - SYNTHETIC_THETA) <= 1
    assert report["hypotheses_per_centre"] < full.report["blocks"][0]["hypotheses_per_centre"] / 2
    assert pt.report["totals"]["template_scores"] < full.report["totals"]["template_scores"] / 2
    for line in _true_rows(block):
        mid = line.interpolate(0.5, normalized=True)
        matches = [s for s in pt.sections if s.geometry.distance(mid) < 0.3]
        assert len(matches) == 1 and matches[0].geometry.hausdorff_distance(line) < 0.5
    blocks = gpd.read_file(pt.run_dir / "shapefiles" / "blocks.shp")
    assert blocks.search.iloc[0] == "pretest" and abs(blocks.pt_theta.iloc[0] - SYNTHETIC_THETA) <= 1


def test_kernels_are_cached(run):
    result, _ = run
    assert result.report["kernels"]["cache"]["cache"].startswith("miss")
    assert list((result.run_dir.parent / "kernel_cache").glob("kernels_*.npz"))


def test_missing_row_is_estimated_where_it_belongs(run_row_estimate):
    result, block = run_row_estimate
    truth = _row_line(block, ERASED_ROW)
    assert truth.length > 30  # a long row through the middle
    estimated = [(s, r) for s, r in zip(result.sections, result.reviews) if r["estimated"]]
    assert len(estimated) == 1
    sec, rec = estimated[0]
    assert sec.geometry.hausdorff_distance(truth) < 0.3
    assert (rec["conf"], rec["review"]) == ("estimated", 1) and "estimated between rows" in rec["issues"]
    # ev_mean is the five-line template score: its four neighbour lines still sit on real rows, so about
    # 0.45 even with nothing on the centre line, below the 0.5 a detected row needs.
    assert rec["ev_mean"] < 0.5
    report = result.report
    assert report["settings_notes"] and report["totals"]["rows_estimated"] == 1
    mre = report["blocks"][0]["validation"]["missing_row_estimate"]
    assert mre["row_width_m"] == pytest.approx(SYNTHETIC_SPACING, abs=0.05) and mre["rows_added"] == 1
    rows = gpd.read_file(result.run_dir / "shapefiles" / "rows.shp")
    assert int(rows.estimated.sum()) == 1 and set(rows[rows.estimated == 1].conf) == {"estimated"}
    blocks = gpd.read_file(result.run_dir / "shapefiles" / "blocks.shp")
    assert int(blocks.estimated.iloc[0]) == 1


def test_without_the_estimate_the_erased_row_stays_missing(run_dropping):
    result, block = run_dropping
    assert result.report["totals"]["rows_estimated"] == 0


def test_final_rows_have_no_turn_above_the_smoothing_limit(run, run_row_estimate):
    from tools.reconstruction import turn_angles

    for result, _ in (run, run_row_estimate):
        for s in result.sections:
            coords = np.asarray(s.geometry.coords)
            if len(coords) > 2:
                assert turn_angles(coords).max() <= 20.0 + 1e-6


def test_rows_are_numbered_without_gaps_once_the_missing_row_is_estimated(run_numbering):
    result, block = run_numbering
    nos = [r["row_no"] for r in result.reviews]
    assert sorted(nos) == list(range(1, len(nos) + 1))  # one number per row, none skipped
    assert all(r["row_no_ok"] == 1 for r in result.reviews)
    # Numbers increase across the rows: compare with each row's position along the across direction.
    th = np.radians(SYNTHETIC_THETA)
    n = np.array([-np.sin(th), np.cos(th)])
    pos = [np.dot(np.array(s.geometry.interpolate(0.5, normalized=True).coords[0]), n) for s in result.sections]
    order = [r["row_no"] for _, r in sorted(zip(pos, result.reviews), key=lambda t: t[0])]
    assert order == sorted(order)
    est = next(r for r in result.reviews if r["estimated"])
    assert 1 < est["row_no"] < len(nos)  # the estimated row takes the number between its neighbours
    blocks = gpd.read_file(result.run_dir / "shapefiles" / "blocks.shp")
    assert int(blocks.max_row_no.iloc[0]) == len(nos) and int(blocks.missing_no.iloc[0]) == 0
    rows = gpd.read_file(result.run_dir / "shapefiles" / "rows.shp")
    assert {"row_no", "row_no_ok"} <= set(rows.columns)


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


def _row_mask(grid, i) -> np.ndarray:
    rows, cols = np.indices(grid.shape)
    x, y = grid.array_to_world(rows, cols)
    th = np.radians(SYNTHETIC_THETA)
    s = (x - grid.c) * -np.sin(th) + (y - grid.f) * np.cos(th)
    return np.abs(s - i * SYNTHETIC_SPACING) <= 0.5


def _row_line(block, i) -> LineString:
    grid = synthetic_grid()
    th = np.radians(SYNTHETIC_THETA)
    u, n = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    c = np.array([grid.c, grid.f]) + i * SYNTHETIC_SPACING * n
    return LineString([c - 200 * u, c + 200 * u]).intersection(block)


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
