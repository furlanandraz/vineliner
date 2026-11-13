"""End-to-end run on a synthetic striped vineyard with known rows, on in-memory inputs."""

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from synthetic import SYNTHETIC_SPACING, SYNTHETIC_THETA, stripe_evidence, synthetic_grid
from vineliner.core.config import RunSettings, merge, plugin_config
from vineliner.core.export import azimuth_deg, clean_json, row_record
from vineliner.core.geo import axial_difference
from vineliner.core.pipeline import Raster, assemble_inputs, run
from vineliner.core.vineyards import Block

# The packaged defaults, made faster for the test (a narrow rotation window, coarse widths, sparse
# centres) and with the optional stages off; the fixtures below turn them on one at a time.
BASE = {
    "evidence": {"method": "gray_stretch"},
    "rotation": {"start_deg": 20, "stop_deg": 41, "step_deg": 1},  # real runs search 0-179
    "spacing": {"nominal_m": 1.5, "step_m": 0.05},
    "centres": {"spacing_m": 3.0},
    "reconstruction": {"link_along_m": 6.0, "max_row_length_m": 200.0, "drop_hanging_rows": False},
    "validation": {"spacing_min_m": 1.3, "spacing_max_m": 1.7},
    "centroid_pretest": {"enabled": False},
    "missing_row_estimate": {"enabled": False},
    "row_numbering": {"enabled": False},
}
PRETEST = {"centroid_pretest": {"enabled": True, "theta_window_deg": 5.0, "spacing_window_frac": 0.05, "neighbour_rings": 3}}
ERASED_ROW = -28  # row index i of the line (p - origin) . n = i * spacing; the block holds rows -46 .. -9
NAN_CORNER_PX = 20  # NIR-R-G and NDVI have no data (NaN) in this top-left corner, outside the block


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("kernel_cache")  # test kernels stay out of the QGIS profile


@pytest.fixture(scope="module")
def run_rgb(cache):
    return _run(cache, {})


@pytest.fixture(scope="module")
def run_dropping(cache):
    """Without bridging the rows break at the faint patch and dangle; drop_hanging_rows removes them."""
    return _run(cache, {"reconstruction": {"max_gap_m": 0.5, "drop_hanging_rows": True}})


@pytest.fixture(scope="module")
def run_pretest(cache):
    """Centroid pre-test on, without the faint patch (which covers the block centroid)."""
    return _run(cache, PRETEST, patch=False)


@pytest.fixture(scope="module")
def run_pretest_bad_centre(cache):
    """Pre-test on, with the faint patch over the centroid: a neighbouring box has to stand in."""
    return _run(cache, PRETEST)


@pytest.fixture(scope="module")
def run_pretest_no_neighbours(cache):
    """As above but without neighbouring boxes: the block falls back to the full search."""
    return _run(cache, {"centroid_pretest": dict(PRETEST["centroid_pretest"], neighbour_rings=0)})


@pytest.fixture(scope="module")
def run_row_estimate(cache):
    """One row erased from the image; the missing-row estimate should put it back."""
    return _run(cache, {"missing_row_estimate": {"enabled": True}}, patch=False, erase_row=ERASED_ROW)


@pytest.fixture(scope="module")
def run_numbering(cache):
    """Erased row, estimate on, and row numbering: the plugin's defaults."""
    overrides = {"missing_row_estimate": {"enabled": True}, "row_numbering": {"enabled": True}}
    return _run(cache, overrides, patch=False, erase_row=ERASED_ROW)


@pytest.fixture(scope="module")
def run_ndvi_only(cache):
    return _run(cache, {"evidence": {"ndvi_method": "stretch"}}, patch=False, layers=("ndvi",))


@pytest.fixture(scope="module")
def run_nir_r_g_only(cache):
    """The NIR-R-G layer alone: NIR evidence plus NDVI computed from its NIR and red bands."""
    return _run(cache, {"evidence": {"nir_tophat_size_m": 1.2, "ndvi_method": "stretch"}}, patch=False, layers=("nir",))


def _scene(patch=True, erase_row=None):
    grid = synthetic_grid()
    V = stripe_evidence(grid, half_width=0.3)
    if patch:
        V[_patch_mask(grid)] = 0  # a faint patch with no visible rows, to be bridged
    if erase_row is not None:
        V[_row_mask(grid, erase_row)] = 0  # a whole row missing from the image
    return grid, V


def _layers(grid, V, names):
    rng = np.random.default_rng(0)
    gray = np.clip(200 - 110 * V + rng.normal(0, 8, V.shape), 0, 254).astype(np.float32)  # dark rows on bright soil
    nir = (1000 + 1200 * V + rng.normal(0, 60, V.shape)).astype(np.float32)  # reflectance-like, canopy bright
    ndvi = np.clip(0.3 + 0.5 * V + rng.normal(0, 0.04, V.shape), -1, 1).astype(np.float32)
    nir_r_g = np.stack([nir, gray, gray])
    nir_r_g[:, :NAN_CORNER_PX, :NAN_CORNER_PX] = np.nan
    ndvi[:NAN_CORNER_PX, :NAN_CORNER_PX] = np.nan
    all_layers = {
        "rgb": Raster(np.stack([gray, gray, gray]), grid, ("red", "green", "blue")),
        "nir": Raster(nir_r_g, grid, ("nir", "red", "green")),
        "ndvi": Raster(ndvi[np.newaxis], grid, ("ndvi",)),
    }
    return {k: all_layers[k] for k in names}


def _block(grid) -> Polygon:
    x0, y0 = grid.c, grid.f
    return Polygon([(x0 + 8, y0 - 10), (x0 + 50, y0 - 6), (x0 + 54, y0 - 50), (x0 + 10, y0 - 52)])


def _run(cache, overrides, patch=True, erase_row=None, layers=("rgb",)):
    grid, V = _scene(patch, erase_row)
    polygon = _block(grid)
    settings = RunSettings.from_config(merge(merge(plugin_config(1.4, 1.6), BASE), overrides))
    inputs = assemble_inputs(_layers(grid, V, layers), [Block(7, polygon, "test")])
    result = run(settings, inputs, cache_dir=cache, log=lambda *_: None)
    return result, polygon


def test_every_true_row_is_found_once(run_rgb):
    _assert_every_true_row_found_once(*run_rgb)


def test_single_layer_runs_find_every_true_row(run_ndvi_only, run_nir_r_g_only):
    _assert_every_true_row_found_once(*run_ndvi_only)
    _assert_every_true_row_found_once(*run_nir_r_g_only)


def test_components_follow_the_layers_given(run_rgb, run_ndvi_only, run_nir_r_g_only):
    assert run_rgb[0].report["inputs"]["components"] == ["rgb"] and run_rgb[0].report["evidence"]["weights"] == {"rgb": 1.0}
    ndvi = run_ndvi_only[0].report
    assert ndvi["inputs"]["components"] == ["ndvi"] and ndvi["inputs"]["ndvi"] == "supplied"
    assert ndvi["inputs"]["invalid_pixels"] == NAN_CORNER_PX**2  # NaN is no data
    nrg = run_nir_r_g_only[0].report
    assert nrg["inputs"]["components"] == ["nir", "ndvi"] and nrg["inputs"]["ndvi"] == "computed from NIR-R-G"
    assert nrg["evidence"]["weights"] == {"nir": 0.5, "ndvi": 0.5} and nrg["evidence"]["nir"]["method"] == "white_tophat"


def test_each_block_uses_the_component_that_shows_its_rows(cache):
    """RGB is pure noise here, NDVI shows the rows: the block takes NDVI and finds every row."""
    grid, V = _scene(patch=False)
    layers = _layers(grid, V, ("rgb", "ndvi"))
    rng = np.random.default_rng(1)
    layers["rgb"].data[:] = rng.normal(150, 20, layers["rgb"].data.shape).astype(np.float32)
    polygon = _block(grid)
    settings = RunSettings.from_config(merge(merge(plugin_config(1.4, 1.6), BASE), {"evidence": {"ndvi_method": "stretch"}}))
    assert settings.evidence.combine == "best_per_block"  # the packaged default
    result = run(settings, assemble_inputs(layers, [Block(7, polygon, "test")]), cache_dir=cache, log=lambda *_: None)
    block = result.report["blocks"][0]
    assert block["evidence_component"] == "ndvi"
    assert block["component_scores"]["ndvi"] > block["component_scores"]["rgb"] + 0.1
    _assert_every_true_row_found_once(result, polygon)
    weighted = RunSettings.from_config(merge(merge(plugin_config(1.4, 1.6), BASE), {"evidence": {"combine": "weighted", "ndvi_method": "stretch"}}))
    mixed = run(weighted, assemble_inputs(layers, [Block(7, polygon, "test")]), cache_dir=cache, log=lambda *_: None)
    assert mixed.report["blocks"][0]["evidence_component"] == "weighted" and mixed.report["blocks"][0]["component_scores"] is None


def test_a_component_without_contrast_and_a_polygon_without_data_are_left_out(cache):
    """RGB is one flat value (no contrast), and polygon 8 lies where NDVI has no data: the run goes on without them."""
    grid, V = _scene(patch=False)
    layers = _layers(grid, V, ("rgb", "ndvi"))
    layers["rgb"].data[:] = 150.0
    side = NAN_CORNER_PX * grid.a
    no_data = Polygon([(grid.c + 0.2 * side, grid.f - 0.2 * side), (grid.c + 0.8 * side, grid.f - 0.2 * side),
                       (grid.c + 0.8 * side, grid.f - 0.8 * side), (grid.c + 0.2 * side, grid.f - 0.8 * side)])
    polygon = _block(grid)
    settings = RunSettings.from_config(merge(merge(plugin_config(1.4, 1.6), BASE), {"evidence": {"ndvi_method": "stretch"}}))
    messages = []
    result = run(settings, assemble_inputs(layers, [Block(7, polygon, "test"), Block(8, no_data, "test")]), cache_dir=cache,
                 log=messages.append)
    assert "left_out" in result.report["evidence"]["all_blocks"]["rgb"]
    reports = {b["block_id"]: b for b in result.report["blocks"]}
    assert reports[7]["evidence_component"] == "ndvi" and reports[7]["component_scores"]["rgb"] is None
    _assert_every_true_row_found_once(result, polygon)
    assert reports[8]["no_rows"] == "no data inside the polygon"
    assert [b.sections for b in result.blocks if b.block.id == 8] == [[]]
    assert any(m.startswith("  WARNING: block 8") for m in messages)


def test_a_row_width_range_that_misses_the_rows_is_warned_about(cache):
    """Rows 1.5 m apart searched at 1.8-2.2 m: most centres pick the smallest width, and the log says so."""
    grid, V = _scene(patch=False)
    settings = RunSettings.from_config(merge(merge(plugin_config(1.8, 2.2), BASE), {"spacing": {"nominal_m": 2.0, "step_m": 0.1}}))
    messages = []
    result = run(settings, assemble_inputs(_layers(grid, V, ("rgb",)), [Block(7, _block(grid), "test")]), cache_dir=cache,
                 log=messages.append)
    assert result.report["blocks"][0]["width_at_range_edge"]["min"] > 0.5
    warning = next(m for m in messages if "WARNING" in m and "smallest row width" in m)
    assert "lower its minimum" in warning and "1.80 m" in warning


def test_a_matching_range_gives_no_width_warning(run_rgb):
    edge = run_rgb[0].report["blocks"][0]["width_at_range_edge"]
    assert edge["min"] < 0.25 and edge["max"] < 0.25


def test_long_rows_have_the_true_direction(run_rgb):
    result, _ = run_rgb
    for s in result.sections:
        if s.length_m > 10:
            (x0, y0), (x1, y1) = s.geometry.coords[0], s.geometry.coords[-1]
            tolerance = 2.0 if s.length_m < 30 else 1.0  # end-to-end angle; short rows amplify end offsets
            assert axial_difference(np.degrees(np.arctan2(y1 - y0, x1 - x0)), SYNTHETIC_THETA) < tolerance


def test_only_corner_stubs_need_review(run_rgb):
    result, _ = run_rgb
    issues = result.report["totals"]["issues"]
    assert issues["completion"] == 0 and issues["cross_row"] == 0
    for r in result.reviews:
        if r["review"]:
            # Rows shorter than [review] min_length_m (4 m) at the block corners are flagged as short.
            assert r["length_m"] < 4.0 and r["issues"].startswith("short"), r["issues"]


def test_needs_review_is_the_review_flag(run_rgb):
    result, _ = run_rgb
    flags = [row_record(r)["needs_review"] for r in result.reviews]
    assert flags == [bool(r["review"]) for r in result.reviews] and any(flags) and not all(flags)


def test_faint_patch_is_bridged_and_reported(run_rgb):
    result, _ = run_rgb
    patch = _patch_polygon(synthetic_grid())
    by_id = {r["id"]: r for r in result.reviews}
    crossing = [s for s in result.sections if s.geometry.crosses(patch)]
    assert len(crossing) >= 8
    for s in crossing:
        r = by_id[s.id]
        assert 4.0 <= r["bridged_m"] <= 8.0, r  # the patch is 6 m long along the rows
        assert r["conf"] == "medium" and "bridged" in r["issues"] and r["review"] == 0


def test_block_polygon_has_a_vertex_at_every_row_end(run_rgb):
    result, block = run_rgb
    b = result.blocks[0]
    assert b.block.id == 7 and b.polygon.symmetric_difference(block).area < 1e-6  # shape unchanged
    vertices = np.array(b.polygon.exterior.coords)
    ends = [c for s in b.sections for c in (s.geometry.coords[0], s.geometry.coords[-1])]
    for e in ends:
        assert np.min(np.hypot(*(vertices - np.array(e)).T)) < 1e-9


def test_block_statistics_match_the_truth(run_numbering):
    result, _ = run_numbering
    s = result.blocks[0].statistics
    assert s["average_row_direction"] == pytest.approx(azimuth_deg(SYNTHETIC_THETA), abs=0.5)  # 60 deg from north
    assert s["average_row_width"] == pytest.approx(SYNTHETIC_SPACING, abs=0.02)
    assert 0 < s["confidence_interval"] < 0.02
    lengths = [sec.length_m for sec in result.sections]
    assert s["average_row_length"] == pytest.approx(np.mean(lengths), abs=0.01)
    assert s["rows_in_width"] == s["rows"] - 1  # the estimated row is left out of the width


def test_hanging_rows_are_kept_when_deletion_is_off(run_rgb):
    assert run_rgb[0].blocks[0].report["validation"]["removed_hanging"] == 0


def test_drop_hanging_rows_keeps_only_rows_meeting_the_boundary_twice(run_dropping):
    from vineliner.core.validation import meets_boundary_twice

    result, block = run_dropping
    v = result.blocks[0].report["validation"]
    assert v["removed_hanging"] >= 8  # the pieces either side of the faint patch
    assert len(result.sections) >= 20  # complete rows elsewhere are kept
    assert result.report["totals"]["issues"]["completion"] == 0
    for s in result.sections:
        assert meets_boundary_twice(s.geometry, block, 0.3)


def test_weak_centroid_box_is_backed_up_by_a_neighbour(run_pretest_bad_centre):
    result, _ = run_pretest_bad_centre
    report = result.report["blocks"][0]
    pt = report["centroid_pretest"]
    assert report["search_mode"] == "pretest" and pt["ring"] >= 1 and pt["boxes_tested"] > 1
    assert abs(pt["theta_deg"] - SYNTHETIC_THETA) <= 1
    assert len(result.sections) >= 30


def test_without_neighbours_a_weak_centroid_falls_back(run_pretest_no_neighbours):
    report = run_pretest_no_neighbours[0].report["blocks"][0]
    assert report["search_mode"] == "full" and "accept_min" in report["centroid_pretest"]["reason"]


def test_pretest_search_finds_the_same_rows_with_fewer_scores(run_rgb, run_pretest):
    full, _ = run_rgb
    pt, block = run_pretest
    report = pt.report["blocks"][0]
    assert report["search_mode"] == "pretest" and report["centroid_pretest"]["ring"] == 0
    assert abs(report["centroid_pretest"]["theta_deg"] - SYNTHETIC_THETA) <= 1
    assert report["hypotheses_per_centre"] < full.report["blocks"][0]["hypotheses_per_centre"] / 2
    for line in _true_rows(block):
        mid = line.interpolate(0.5, normalized=True)
        matches = [s for s in pt.sections if s.geometry.distance(mid) < 0.3]
        assert len(matches) == 1 and matches[0].geometry.hausdorff_distance(line) < 0.5


def test_kernels_are_cached(run_rgb, cache):
    assert run_rgb[0].report["kernels"]["cache"]["cache"] in ("hit", "miss (built and saved)")
    assert list(cache.glob("kernels_*.npz"))


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


def test_without_the_estimate_the_erased_row_stays_missing(run_dropping):
    assert run_dropping[0].report["totals"]["rows_estimated"] == 0


def test_final_rows_have_no_turn_above_the_smoothing_limit(run_rgb, run_row_estimate):
    from vineliner.core.reconstruction import turn_angles

    for result, _ in (run_rgb, run_row_estimate):
        for s in result.sections:
            coords = np.asarray(s.geometry.coords)
            if len(coords) > 2:
                assert turn_angles(coords).max() <= 20.0 + 1e-6


def test_rows_are_numbered_without_gaps_once_the_missing_row_is_estimated(run_numbering):
    result, _ = run_numbering
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


def test_report_is_json_ready_and_records_the_settings(run_rgb):
    report = clean_json(run_rgb[0].report)
    assert report["settings"]["spacing"]["min_m"] == 1.4 and report["totals"]["rows"] == len(run_rgb[0].sections)


def test_canceling_stops_the_run(cache):
    from vineliner.core.pipeline import Canceled

    grid, V = _scene()
    settings = RunSettings.from_config(merge(merge(plugin_config(1.4, 1.6), BASE), {}))
    inputs = assemble_inputs(_layers(grid, V, ("rgb",)), [Block(7, _block(grid), "test")])
    calls = []

    def canceled():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(Canceled):
        run(settings, inputs, cache_dir=cache, log=lambda *_: None, is_canceled=canceled)


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
