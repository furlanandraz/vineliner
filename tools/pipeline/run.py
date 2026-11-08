"""End-to-end run (PROJECT.md §1).

    inputs -> valid mask -> blocks -> row evidence -> kernels -> local search
    -> seeds -> chains -> growth/joins -> clipped sections -> validation -> outputs

Everything a run produces goes to runs/<run-id>/, next to the exact
configuration it used.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

from .. import __version__
from ..config import DEFAULT_CONFIG, PROJECT_ROOT, RunSettings, read_toml
from ..export import (
    block_records,
    bridge_layer,
    insert_vertices,
    issue_layer,
    open_choices,
    seed_geometry,
    write_outputs,
    write_overlay,
    write_report,
)
from ..geo import Grid
from ..imagery import combine, gray_from_bands, gray_response, ndvi_response, scale_response, valid_mask
from ..io import read_polygons, read_raster, write_raster
from ..reconstruction import Grower, build_chains, finalise_rows, link_seeds, rows_from_chains, seeds_from_selection
from ..search import (
    RIDGE_FIELDS,
    LocalSearch,
    PointScorer,
    SearchSpace,
    SurfaceStore,
    run_pretest,
    search_centres,
    select_best,
    select_orientation_consensus,
    window_edge_share,
)
from ..template import Template, load_or_build
from ..validation import CHECKS, check_block, evidence_along, mean_evidence, review_section, seed_rms, split_hanging
from ..vineyards import block_mask, make_blocks, pixel_window


EDGE_WARNING_SHARE = 0.25  # pre-test window-edge share above which a block is reported as suspect


@dataclass
class RunResult:
    run_dir: Path
    report: dict
    sections: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    reviews: list = field(default_factory=list)  # row attribute records, parallel to sections


@dataclass
class _Inputs:
    settings: RunSettings
    rasters: list
    grid: Grid
    pixel_m: float
    valid: np.ndarray
    blocks: list
    masks: dict
    polygon_provenance: dict
    ndvi: object = None  # Raster or None


def make_run_dir(root: Path | None = None, name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (root or PROJECT_ROOT / "runs") / (f"{stamp}-{name}" if name else stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_inputs(settings: RunSettings) -> _Inputs:
    s = settings.input
    rgb = read_raster(s.rgb, s.rgb_bands)
    rasters = [rgb]
    if s.nir is not None:
        nir = read_raster(s.nir, s.nir_bands)
        if not nir.grid.same_grid(rgb.grid):
            raise ValueError("RGB and NIR rasters are on different grids; align them explicitly first (PROJECT.md §3)")
        if nir.grid.crs_wkt != rgb.grid.crs_wkt:
            raise ValueError("RGB and NIR rasters have different CRS definitions")
        rasters.append(nir)
    grid = rgb.grid
    grid.require_north_up()
    pixel_m = grid.require_square_pixels()
    nodata = s.nodata_value if s.nodata_value is not None else rgb.nodata
    valid = valid_mask([r.data for r in rasters], nodata, s.nodata_rule)
    ndvi = None
    if s.ndvi is not None:
        ndvi = read_raster(s.ndvi, ("ndvi",))
        if not ndvi.grid.same_grid(rgb.grid) or ndvi.grid.crs_wkt != rgb.grid.crs_wkt:
            raise ValueError("the NDVI raster is not on the RGB grid; align it explicitly first (PROJECT.md §3)")
        band = ndvi.data[0]
        ok = np.isfinite(band) & (band >= -1.0) & (band <= 1.0)  # NDVI is in [-1, 1]
        if ndvi.nodata is not None:
            ok &= band != ndvi.nodata
        valid &= ok

    v = settings.vineyards
    ids, polygons, provenance, records = read_polygons(v.source, v.layer, v.id_field, grid.crs_wkt)
    blocks = make_blocks(ids, polygons, source="file", records=records)
    if v.blocks is not None:
        unknown = set(v.blocks) - {b.id for b in blocks}
        if unknown:
            raise ValueError(f"[vineyards] blocks {sorted(unknown)} are not in {v.source}")
        blocks = [b for b in blocks if b.id in set(v.blocks)]
    provenance["note"] = "supplied polygons used as a test fixture, not automatic classification (PROJECT.md §4)"
    masks = {b.id: block_mask(b, grid) for b in blocks}
    return _Inputs(settings, rasters, grid, pixel_m, valid, blocks, masks, provenance, ndvi)


class _Evidence:
    """Row-evidence responses computed once, scaled per block or over all blocks."""

    def __init__(self, settings: RunSettings, inputs: _Inputs):
        ev = settings.evidence
        self.ev = ev
        self.valid = inputs.valid
        gray = gray_from_bands([_band(inputs.rasters, name) for name in ev.bands])
        self.gray = gray_response(gray, inputs.valid, ev.method, inputs.pixel_m, ev.tophat_size_m)
        self.ndvi = None
        if ev.ndvi_weight > 0:
            self.ndvi = ndvi_response(inputs.ndvi.data[0], inputs.valid, ev.ndvi_method, inputs.pixel_m, ev.ndvi_tophat_size_m)

    def for_region(self, region: np.ndarray) -> tuple[np.ndarray, dict]:
        ev = self.ev
        stats_px = region & self.valid
        v_gray, s_gray = scale_response(self.gray, stats_px, ev.scale_percentile, ev.stretch_percentiles)
        components = [(v_gray, 1.0 - ev.ndvi_weight)]
        info = {"gray": s_gray}
        if self.ndvi is not None:
            v_ndvi, s_ndvi = scale_response(self.ndvi, stats_px, ev.scale_percentile, ev.stretch_percentiles)
            components.append((v_ndvi, ev.ndvi_weight))
            info["ndvi"] = s_ndvi
        V = combine(components, self.valid)
        info["mean_in_region"] = float(V[stats_px].mean())
        return V, info

    def describe(self) -> dict:
        ev = self.ev
        out = {
            "polarity": "high = row",
            "gray": self.gray.info | {"bands": list(ev.bands), "weight": 1.0 - ev.ndvi_weight},
            "scale_region": ev.scale_region,
            "scale_percentile": ev.scale_percentile,
            "stretch_percentiles": list(ev.stretch_percentiles),
        }
        if self.ndvi is not None:
            out["ndvi"] = self.ndvi.info | {"weight": ev.ndvi_weight}
        return out


def build_search(settings: RunSettings, pixel_m: float):
    t = settings.template
    template = Template(t.indices, t.weights, t.line_length_m, t.band_width_px * pixel_m)
    space = SearchSpace(settings.rotation.angles(), settings.spacing.values(), pixel_m, pixel_m, settings.translation.box, settings.spacing_nominal_m)
    return template, space


def plan_run(config: dict) -> dict:
    """What a run would search, without searching: centres, hypotheses, storage."""
    settings = RunSettings.from_config(config)
    inputs = load_inputs(settings)
    template, space = build_search(settings, inputs.pixel_m)
    store = SurfaceStore(Path("."), space, "none")
    blocks = {}
    total = 0
    for b in inputs.blocks:
        rows, _ = search_centres(b.polygon, inputs.grid, inputs.valid & inputs.masks[b.id], settings.centre_spacing_m)
        blocks[b.id] = {"area_m2": round(b.area_m2, 1), "centres": int(rows.size)}
        total += rows.size
    sizes = {}
    for mode in ("all", "winning_spacing"):
        store.mode = mode
        sizes[mode] = store.estimate_bytes(total)
    return {
        "blocks": blocks,
        "centres": int(total),
        "search": space.describe(),
        "template_scores": int(total * space.hypotheses_per_centre()),
        "surface_storage_bytes_uncompressed": sizes,
        "storage_mode": settings.store_surfaces,
    }


def run_pipeline(config: dict, run_dir: Path | None = None, name: str | None = None, log=print, batch_size: int = 64) -> RunResult:
    t_run = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(key: str, t0: float) -> float:
        timings[key] = round(timings.get(key, 0.0) + time.perf_counter() - t0, 3)
        return time.perf_counter()

    t0 = time.perf_counter()
    settings = RunSettings.from_config(config)
    run_dir = run_dir or make_run_dir(name=name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    log(f"Run directory: {run_dir}")

    inputs = load_inputs(settings)
    grid, valid, blocks = inputs.grid, inputs.valid, inputs.blocks
    log(f"Inputs: {grid.width} x {grid.height} px at {inputs.pixel_m} m; {len(blocks)} blocks; {int((~valid).sum())} invalid pixels")
    t0 = lap("inputs", t0)

    region = np.zeros(grid.shape, dtype=bool)
    for b in blocks:
        region |= inputs.masks[b.id]
    evidence = _Evidence(settings, inputs)
    evidence_info = evidence.describe()
    V_all, evidence_info["all_blocks"] = evidence.for_region(region)
    V_out = np.zeros(grid.shape, dtype=np.float32)  # evidence actually used, per block
    t0 = lap("evidence", t0)

    template, space = build_search(settings, inputs.pixel_m)
    log(f"Kernels: {len(space.thetas) * len(space.spacings) * len(template.indices)} needed ...")
    bank, kernel_cache = load_or_build(
        template, space.thetas, space.spacings, inputs.pixel_m, inputs.pixel_m,
        settings.kernels.coverage, settings.kernels.subpixel_samples, settings.kernels.cache_dir,
    )
    log(f"Kernels: {kernel_cache['cache']} ({kernel_cache['seconds']:.1f} s)")
    search = LocalSearch(bank, space, settings.scoring, template.weights, template.centre_position)
    store = SurfaceStore(run_dir / "scores", space, settings.store_surfaces)
    t0 = lap("kernels", t0)

    all_centres, all_seeds, all_chains, all_sections, all_issues, all_reviews, all_joins = [], [], [], [], [], [], []
    all_removed, all_removed_reviews = [], []  # hanging rows, when [export] drop_hanging_rows is on
    all_pretests = []  # when [centroid_pretest] is enabled
    block_reports = []
    centre_profiles = {"centre_id": [], "theta_profile": [], "spacing_profile": []}
    next_id = {"centre": 0, "seed": 0, "chain": 0, "row": 0, "section": 0}

    for block in blocks:
        tb = time.perf_counter()
        M = valid & inputs.masks[block.id]
        r0, c0, h, w = pixel_window(block, grid, search.radius + 2)
        wgrid = grid.window(r0, c0, h, w)
        if settings.evidence.scale_region == "per_block":
            V, block_scaling = evidence.for_region(inputs.masks[block.id])
        else:
            V, block_scaling = V_all, evidence_info["all_blocks"]
        V_out[M] = V[M]
        mv = _crop(V * M, r0, c0, h, w)
        m = _crop(M.astype(np.float32), r0, c0, h, w)

        rows, cols = search_centres(block.polygon, grid, M, settings.centre_spacing_m)
        centre_ids = np.arange(next_id["centre"], next_id["centre"] + rows.size)
        next_id["centre"] += rows.size
        log(f"Block {block.id}: {block.area_m2:.0f} m2, {rows.size} centres")

        def progress(done, total, _id=block.id):
            if done == total or done % (batch_size * 8) == 0:
                log(f"  block {_id}: searched {done}/{total} centres")

        block_search, pretest = search, None
        if settings.centroid_pretest.enabled:
            cp = settings.centroid_pretest
            pretest = run_pretest(
                block, grid, M, search, mv, m, (r0, c0), settings.scoring.accept_min,
                cp.theta_window_deg, cp.spacing_window_frac, settings.centre_spacing_m, cp.neighbour_rings,
            )
            all_pretests.append(pretest)
            if pretest.mode == "pretest":
                block_search = search.restricted(pretest.theta_idx, pretest.spacing_idx)
                where = "centroid box" if pretest.ring == 0 else f"ring-{pretest.ring} box ({pretest.boxes_tested} boxes tested)"
                log(
                    f"  block {block.id}: pre-test from the {where}: {pretest.theta_deg:.0f} deg, {pretest.spacing_m:.2f} m "
                    f"(S = {pretest.score:.2f}); grid searches {block_search.n_theta} rotations x {len(block_search.spacing_idx)} spacings"
                )
            else:
                log(f"  block {block.id}: full grid search ({pretest.reason})")

        results = block_search.run(mv, m, rows - r0, cols - c0, batch_size, store.sink(block.id, centre_ids), progress)
        results.rows, results.cols = rows, cols
        t_search = time.perf_counter() - tb

        rs = settings.reconstruction
        accept_min = settings.scoring.accept_min
        cx, cy = grid.array_to_world(rows, cols)
        if settings.selection.method == "orientation_consensus":
            selection = select_orientation_consensus(
                results, cx, cy, space.thetas, accept_min, settings.selection.consensus_radius_m, settings.selection.consensus_window_deg
            )
        else:
            selection = select_best(results)
        seeds = seeds_from_selection(
            selection, rows, cols, centre_ids, grid, space, block.id, template.indices, settings.seed_lines,
            template.line_length_m, accept_min, settings.scoring.min_valid_coverage, settings.seed_min_band_mean,
            first_id=next_id["seed"],
        )
        next_id["seed"] += len(seeds)
        labels = link_seeds(seeds, rs.link_angle_deg, rs.link_lateral_frac, rs.link_along_m)
        chains = build_chains(seeds, labels, rs.growth_step_m, rs.chain_smoothing_m, first_id=next_id["chain"])
        next_id["chain"] += len(chains)
        rows_ = rows_from_chains(chains, space.spacings, first_id=next_id["row"])
        next_id["row"] += len(rows_)
        scorer = PointScorer(bank, wgrid, mv, m, template.weights, template.centre_position, settings.scoring)
        grower = Grower(rows_, block.polygon, scorer, rs, template.line_length_m)
        grown = grower.run()
        sections = finalise_rows(grown, block.polygon, settings.validation.tolerance_m, first_id=next_id["section"])
        next_id["section"] += len(sections)
        seeds_by_id = {s.id: s for s in seeds}
        along = {sec.id: evidence_along(sec, scorer) for sec in sections}
        ev_by_id = {sid: mean_evidence(ev) for sid, ev in along.items()}

        def review_all(secs, stats_by_id):
            ev = {s.id: ev_by_id[s.id] for s in secs}
            return [
                review_section(sec, stats_by_id[sec.id], along[sec.id], seed_rms(sec, seeds_by_id), rs.growth_min_score, settings.review, ev)
                for sec in secs
            ]

        removed, removed_reviews = [], []
        if settings.export.drop_hanging_rows:
            # Removed rows keep the attributes they had among all rows, so it is clear why they hung.
            _, stats_all, _ = check_block(sections, block.polygon, settings.validation, block.id)
            sections, removed = split_hanging(sections, block.polygon, settings.validation.tolerance_m)
            removed_reviews = review_all(removed, stats_all)
        issues, stats, vsummary = check_block(sections, block.polygon, settings.validation, block.id)
        reviews = review_all(sections, stats)
        vsummary["removed_hanging"] = len(removed)
        vsummary["removed_hanging_m"] = round(sum(s.length_m for s in removed), 1)
        vsummary["review"] = sum(r["review"] for r in reviews)
        vsummary["conf"] = {c: sum(r["conf"] == c for r in reviews) for c in ("high", "medium", "low")}
        vsummary["bridged_m"] = round(sum(sec.bridged_m for sec in sections), 1)

        raw_accepted = int(np.sum(np.nan_to_num(results.best_score, nan=-1) >= accept_min))
        sel_ok = selection.accepted(accept_min)
        accepted = int(sel_ok.sum())
        endpoint_states: dict[str, int] = {}
        for sec in sections:
            for st in (sec.start_state, sec.end_state):
                endpoint_states[st] = endpoint_states.get(st, 0) + 1
        changed = int(np.sum(sel_ok & (selection.theta != results.best_theta)))
        block_reports.append({
            "block_id": block.id,
            "area_m2": round(block.area_m2, 1),
            "evidence_scaling": block_scaling,
            "centres": int(rows.size),
            "accepted_centres_raw": raw_accepted,
            "accepted_centres": accepted,
            "selection": selection.method,
            "selection_changed_theta": changed,
            "search_mode": pretest.mode if pretest else "full",
            "centroid_pretest": pretest.describe(space.thetas, space.spacings) if pretest else None,
            "pretest_window_edge_share": (
                window_edge_share(np.asarray(space.thetas)[selection.theta[sel_ok]], pretest, space.thetas, settings.rotation.step_deg)
                if pretest and sel_ok.any() else None
            ),
            "hypotheses_per_centre": block_search.hypotheses_per_centre,
            "theta_median_deg": float(np.median(np.asarray(space.thetas)[selection.theta[sel_ok]])) if sel_ok.any() else None,
            "spacing_median_m": float(np.median(np.asarray(space.spacings)[selection.spacing[sel_ok]])) if sel_ok.any() else None,
            "seeds": len(seeds),
            "chains": len(chains),
            "chains_flagged_lateral_spread": sum("lateral_spread" in c.flags for c in chains),
            "rows_after_growth": len(grown),
            "joins": len(grower.joins),
            "growth_rounds": grower.rounds,
            "sections": len(sections),
            "endpoint_states": endpoint_states,
            "validation": vsummary,
            "seconds": {"search": round(t_search, 2), "total": round(time.perf_counter() - tb, 2)},
        })
        log(
            f"  block {block.id}: {accepted}/{rows.size} centres accepted ({changed} re-oriented), {len(seeds)} seeds, {len(chains)} chains, "
            f"{len(grower.joins)} joins -> {len(sections)} rows ({vsummary['conf']['high']} high, {vsummary['conf']['medium']} medium, "
            f"{vsummary['conf']['low']} low confidence)"
            + (f", {len(removed)} hanging rows removed" if settings.export.drop_hanging_rows else "")
            + f" ({time.perf_counter() - tb:.1f} s)"
        )

        edge = block_reports[-1]["pretest_window_edge_share"]
        if edge is not None and edge > EDGE_WARNING_SHARE:
            log(
                f"  block {block.id}: WARNING {100 * edge:.0f}% of accepted centres chose a direction on the edge of the "
                f"pre-test rotation window; the pre-test may be off. Widen [centroid_pretest] theta_window_deg or turn it off."
            )

        for i in range(rows.size):
            all_centres.append(_centre_record(results, selection, i, int(centre_ids[i]), block.id, space, template, accept_min, cx[i], cy[i]))
        centre_profiles["centre_id"].append(centre_ids)
        centre_profiles["theta_profile"].append(results.theta_profile)
        centre_profiles["spacing_profile"].append(results.spacing_profile)
        all_seeds += seeds
        all_chains += chains
        all_sections += sections
        all_issues += issues
        all_reviews += reviews
        all_removed += removed
        all_removed_reviews += removed_reviews
        all_joins += grower.joins
    t0 = lap("blocks", t0)

    write_raster(run_dir / "evidence.tif", V_out, grid)
    write_raster(run_dir / "valid.tif", (valid & region).astype(np.uint8), grid)
    store.write_manifest()
    if centre_profiles["centre_id"]:
        np.savez_compressed(
            run_dir / "centre_profiles.npz",
            centre_id=np.concatenate(centre_profiles["centre_id"]),
            theta_profile=np.concatenate(centre_profiles["theta_profile"]),
            spacing_profile=np.concatenate(centre_profiles["spacing_profile"]),
            thetas_deg=np.asarray(space.thetas),
            spacings_m=np.asarray(space.spacings),
        )

    # Products for review: rows with attributes, blocks with a vertex at every row end, issues, bridges.
    row_layer = (all_reviews, [s.geometry for s in all_sections])
    row_ends = {b.id: [p for s in all_sections if s.block_id == b.id for p in s.boundary_points()] for b in blocks}
    block_layer = (block_records(blocks, block_reports, all_sections, all_reviews), [insert_vertices(b.polygon, row_ends[b.id]) for b in blocks])
    issues_layer = issue_layer(all_issues)
    bridges_layer = bridge_layer(all_sections)
    shapefiles = {"rows": row_layer, "blocks": block_layer, "issues": issues_layer, "bridges": bridges_layer}
    if settings.export.drop_hanging_rows:
        shapefiles["removed"] = (all_removed_reviews, [s.geometry for s in all_removed])
    gpkg_layers = dict(shapefiles) | _diagnostic_layers(all_centres, all_seeds, all_chains, all_sections, all_joins)
    gpkg_layers["pretests"] = (  # the box each block's pre-test came from
        [{"block": p.block_id, "mode": p.mode, "ring": p.ring, "tested": p.boxes_tested, "reason": p.reason,
          "theta_deg": p.theta_deg, "spacing_m": p.spacing_m, "score": p.score}
         for p in all_pretests if p.x is not None],
        [Point(p.x, p.y) for p in all_pretests if p.x is not None],
    )
    written = write_outputs(run_dir, grid.crs_wkt, gpkg_layers, shapefiles, settings.export.shapefile, settings.export.geopackage)
    if settings.export.overlay:
        write_overlay(run_dir / "overlay.png", inputs.rasters[0].data, grid, blocks, all_sections, all_reviews, all_issues, removed=all_removed)
    t0 = lap("export", t0)
    timings["total"] = round(time.perf_counter() - t_run, 3)

    report = {
        "vineliner_version": __version__,
        "run_dir": str(run_dir),
        "created": datetime.now().isoformat(timespec="seconds"),
        "open_choices_set_by_this_run": open_choices(read_toml(DEFAULT_CONFIG), config),
        "inputs": {
            "rasters": [r.metadata | {"band_names": list(r.band_names)} for r in inputs.rasters + ([inputs.ndvi] if inputs.ndvi else [])],
            "invalid_pixels": int((~valid).sum()),
            "polygons": inputs.polygon_provenance,
        },
        "evidence": evidence_info,
        "kernels": bank.provenance() | {"cache": kernel_cache},
        "search": space.describe() | {"centre_spacing_m": settings.centre_spacing_m, "sparse_nnz": search.nnz, "window_radius_px": search.radius},
        "storage": {"mode": store.mode, "files": len(store.files), "bytes_on_disk": store.bytes_written},
        "blocks": block_reports,
        "totals": {
            "centres": len(all_centres),
            "seeds": len(all_seeds),
            "chains": len(all_chains),
            "rows": len(all_sections),
            "rows_for_review": sum(r["review"] for r in all_reviews),
            "conf": {c: sum(r["conf"] == c for r in all_reviews) for c in ("high", "medium", "low")},
            "row_length_m": round(sum(s.length_m for s in all_sections), 1),
            "length_by_conf_m": {c: round(sum(r["length_m"] for r in all_reviews if r["conf"] == c), 1) for c in ("high", "medium", "low")},
            "bridged_m": round(sum(s.bridged_m for s in all_sections), 1),
            "dangling_ends": sum(r["dangling"] for r in all_reviews),
            "drop_hanging_rows": settings.export.drop_hanging_rows,
            "removed_hanging": len(all_removed),
            "removed_hanging_m": round(sum(s.length_m for s in all_removed), 1),
            "centroid_pretest_enabled": settings.centroid_pretest.enabled,
            "blocks_by_search_mode": {m: sum(b["search_mode"] == m for b in block_reports) for m in ("pretest", "full")},
            "template_scores": int(sum(b["centres"] * b["hypotheses_per_centre"] for b in block_reports)),
            "template_scores_full_search": int(len(all_centres) * space.hypotheses_per_centre()),
            "crossings": sum(1 for f in all_issues if f.check == "cross_row"),
            "issues": {c: sum(b["validation"]["issues"][c] for b in block_reports) for c in CHECKS},
        },
        "outputs": written,
        "timings_s": timings,
        "caveats": [
            "Scores are overlap/matching scores, not probabilities (PROJECT.md §8).",
            "Checks are reported, never enforced; review = 1 marks rows for a human (PROJECT.md §11).",
            "Passing the checks does not prove rows were read correctly; missing rows can pass (PROJECT.md §11).",
            "Block polygons are a supplied fixture, not automatic classification (PROJECT.md §4).",
        ],
    }
    write_report(run_dir / "report.json", report)
    t = report["totals"]
    log(
        f"Done in {timings['total']:.0f} s: {t['rows']} rows, {t['row_length_m'] / 1000:.1f} km "
        f"({t['conf']['high']} high, {t['conf']['medium']} medium, {t['conf']['low']} low confidence; {t['rows_for_review']} for review)"
    )
    return RunResult(run_dir=run_dir, report=report, sections=all_sections, issues=all_issues, reviews=all_reviews)


def _band(rasters, name: str) -> np.ndarray:
    """Band by meaning. The RGB file is searched first, then the NIR file."""
    for r in rasters:
        if name in r.band_names:
            return r.band(name)
    raise ValueError(f"[evidence] band {name!r} is not in any input; known: {[r.band_names for r in rasters]}")


def _crop(array: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """Window of array starting at (r0, c0); parts outside the array are 0."""
    out = np.zeros((h, w), dtype=np.float32)
    rs, cs = max(r0, 0), max(c0, 0)
    re, ce = min(r0 + h, array.shape[0]), min(c0 + w, array.shape[1])
    if rs < re and cs < ce:
        out[rs - r0:re - r0, cs - c0:ce - c0] = array[rs:re, cs:ce]
    return out


def _centre_record(results, selection, i, centre_id, block_id, space, template, accept_min, x, y) -> dict:
    """Selected hypothesis (seeds come from it), plus the unconstrained winner as raw_*."""
    score = float(selection.score[i])
    has = np.isfinite(score)
    raw = float(results.best_score[i])
    raw_has = np.isfinite(raw)
    rec = {
        "id": centre_id,
        "block": block_id,
        "score": score if has else None,
        "accepted": int(has and score >= accept_min),
        "theta_deg": float(space.thetas[selection.theta[i]]) if has else None,
        "spacing_m": float(space.spacings[selection.spacing[i]]) if has else None,
        "ox_px": int(selection.offset[i, 0]),
        "oy_px": int(selection.offset[i, 1]),
        "consensus_deg": _num(selection.consensus_deg[i]),
        "raw_score": raw if raw_has else None,
        "raw_theta": float(space.thetas[results.best_theta[i]]) if raw_has else None,
        "raw_spacing": float(space.spacings[results.best_spacing[i]]) if raw_has else None,
        "raw_rank": int(results.best_rank[i]),
        "geometry": Point(x, y),
    }
    for b, k in enumerate(template.indices):
        rec[f"m_{k:+d}"] = _num(selection.means[i, b])
        rec[f"cov_{k:+d}"] = _num(selection.coverage[i, b])
    for f, value in zip(RIDGE_FIELDS, results.ridge[i]):
        rec[f"raw_ridge_{f}"] = _num(value)  # persistence of the unconstrained winner
    rec["compet_n"] = _num(results.competing[i, 0])
    rec["compet_theta"] = _num(results.competing[i, 1])
    rec["compet_score"] = _num(results.competing[i, 2])
    return rec


def _num(v) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


def _diagnostic_layers(centres, seeds, chains, sections, joins) -> dict:
    """GeoPackage-only layers for looking into how rows were made."""
    layers = {}
    layers["centres"] = ([{k: v for k, v in c.items() if k != "geometry"} for c in centres], [c["geometry"] for c in centres])
    layers["seeds"] = (
        [{
            "id": s.id, "block": s.block_id, "centre": s.centre_id, "line": s.line, "theta_deg": s.theta_deg,
            "spacing_m": s.spacing_m, "score": s.score, "band_mean": s.band_mean, "band_cov": s.band_coverage,
        } for s in seeds],
        [seed_geometry(s) for s in seeds],
    )
    layers["chains"] = (
        [{
            "id": c.id, "block": c.block_id, "n_seeds": len(c.seed_ids), "theta_deg": round(c.theta_deg, 2),
            "spacing_m": c.spacing_m, "spread_m": round(c.max_lateral_spread_m, 3), "flags": ",".join(c.flags),
        } for c in chains],
        [LineString(c.vertices) for c in chains],
    )
    endpoints, endpoint_geoms = [], []
    for s in sections:
        for side, coord, state in (("start", s.geometry.coords[0], s.start_state), ("end", s.geometry.coords[-1], s.end_state)):
            endpoints.append({"section": s.id, "block": s.block_id, "side": side, "state": state})
            endpoint_geoms.append(Point(coord))
    layers["endpoints"] = (endpoints, endpoint_geoms)
    layers["joins"] = ([{"row": j.row_id, "merged": j.merged_row_id} for j in joins], [Point(j.x, j.y) for j in joins])
    return layers
