"""End-to-end run on in-memory inputs (PROJECT.md §1).

    layers on one grid -> valid mask -> row evidence -> kernels -> local search
    -> seeds -> chains -> growth/joins -> clipped rows -> checks -> estimate
    -> review flags and numbering -> per-block statistics

run() reads nothing from disk and writes nothing (apart from the optional
kernel cache). The QGIS plugin turns its RunResult into output layers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from ..config import RunSettings
from ..export import block_statistics, insert_vertices
from ..imagery import bright_response, combine, gray_from_bands, gray_response, scale_response
from ..reconstruction import (
    Grower,
    build_chains,
    fill_gaps,
    finalise_rows,
    find_gaps,
    limit_turns,
    link_seeds,
    number_rows,
    row_width,
    rows_from_chains,
    seeds_from_selection,
    squash_pieces,
)
from ..search import (
    LocalSearch,
    box_scores,
    PointScorer,
    SearchSpace,
    run_pretest,
    search_centres,
    select_best,
    select_orientation_consensus,
    window_edge_share,
)
from ..template import Template, load_or_build
from ..validation import (
    CHECKS,
    CONF_CLASSES,
    check_block,
    evidence_along,
    mark_estimated,
    mark_numbered,
    mean_evidence,
    meets_boundary_twice,
    review_section,
    seed_rms,
    split_hanging,
)
from ..vineyards import pixel_window
from .inputs import LAYER_NAMES, InputError, Inputs

EDGE_WARNING_SHARE = 0.25  # pre-test window-edge share above which a block is reported as suspect
PROGRESS_KERNELS = 0.05  # share of the progress bar for building/loading kernels; the blocks share the rest


class Canceled(Exception):
    """The caller asked to stop (is_canceled() returned True)."""


@dataclass
class BlockResult:
    block: object  # vineyards.Block
    sections: list  # final rows (reconstruction.RowSection)
    reviews: list  # review records, parallel to sections; review = 1 means needs_review
    statistics: dict  # export.summary.block_statistics
    polygon: object  # the block polygon with a vertex at every row end
    report: dict


@dataclass
class RunResult:
    blocks: list[BlockResult]
    report: dict
    evidence: np.ndarray | None = None  # row evidence used, per block, on the analysis grid

    @property
    def sections(self) -> list:
        return [s for b in self.blocks for s in b.sections]

    @property
    def reviews(self) -> list:
        return [r for b in self.blocks for r in b.reviews]


def _empty_result(block, reason: str, centres: int = 0) -> BlockResult:
    """A block without rows: its polygon unchanged, empty statistics, and why in the report."""
    stats = block_statistics([], [], {})
    return BlockResult(block, [], [], stats, block.polygon,
                       {"block_id": block.id, "centres": centres, "no_rows": reason, "statistics": stats})


class _Evidence:
    """Row-evidence responses computed once per component, scaled per block or over all blocks."""

    def __init__(self, settings: RunSettings, inputs: Inputs):
        ev = settings.evidence
        self.ev = ev
        self.valid = inputs.valid
        self.weights = ev.weights_for(inputs.components)  # component -> share of V, summing to 1
        self.responses = {}
        for layer in self.weights:
            if layer == "rgb":
                gray = gray_from_bands([_band(inputs.band_rasters, name) for name in ev.bands])
                self.responses[layer] = gray_response(gray, inputs.valid, ev.method, inputs.pixel_m, ev.tophat_size_m)
            elif layer == "nir":
                nir = inputs.layers["nir"].band("nir")
                self.responses[layer] = bright_response(nir, inputs.valid, ev.nir_method, inputs.pixel_m, ev.nir_tophat_size_m, "nir")
            else:
                ndvi = inputs.layers["ndvi"].data[0]
                self.responses[layer] = bright_response(ndvi, inputs.valid, ev.ndvi_method, inputs.pixel_m, ev.ndvi_tophat_size_m, "ndvi")

    def for_region(self, region: np.ndarray, only: str | None = None) -> tuple[np.ndarray, dict]:
        """V scaled over region: the weighted mean of the components, or one component alone (only).

        A component that cannot be scaled there (no contrast) is left out and the
        others re-weighted; ValueError when none can be.
        """
        ev = self.ev
        stats_px = region & self.valid
        components, info, failures = [], {}, []
        for layer, resp in self.responses.items():
            if only is not None and layer != only:
                continue
            try:
                v, info[layer] = scale_response(resp, stats_px, ev.scale_percentile, ev.stretch_percentiles)
            except ValueError as exc:
                if only is not None:
                    raise
                info[layer], failures = {"left_out": str(exc)}, failures + [str(exc)]
                continue
            components.append((v, 1.0 if only is not None else self.weights[layer]))
        if not components:
            raise ValueError("; ".join(dict.fromkeys(failures)))
        total = sum(w for _, w in components)
        if total > 0 and failures:
            components = [(v, w / total) for v, w in components]
        V = combine(components, self.valid)
        info["mean_in_region"] = float(V[stats_px].mean()) if stats_px.any() else None
        return V, info

    def describe(self) -> dict:
        ev = self.ev
        out = {
            "polarity": "high = row",
            "weights": dict(self.weights),
            "scale_region": ev.scale_region,
            "scale_percentile": ev.scale_percentile,
            "stretch_percentiles": list(ev.stretch_percentiles),
        }
        for layer, resp in self.responses.items():
            out[layer] = resp.info | {"weight": self.weights[layer]} | ({"bands": list(ev.bands)} if layer == "rgb" else {})
        return out


def build_search(settings: RunSettings, pixel_m: float):
    t = settings.template
    template = Template(t.indices, t.weights, t.line_length_m, t.band_width_px * pixel_m)
    space = SearchSpace(settings.rotation.angles(), settings.spacing.values(), pixel_m, pixel_m, settings.translation.box, settings.spacing_nominal_m)
    return template, space


def run(
    settings: RunSettings,
    inputs: Inputs,
    cache_dir: str | Path | None = None,
    log=print,
    progress=None,
    is_canceled=None,
    batch_size: int = 64,
) -> RunResult:
    """Find the rows of every block.

    cache_dir: where kernels are cached when [kernels] cache is on (None: build every run).
    log(str): messages; progress(fraction in [0, 1]); is_canceled() -> bool, checked between search batches.
    """
    t_run = time.perf_counter()
    timings: dict[str, float] = {}
    progress = progress or (lambda f: None)

    def check_canceled():
        if is_canceled is not None and is_canceled():
            raise Canceled("canceled")

    def lap(key: str, t0: float) -> float:
        timings[key] = round(timings.get(key, 0.0) + time.perf_counter() - t0, 3)
        return time.perf_counter()

    t0 = time.perf_counter()
    for note in settings.notes:
        log(f"Note: {note}")
    grid, valid, blocks = inputs.grid, inputs.valid, inputs.blocks
    names = ", ".join(LAYER_NAMES[k] for k in inputs.components)
    log(f"Inputs: {names}{f' (NDVI {inputs.ndvi_source})' if inputs.ndvi_source else ''}; "
        f"{grid.width} x {grid.height} px at {inputs.pixel_m:g} m; {len(blocks)} blocks; {int((~valid).sum())} invalid pixels")

    region = np.zeros(grid.shape, dtype=bool)
    for b in blocks:
        region |= inputs.masks[b.id]
    evidence = _Evidence(settings, inputs)
    log("Evidence: " + ", ".join(f"{LAYER_NAMES[k]} {w:.2f}" for k, w in evidence.weights.items()))
    evidence_info = evidence.describe()
    try:
        V_all, evidence_info["all_blocks"] = evidence.for_region(region)
    except ValueError as exc:
        raise InputError(f"the images inside the vineyard polygons cannot be used: {exc}") from exc
    V_out = np.zeros(grid.shape, dtype=np.float32)  # evidence actually used, per block
    t0 = lap("evidence", t0)
    check_canceled()

    template, space = build_search(settings, inputs.pixel_m)
    log(f"Kernels: {len(space.thetas) * len(space.spacings) * len(template.indices)} needed ...")
    bank, kernel_cache = load_or_build(
        template, space.thetas, space.spacings, inputs.pixel_m, inputs.pixel_m,
        settings.kernels.coverage, settings.kernels.subpixel_samples,
        Path(cache_dir) if (settings.kernels.cache and cache_dir) else None,
    )
    log(f"Kernels: {kernel_cache['cache']} ({kernel_cache['seconds']:.1f} s)")
    search = LocalSearch(bank, space, settings.scoring, template.weights, template.centre_position)
    t0 = lap("kernels", t0)
    progress(PROGRESS_KERNELS)
    check_canceled()

    centres = {b.id: search_centres(b.polygon, grid, valid & inputs.masks[b.id], settings.centre_spacing_m) for b in blocks}
    total_centres = max(1, sum(r.size for r, _ in centres.values()))
    done_centres = 0
    results_out: list[BlockResult] = []
    next_id = {"seed": 0, "chain": 0, "row": 0, "section": 0}
    rs = settings.reconstruction
    accept_min = settings.scoring.accept_min
    max_turn = settings.smoothing.max_turn_deg if settings.smoothing.enabled else None

    for block in blocks:
        tb = time.perf_counter()
        check_canceled()
        M = valid & inputs.masks[block.id]
        if not M.any():
            log(f"  WARNING: block {block.id}: no input layer has data inside this polygon; no rows")
            results_out.append(_empty_result(block, "no data inside the polygon"))
            continue
        r0, c0, h, w = pixel_window(block, grid, search.radius + 2)
        wgrid = grid.window(r0, c0, h, w)
        m = _crop(M.astype(np.float32), r0, c0, h, w)
        scaling_region = inputs.masks[block.id] if settings.evidence.scale_region == "per_block" else region
        try:
            if settings.evidence.scale_region == "per_block":
                V, block_scaling = evidence.for_region(scaling_region)
            else:
                V, block_scaling = V_all, evidence_info["all_blocks"]
        except ValueError as exc:  # no contrast: a uniform image inside the polygon
            log(f"  WARNING: block {block.id}: {exc}; no rows")
            results_out.append(_empty_result(block, str(exc)))
            continue
        chosen, component_scores = None, None
        if settings.evidence.combine == "best_per_block" and len(evidence.responses) > 1:
            # Each component alone, scored at the same boxes around the centroid; the best one is the block's evidence.
            component_scores, candidates = {}, {}
            for layer in evidence.responses:
                try:
                    Vk, info_k = evidence.for_region(scaling_region, only=layer)
                except ValueError:
                    component_scores[layer] = None  # no contrast in this component: not a candidate
                    continue
                scores = box_scores(block, grid, M, search, _crop(Vk * M, r0, c0, h, w), m, (r0, c0), settings.centre_spacing_m)
                scores = scores[np.isfinite(scores)]
                component_scores[layer] = round(float(np.median(scores)), 4) if scores.size else None
                candidates[layer] = (Vk, info_k)
            scored = [k for k, v in component_scores.items() if v is not None]
            if scored:
                chosen = max(scored, key=component_scores.get)
                V, block_scaling = candidates[chosen]
                log(f"  block {block.id}: evidence from {LAYER_NAMES[chosen]} (median box score "
                    + ", ".join(f"{LAYER_NAMES[k]} {v:.2f}" for k, v in component_scores.items() if v is not None) + ")")
            check_canceled()
        V_out[M] = V[M]
        mv = _crop(V * M, r0, c0, h, w)
        rows, cols = centres[block.id]
        centre_ids = np.arange(rows.size)
        log(f"Block {block.id}: {block.area_m2:.0f} m2, {rows.size} centres")

        def search_progress(done, total, _start=done_centres):
            check_canceled()
            progress(PROGRESS_KERNELS + (1 - PROGRESS_KERNELS) * (_start + done) / total_centres)

        if rows.size == 0:
            log(f"  block {block.id}: no valid search centres inside the polygon; no rows")
            results_out.append(_empty_result(block, "no valid search centres inside the polygon"))
            continue

        block_search, pretest = search, None
        if settings.centroid_pretest.enabled:
            cp = settings.centroid_pretest
            pretest = run_pretest(
                block, grid, M, search, mv, m, (r0, c0), accept_min,
                cp.theta_window_deg, cp.spacing_window_frac, settings.centre_spacing_m, cp.neighbour_rings,
            )
            if pretest.mode == "pretest":
                block_search = search.restricted(pretest.theta_idx, pretest.spacing_idx)
                where = "centroid box" if pretest.ring == 0 else f"ring-{pretest.ring} box ({pretest.boxes_tested} boxes tested)"
                log(
                    f"  block {block.id}: pre-test from the {where}: {pretest.theta_deg:.0f} deg, {pretest.spacing_m:.2f} m "
                    f"(S = {pretest.score:.2f}); grid searches {block_search.n_theta} rotations x {len(block_search.spacing_idx)} widths"
                )
            else:
                log(f"  block {block.id}: full grid search ({pretest.reason})")
        check_canceled()

        results = block_search.run(mv, m, rows - r0, cols - c0, batch_size, None, search_progress)
        results.rows, results.cols = rows, cols
        done_centres += rows.size
        t_search = time.perf_counter() - tb

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
        smoothed_vertices = 0
        if settings.smoothing.enabled:
            for r in grown:
                before = len(r.points)
                points, r.seg_support = limit_turns(r.points, settings.smoothing.max_turn_deg, r.seg_support)
                r.points = list(points)
                smoothed_vertices += before - len(r.points)
        sections = finalise_rows(grown, block.polygon, settings.validation.tolerance_m, first_id=next_id["section"], max_turn_deg=max_turn)
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

        removed = []
        if settings.drop_hanging_rows:
            sections, removed = split_hanging(sections, block.polygon, settings.validation.tolerance_m)
        issues, stats, vsummary = check_block(sections, block.polygon, settings.validation, block.id)

        estimated, gaps, width_info, squashed = [], [], None, set()
        if settings.missing_row_estimate.enabled and sections:
            mre = settings.missing_row_estimate
            width, width_info = row_width(sections, stats, float(np.median([sec.spacing_m for sec in sections])))
            if width is not None:
                gaps = find_gaps(sections, width, mre.gap_tolerance_frac, mre.whole_row_tolerance_frac, mre.max_missing_rows, settings.validation.sample_step_m)
                estimated = fill_gaps(
                    gaps, sections, width, mre.station_factor, block.polygon, settings.validation.tolerance_m,
                    space.spacings, next_id["section"], meets_boundary_twice, max_turn,
                )
                next_id["section"] += len(estimated)
                if mre.merge_pieces and estimated and removed:
                    # Squash removed hanging pieces that lie on an estimated row into it.
                    estimated, squashed = squash_pieces(
                        estimated, removed, width, mre.merge_buffer_frac, mre.merge_min_inside_frac, mre.merge_max_angle_deg,
                        ev_by_id, block.polygon, settings.validation.tolerance_m, max_turn, meets_boundary_twice,
                    )
                    removed = [sec for sec in removed if sec.id not in squashed]
                for sec in estimated:
                    along[sec.id] = evidence_along(sec, scorer)
                    ev_by_id[sec.id] = mean_evidence(along[sec.id])
                sections = sections + estimated
                # Re-check with the estimated rows in place; neighbours of a filled gap are no longer too far apart.
                issues, stats, vsummary = check_block(sections, block.polygon, settings.validation, block.id)
        reviews = review_all(sections, stats)
        for rec, sec in zip(reviews, sections):
            if "estimated" in sec.flags:
                mark_estimated(rec, sec)
        numbering = None
        if settings.row_numbering.enabled and sections:
            # Number the final rows with the block's row width (from the estimate when it ran).
            d = width_info.get("row_width_m") if width_info else None
            if d is None:
                d, _ = row_width(sections, stats, float(np.median([sec.spacing_m for sec in sections])))
            if d is None:
                d = float(np.median([sec.spacing_m for sec in sections]))
            numbering = number_rows(sections, d, settings.row_numbering.transect_interval_m)
            for rec, sec in zip(reviews, sections):
                mark_numbered(rec, numbering.numbers[sec.id], numbering.ok[sec.id])

        block_stats = block_statistics(sections, reviews, stats)
        row_ends = [p for s in sections for p in s.boundary_points()]
        polygon = insert_vertices(block.polygon, row_ends)

        vsummary["row_numbering"] = numbering.describe() if numbering else None
        vsummary["missing_row_estimate"] = None if width_info is None else width_info | {
            "gaps_too_wide": len(gaps),
            "gaps_filled": sum(g.filled for g in gaps),
            "rows_added": len(estimated),
            "rows_with_squashed_pieces": sum(1 for sec in estimated if sec.detected_m),
            "pieces_squashed": len(squashed),
            "gaps_not_filled": [{"rows": [g.a, g.b], "gap_m": round(g.gap_m, 3), "reason": g.reason} for g in gaps if not g.filled],
        }
        vsummary["removed_hanging"] = len(removed)
        vsummary["removed_hanging_m"] = round(sum(s.length_m for s in removed), 1)
        vsummary["review"] = sum(r["review"] for r in reviews)
        vsummary["conf"] = {c: sum(r["conf"] == c for r in reviews) for c in CONF_CLASSES}
        vsummary["bridged_m"] = round(sum(sec.bridged_m for sec in sections), 1)

        sel_ok = selection.accepted(accept_min)
        accepted = int(sel_ok.sum())
        changed = int(np.sum(sel_ok & (selection.theta != results.best_theta)))
        width_edge = _width_edge_shares(np.asarray(space.spacings), selection.spacing[sel_ok])
        report = {
            "block_id": block.id,
            "area_m2": round(block.area_m2, 1),
            "evidence_scaling": block_scaling,
            "evidence_component": chosen or "weighted",
            "component_scores": component_scores,
            "centres": int(rows.size),
            "accepted_centres": accepted,
            "selection_changed_theta": changed,
            "search_mode": pretest.mode if pretest else "full",
            "centroid_pretest": pretest.describe(space.thetas, space.spacings) if pretest else None,
            "pretest_window_edge_share": (
                window_edge_share(np.asarray(space.thetas)[selection.theta[sel_ok]], pretest, space.thetas, settings.rotation.step_deg)
                if pretest and sel_ok.any() else None
            ),
            "width_at_range_edge": width_edge,  # shares of accepted centres at the smallest / largest width searched
            "hypotheses_per_centre": block_search.hypotheses_per_centre,
            "seeds": len(seeds),
            "chains": len(chains),
            "rows_after_growth": len(grown),
            "smoothing_vertices_removed": smoothed_vertices,
            "joins": len(grower.joins),
            "head_on_joins": grower.head_on_joins,
            "rows": len(sections),
            "statistics": block_stats,
            "validation": vsummary,
            "seconds": {"search": round(t_search, 2), "total": round(time.perf_counter() - tb, 2)},
        }
        results_out.append(BlockResult(block, sections, reviews, block_stats, polygon, report))
        log(
            f"  block {block.id}: {accepted}/{rows.size} centres accepted, {len(seeds)} seeds, {len(chains)} chains, "
            f"{len(grower.joins)} joins -> {len(sections)} rows ({block_stats['rows_needing_review']} need review"
            + (f", {len(removed)} hanging pieces removed" if settings.drop_hanging_rows else "")
            + (f", {len(estimated)} missing rows estimated" if settings.missing_row_estimate.enabled else "")
            + f") in {time.perf_counter() - tb:.1f} s"
        )
        edge = report["pretest_window_edge_share"]
        if edge is not None and edge > EDGE_WARNING_SHARE:
            log(
                f"  block {block.id}: WARNING {100 * edge:.0f}% of accepted centres chose a direction on the edge of the "
                "pre-test rotation window; the pre-test may be off."
            )
        widths = space.spacings
        for side, share, width, advice in (
            ("smallest", width_edge["min"], widths[0], "closer together than the range allows: lower its minimum"),
            ("largest", width_edge["max"], widths[-1], "further apart than the range allows: raise its maximum"),
        ):
            if share > EDGE_WARNING_SHARE:
                log(
                    f"  block {block.id}: WARNING {100 * share:.0f}% of accepted centres chose the {side} row width searched "
                    f"({width:.2f} m). The rows are probably {advice} (now {widths[0]:.2f}-{widths[-1]:.2f} m)."
                )
    t0 = lap("blocks", t0)
    progress(1.0)
    timings["total"] = round(time.perf_counter() - t_run, 3)

    sections = [s for b in results_out for s in b.sections]
    reviews = [r for b in results_out for r in b.reviews]
    report = {
        "vineliner_version": __version__,
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings_notes": list(settings.notes),
        "settings": settings.raw,
        "inputs": {
            "components": list(inputs.components),
            "ndvi": inputs.ndvi_source,
            "layers": {k: r.metadata for k, r in inputs.layers.items()},
            "grid": {"width": grid.width, "height": grid.height, "pixel_m": inputs.pixel_m, "transform": list(grid.affine),
                     "crs_wkt": grid.crs_wkt},
            "invalid_pixels": int((~valid).sum()),
        },
        "evidence": evidence_info,
        "kernels": bank.provenance() | {"cache": kernel_cache},
        "search": space.describe() | {"centre_spacing_m": settings.centre_spacing_m, "sparse_nnz": search.nnz, "window_radius_px": search.radius},
        "blocks": [b.report for b in results_out],
        "totals": {
            "rows": len(sections),
            "rows_needing_review": sum(bool(r["review"]) for r in reviews),
            "row_length_m": round(sum(s.length_m for s in sections), 1),
            "rows_estimated": sum(r["estimated"] for r in reviews),
            "bridged_m": round(sum(s.bridged_m for s in sections), 1),
            "conf": {c: sum(r["conf"] == c for r in reviews) for c in CONF_CLASSES},
            "issues": {c: sum((b.report.get("validation") or {}).get("issues", {}).get(c, 0) for b in results_out) for c in CHECKS},
        },
        "timings_s": timings,
        "caveats": [
            "Scores are overlap/matching scores, not probabilities (PROJECT.md §8).",
            "Checks are reported, never enforced; needs_review marks rows for a person (PROJECT.md §11).",
            "Passing the checks does not prove rows were read correctly; missing rows can pass (PROJECT.md §11).",
            "Block polygons are supplied by the user, not classified automatically (PROJECT.md §4).",
        ],
    }
    t = report["totals"]
    log(f"Done in {timings['total']:.0f} s: {t['rows']} rows, {t['row_length_m'] / 1000:.2f} km, {t['rows_needing_review']} need review")
    return RunResult(blocks=results_out, report=report, evidence=V_out)


def _width_edge_shares(widths: np.ndarray, chosen_idx: np.ndarray) -> dict:
    """Shares of accepted centres whose row width is the smallest / largest one searched (0 with one width)."""
    if len(widths) < 2 or chosen_idx.size == 0:
        return {"min": 0.0, "max": 0.0}
    chosen_idx = np.asarray(chosen_idx)
    return {"min": round(float(np.mean(chosen_idx == 0)), 3), "max": round(float(np.mean(chosen_idx == len(widths) - 1)), 3)}


def _band(rasters, name: str) -> np.ndarray:
    """Band by meaning. The RGB layer is searched first, then the NIR-R-G layer."""
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
