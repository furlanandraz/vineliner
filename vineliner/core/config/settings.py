"""Typed, validated view of a run configuration (PROJECT.md §12).

The configuration is core/defaults.toml with the plugin dialog's values merged
on top (config.loader.plugin_config). RunSettings.from_config() lists every
missing or invalid key at once. Nothing is filled in silently.

Input layers are not part of the configuration: the plugin passes them to the
pipeline, and EvidenceSettings.weights_for() rescales the evidence weights over
the components those layers provide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .spacing import SpacingRange

_REQUIRED = object()

INPUT_LAYERS = ("rgb", "nir", "ndvi")  # RGB, NIR-R-G, NDVI; also the evidence components, in this order
COVERAGE_METHODS = ("exact", "sampled", "centre", "all_touched")
EVIDENCE_METHODS = ("gray_stretch", "black_tophat")  # RGB component: rows darker than the ground
BRIGHT_METHODS = ("white_tophat", "stretch")  # NIR and NDVI components: rows brighter than the ground
NDVI_METHODS = BRIGHT_METHODS
SCALE_REGIONS = ("per_block", "all_blocks")
COMBINE_METHODS = ("best_per_block", "weighted")  # how several evidence components make V
NORMALISATIONS = ("renormalise_valid", "full_kernel")
TRANSLATION_BOXES = ("tested_d", "nominal_d0")
SELECTION_METHODS = ("best", "orientation_consensus")


class ConfigError(ValueError):
    """The configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class EvidenceSettings:
    """V is the weighted mean of one component per input layer in use (imagery/evidence.py)."""

    method: str  # RGB component
    bands: tuple[str, ...]
    tophat_size_m: float | None
    scale_percentile: float
    stretch_percentiles: tuple[float, float]
    scale_region: str  # "per_block" or "all_blocks"
    ndvi_weight: float  # relative weights, rescaled over the components in use
    ndvi_method: str
    ndvi_tophat_size_m: float | None
    rgb_weight: float = 1.0
    nir_weight: float = 0.0
    nir_method: str = "white_tophat"
    nir_tophat_size_m: float | None = None
    combine: str = "weighted"  # "weighted": weighted mean; "best_per_block": the component showing rows best, per block

    def weights_for(self, components) -> dict[str, float]:
        """Shares of V for the components in use (those with weight > 0), summing to 1, in INPUT_LAYERS order."""
        raw = {"rgb": self.rgb_weight, "nir": self.nir_weight, "ndvi": self.ndvi_weight}
        used = {name: raw[name] for name in INPUT_LAYERS if name in components and raw[name] > 0}
        if not used:
            raise ConfigError(f"the evidence components in use ({', '.join(components)}) all have weight 0")
        total = sum(used.values())
        return {name: w / total for name, w in used.items()}


@dataclass(frozen=True)
class TemplateSettings:
    indices: tuple[int, ...]
    weights: tuple[float, ...]
    line_length_m: float
    band_width_px: float


@dataclass(frozen=True)
class KernelSettings:
    coverage: str
    subpixel_samples: int | None
    cache: bool = False  # save kernels in the cache folder the caller gives and reuse them


@dataclass(frozen=True)
class RotationSettings:
    start_deg: float
    stop_deg: float  # exclusive
    step_deg: float

    def angles(self) -> tuple[float, ...]:
        n = math.ceil((self.stop_deg - self.start_deg) / self.step_deg - 1e-9)
        return tuple(round(self.start_deg + i * self.step_deg, 9) for i in range(n))


@dataclass(frozen=True)
class TranslationSettings:
    box: str
    order: str


@dataclass(frozen=True)
class ScoringSettings:
    accept_min: float
    normalisation: str
    min_valid_coverage: float


@dataclass(frozen=True)
class CentroidPretestSettings:
    """Centroid pre-test (search/pretest.py)."""

    enabled: bool
    theta_window_deg: float
    spacing_window_frac: float
    neighbour_rings: int


@dataclass(frozen=True)
class SelectionSettings:
    method: str
    consensus_radius_m: float | None
    consensus_window_deg: float | None


@dataclass(frozen=True)
class ReconstructionSettings:
    growth_step_m: float
    min_curvature_radius_m: float
    link_angle_deg: float
    link_lateral_frac: float
    link_along_m: float
    chain_smoothing_m: float
    growth_min_score: float
    max_gap_m: float
    bridge_tangent_m: float
    join_distance_frac: float
    join_angle_deg: float
    max_row_length_m: float

    @property
    def max_turn_deg(self) -> float:
        """Largest direction change per growth step allowed by the minimum curvature radius."""
        return math.degrees(self.growth_step_m / self.min_curvature_radius_m)


@dataclass(frozen=True)
class ValidationSettings:
    spacing_min_m: float
    spacing_max_m: float
    tolerance_m: float
    sample_step_m: float


@dataclass(frozen=True)
class ReviewSettings:
    """Thresholds that set a row's review flag. They only label rows; nothing is removed."""

    max_bridged_m: float
    max_bridged_frac: float
    max_spacing_out_frac: float
    min_support_frac: float
    min_length_m: float


@dataclass(frozen=True)
class SmoothingSettings:
    """Maximum direction change between row vertices (reconstruction/smoothing.py)."""

    enabled: bool
    max_turn_deg: float


@dataclass(frozen=True)
class NumberingSettings:
    """Row numbering per block (reconstruction/numbering.py)."""

    enabled: bool
    transect_interval_m: float


@dataclass(frozen=True)
class MissingRowSettings:
    """Missing-row estimate (reconstruction/missing.py)."""

    enabled: bool
    gap_tolerance_frac: float  # neighbours within width * (1 +/- this) are correctly spaced
    whole_row_tolerance_frac: float  # a filled gap's new spacing must lie this close to the width
    station_factor: float
    max_missing_rows: int
    merge_pieces: bool = True  # squash detected hanging pieces into estimated rows (squash.py)
    merge_buffer_frac: float = 0.5  # buffer half-width around an estimated row, as a fraction of the row width
    merge_min_inside_frac: float = 0.8  # share of a piece that must lie inside the buffer
    merge_max_angle_deg: float = 10.0  # largest direction difference between piece and estimated row


@dataclass(frozen=True)
class RunSettings:
    evidence: EvidenceSettings
    template: TemplateSettings
    kernels: KernelSettings
    spacing: SpacingRange
    spacing_nominal_m: float
    rotation: RotationSettings
    translation: TranslationSettings
    centre_spacing_m: float
    centroid_pretest: CentroidPretestSettings
    scoring: ScoringSettings
    selection: SelectionSettings
    seed_lines: tuple[int, ...]
    seed_min_band_mean: float
    reconstruction: ReconstructionSettings
    validation: ValidationSettings
    review: ReviewSettings
    smoothing: SmoothingSettings
    missing_row_estimate: MissingRowSettings
    row_numbering: NumberingSettings
    drop_hanging_rows: bool  # keep only rows meeting their block boundary at exactly their two ends
    raw: dict[str, Any]
    notes: tuple[str, ...] = ()  # settings the code adjusted, reported at the start of a run

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RunSettings:
        r = _Reader(config)

        # [evidence]: one component per input layer; the pipeline uses those the inputs provide.
        weights_in = {name: r.get("evidence", f"{name}_weight", 1.0 if name == "rgb" else 0.0, kind=float) for name in INPUT_LAYERS}
        method = r.get("evidence", "method", choices=EVIDENCE_METHODS)
        ev_bands = r.get("evidence", "bands", ["red", "green", "blue"], kind=list)
        tophat = r.get("evidence", "tophat_size_m", _REQUIRED if method == "black_tophat" else None, kind=float)
        scale_pct = r.get("evidence", "scale_percentile", 99.0, kind=float)
        stretch = r.get("evidence", "stretch_percentiles", [2.0, 98.0], kind=list)
        scale_region = r.get("evidence", "scale_region", choices=SCALE_REGIONS)
        combine = r.get("evidence", "combine", "weighted", choices=COMBINE_METHODS)
        bright = {}
        for name in ("nir", "ndvi"):
            m = r.get("evidence", f"{name}_method", choices=BRIGHT_METHODS)
            bright[name] = (m, r.get("evidence", f"{name}_tophat_size_m", _REQUIRED if m == "white_tophat" else None, kind=float))

        # [template]
        indices = r.get("template", "indices", kind=list)
        weights = r.get("template", "weights", kind=list)
        line_length = r.get("template", "line_length_m", kind=float)
        band_width = r.get("template", "band_width_px", kind=float)

        # [kernels]
        coverage = r.get("kernels", "coverage", choices=COVERAGE_METHODS)
        samples = r.get("kernels", "subpixel_samples", _REQUIRED if coverage == "sampled" else None, kind=int)
        use_cache = r.get("kernels", "cache", False, kind=bool)

        # [spacing], [rotation], [translation], [centres]
        nominal = r.get("spacing", "nominal_m", kind=float)
        s_min = r.get("spacing", "min_m", kind=float)
        s_max = r.get("spacing", "max_m", kind=float)
        s_step = r.get("spacing", "step_m", kind=float)
        rot = [r.get("rotation", k, kind=float) for k in ("start_deg", "stop_deg", "step_deg")]
        box = r.get("translation", "box", choices=TRANSLATION_BOXES)
        order = r.get("translation", "order", "centre_outward", choices=("centre_outward",))
        centre_spacing = r.get("centres", "spacing_m", kind=float)
        pt_enabled = r.get("centroid_pretest", "enabled", False, kind=bool)
        pt_theta = r.get("centroid_pretest", "theta_window_deg", 10.0, kind=float)
        pt_spacing = r.get("centroid_pretest", "spacing_window_frac", 0.10, kind=float)
        pt_rings = r.get("centroid_pretest", "neighbour_rings", 3, kind=int)

        # [scoring], [seeds]
        accept_min = r.get("scoring", "accept_min", kind=float)
        normalisation = r.get("scoring", "normalisation", choices=NORMALISATIONS)
        min_cov = r.get("scoring", "min_valid_coverage", kind=float)
        seed_lines = r.get("seeds", "lines", kind=list)
        seed_min_mean = r.get("seeds", "min_band_mean", kind=float)
        sel_method = r.get("selection", "method", choices=SELECTION_METHODS)
        need = _REQUIRED if sel_method == "orientation_consensus" else None
        sel_radius = r.get("selection", "consensus_radius_m", need, kind=float)
        sel_window = r.get("selection", "consensus_window_deg", need, kind=float)

        # [reconstruction]
        rec_keys = (
            "growth_step_m", "min_curvature_radius_m", "link_angle_deg", "link_lateral_frac",
            "link_along_m", "chain_smoothing_m", "growth_min_score", "max_gap_m", "bridge_tangent_m", "join_distance_frac",
            "join_angle_deg", "max_row_length_m",
        )
        rec = {k: r.get("reconstruction", k, kind=float) for k in rec_keys}
        drop_hanging = r.get("reconstruction", "drop_hanging_rows", False, kind=bool)

        # [validation], [review], [smoothing], [missing_row_estimate], [row_numbering]
        val = [r.get("validation", k, kind=float) for k in ("spacing_min_m", "spacing_max_m", "tolerance_m")]
        sample_step = r.get("validation", "sample_step_m", 1.0, kind=float)
        review_keys = ("max_bridged_m", "max_bridged_frac", "max_spacing_out_frac", "min_support_frac", "min_length_m")
        review = {k: r.get("review", k, kind=float) for k in review_keys}
        num_enabled = r.get("row_numbering", "enabled", False, kind=bool)
        num_interval = r.get("row_numbering", "transect_interval_m", 10.0, kind=float)
        smooth_enabled = r.get("smoothing", "enabled", True, kind=bool)
        smooth_turn = r.get("smoothing", "max_turn_deg", 20.0, kind=float)
        mre_enabled = r.get("missing_row_estimate", "enabled", False, kind=bool)
        mre_tol = r.get("missing_row_estimate", "gap_tolerance_frac", 0.50, kind=float)
        mre_whole = r.get("missing_row_estimate", "whole_row_tolerance_frac", 0.20, kind=float)
        mre_station = r.get("missing_row_estimate", "station_factor", 5.0, kind=float)
        mre_max = r.get("missing_row_estimate", "max_missing_rows", 3, kind=int)
        mre_merge = r.get("missing_row_estimate", "merge_pieces", True, kind=bool)
        mre_buf = r.get("missing_row_estimate", "merge_buffer_frac", 0.5, kind=float)
        mre_inside = r.get("missing_row_estimate", "merge_min_inside_frac", 0.8, kind=float)
        mre_angle = r.get("missing_row_estimate", "merge_max_angle_deg", 10.0, kind=float)

        r.raise_if_failed()

        # Cross-key checks, once every value is present.
        problems = []
        if len(indices) != len(weights):
            problems.append("[template] indices and weights differ in length")
        elif not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            problems.append(f"[template] weights sum to {sum(weights)}, not 1.00 (PROJECT.md §6)")
        if 0 not in indices:
            problems.append("[template] indices must contain the centre line 0")
        elif list(indices) != [-k for k in reversed(indices)]:
            problems.append("[template] indices must be symmetric about 0 (weights of lines outside a block are mirrored)")
        if band_width <= 0:
            problems.append("[template] band_width_px must be > 0 for rasterisation")
        if line_length <= 0:
            problems.append("[template] line_length_m must be > 0")
        if not set(seed_lines) <= set(indices):
            problems.append(f"[seeds] lines {seed_lines} must be template indices {indices}")
        if not 0 < min_cov <= 1:
            problems.append("[scoring] min_valid_coverage must be in (0, 1]")
        if not 0 < mre_buf <= 1 or not 0 < mre_inside <= 1 or not 0 < mre_angle < 90:
            problems.append(
                "[missing_row_estimate] needs 0 < merge_buffer_frac <= 1, 0 < merge_min_inside_frac <= 1 "
                "and 0 < merge_max_angle_deg < 90"
            )
        if num_interval <= 0:
            problems.append("[row_numbering] transect_interval_m must be > 0")
        if not 0 < smooth_turn < 180:
            problems.append("[smoothing] max_turn_deg must be between 0 and 180")
        if not 0 < mre_tol <= 1 or not 0 < mre_whole <= 0.5 or mre_station <= 0 or mre_max < 1:
            problems.append(
                "[missing_row_estimate] needs 0 < gap_tolerance_frac <= 1, 0 < whole_row_tolerance_frac <= 0.5, "
                "station_factor > 0 and max_missing_rows >= 1"
            )
        if rot[2] <= 0 or rot[1] <= rot[0]:
            problems.append("[rotation] needs start_deg < stop_deg and step_deg > 0")
        if centre_spacing <= 0:
            problems.append("[centres] spacing_m must be > 0")
        if pt_theta < 0 or pt_spacing < 0 or pt_rings < 0:
            problems.append("[centroid_pretest] windows and neighbour_rings must be >= 0")
        if val[0] > val[1]:
            problems.append("[validation] spacing_min_m is above spacing_max_m")
        if any(w < 0 for w in weights_in.values()) or not any(w > 0 for w in weights_in.values()):
            problems.append("[evidence] rgb_weight, nir_weight and ndvi_weight must be >= 0, and one of them > 0")
        try:
            spacing = SpacingRange(s_min, s_max, s_step)
        except ValueError as exc:
            problems.append(f"[spacing] {exc}")
        if problems:
            raise ConfigError("Invalid configuration:\n  " + "\n  ".join(problems))

        notes = []
        # The estimate needs complete rows, so it turns hanging-row deletion on (TOML cannot tie settings together).
        if mre_enabled and not drop_hanging:
            drop_hanging = True
            notes.append(
                "[missing_row_estimate] is on, so [reconstruction] drop_hanging_rows is turned on too: "
                "rows are only estimated between complete neighbouring rows."
            )

        return cls(
            evidence=EvidenceSettings(
                method=method,
                bands=tuple(ev_bands),
                tophat_size_m=tophat,
                scale_percentile=scale_pct,
                stretch_percentiles=(float(stretch[0]), float(stretch[1])),
                scale_region=scale_region,
                combine=combine,
                ndvi_weight=weights_in["ndvi"],
                ndvi_method=bright["ndvi"][0],
                ndvi_tophat_size_m=bright["ndvi"][1],
                rgb_weight=weights_in["rgb"],
                nir_weight=weights_in["nir"],
                nir_method=bright["nir"][0],
                nir_tophat_size_m=bright["nir"][1],
            ),
            template=TemplateSettings(
                indices=tuple(int(i) for i in indices),
                weights=tuple(float(w) for w in weights),
                line_length_m=line_length,
                band_width_px=band_width,
            ),
            kernels=KernelSettings(coverage=coverage, subpixel_samples=samples, cache=use_cache),
            spacing=spacing,
            spacing_nominal_m=nominal,
            rotation=RotationSettings(*rot),
            translation=TranslationSettings(box=box, order=order),
            centre_spacing_m=centre_spacing,
            centroid_pretest=CentroidPretestSettings(
                enabled=pt_enabled, theta_window_deg=pt_theta, spacing_window_frac=pt_spacing, neighbour_rings=pt_rings
            ),
            scoring=ScoringSettings(
                accept_min=accept_min,
                normalisation=normalisation,
                min_valid_coverage=min_cov,
            ),
            selection=SelectionSettings(method=sel_method, consensus_radius_m=sel_radius, consensus_window_deg=sel_window),
            seed_lines=tuple(int(k) for k in seed_lines),
            seed_min_band_mean=seed_min_mean,
            reconstruction=ReconstructionSettings(**rec),
            validation=ValidationSettings(*val, sample_step_m=sample_step),
            review=ReviewSettings(**review),
            drop_hanging_rows=drop_hanging,
            smoothing=SmoothingSettings(enabled=smooth_enabled, max_turn_deg=smooth_turn),
            row_numbering=NumberingSettings(enabled=num_enabled, transect_interval_m=num_interval),
            missing_row_estimate=MissingRowSettings(
                mre_enabled, mre_tol, mre_whole, mre_station, mre_max, mre_merge, mre_buf, mre_inside, mre_angle
            ),
            raw=config,
            notes=tuple(notes),
        )


class _Reader:
    """Reads config values and collects every problem before reporting."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.missing: list[str] = []
        self.invalid: list[str] = []

    def get(self, section: str, key: str, default: Any = _REQUIRED, kind: type | None = None, choices=None):
        table = self.config.get(section, {})
        if key not in table:
            if default is _REQUIRED:
                self.missing.append(f"[{section}] {key}")
            return None if default is _REQUIRED else default
        value = table[key]
        name = f"[{section}] {key}"
        if choices is not None and value not in choices:
            self.invalid.append(f"{name} = {value!r}; choose one of {', '.join(map(repr, choices))}")
            return None
        if kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                self.invalid.append(f"{name} must be a number, got {value!r}")
                return None
            return float(value)
        if kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                self.invalid.append(f"{name} must be a whole number, got {value!r}")
                return None
            return value
        if kind is not None and not isinstance(value, kind):
            self.invalid.append(f"{name} must be a {kind.__name__}, got {value!r}")
            return None
        return value

    def raise_if_failed(self) -> None:
        lines = []
        if self.missing:
            lines.append("Missing keys:")
            lines += [f"  {m}" for m in self.missing]
        if self.invalid:
            lines.append("Invalid values:")
            lines += [f"  {i}" for i in self.invalid]
        if lines:
            raise ConfigError("\n".join(lines))
