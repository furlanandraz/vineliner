"""Typed, validated view of a run configuration (PROJECT.md §12).

RunSettings.from_config() refuses to build while any OPEN key is unset and
lists every missing or invalid key at once. Nothing is filled in silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .loader import resolve_path
from .spacing import SpacingRange

_REQUIRED = object()

COVERAGE_METHODS = ("exact", "sampled", "centre", "all_touched")
EVIDENCE_METHODS = ("gray_stretch", "black_tophat")
NDVI_METHODS = ("white_tophat", "stretch")
SCALE_REGIONS = ("per_block", "all_blocks")
NODATA_RULES = ("any_band", "all_bands")
NORMALISATIONS = ("renormalise_valid", "full_kernel")
UNAVAILABLE_BAND_POLICIES = ("renormalise", "zero")
TRANSLATION_BOXES = ("tested_d", "nominal_d0")
SURFACE_STORAGE = ("all", "winning_spacing", "none")
SELECTION_METHODS = ("best", "orientation_consensus")


class ConfigError(ValueError):
    """The configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class InputSettings:
    rgb: Path
    rgb_bands: tuple[str, ...]
    nir: Path | None
    nir_bands: tuple[str, ...] | None
    nodata_value: float | None  # None: use the value declared in the file
    nodata_rule: str
    ndvi: Path | None = None  # single-band NDVI raster on the same grid; its own NoData applies


@dataclass(frozen=True)
class VineyardSettings:
    source: Path
    layer: str | None
    id_field: str | None
    blocks: tuple[int, ...] | None  # None: every block


@dataclass(frozen=True)
class EvidenceSettings:
    method: str
    bands: tuple[str, ...]
    tophat_size_m: float | None
    scale_percentile: float
    stretch_percentiles: tuple[float, float]
    scale_region: str  # "per_block" or "all_blocks"
    ndvi_weight: float  # 0 = RGB-only evidence
    ndvi_method: str | None
    ndvi_tophat_size_m: float | None


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
    cache_dir: Path | None = None  # None: build kernels every run


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
    unavailable_bands: str


@dataclass(frozen=True)
class CentroidPretestSettings:
    """Centroid pre-test (tools/search/pretest.py)."""

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
class ExportSettings:
    shapefile: bool
    geopackage: bool
    overlay: bool
    drop_hanging_rows: bool = False  # keep only rows meeting their block boundary at exactly their two ends


@dataclass(frozen=True)
class RunSettings:
    input: InputSettings
    vineyards: VineyardSettings
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
    store_surfaces: str
    export: ExportSettings
    raw: dict[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RunSettings:
        r = _Reader(config)

        # [input]
        rgb = r.get("input", "rgb", kind=str)
        rgb_bands = r.get("input", "rgb_bands", kind=list)
        nir = r.get("input", "nir", None, kind=str)
        nir_bands = r.get("input", "nir_bands", None if nir is None else _REQUIRED, kind=list)
        nodata_value = r.get("input", "nodata_value", None, kind=float)
        nodata_rule = r.get("input", "nodata_rule", "any_band", choices=NODATA_RULES)
        ndvi = r.get("input", "ndvi", None, kind=str)

        # [vineyards]
        source = r.get("vineyards", "source", kind=str)
        layer = r.get("vineyards", "layer", None, kind=str)
        id_field = r.get("vineyards", "id_field", None, kind=str)
        blocks = r.get("vineyards", "blocks", None, kind=list)

        # [evidence]
        method = r.get("evidence", "method", choices=EVIDENCE_METHODS)
        ev_bands = r.get("evidence", "bands", ["red", "green", "blue"], kind=list)
        tophat = r.get("evidence", "tophat_size_m", _REQUIRED if method == "black_tophat" else None, kind=float)
        scale_pct = r.get("evidence", "scale_percentile", 99.0, kind=float)
        stretch = r.get("evidence", "stretch_percentiles", [2.0, 98.0], kind=list)
        scale_region = r.get("evidence", "scale_region", choices=SCALE_REGIONS)
        ndvi_weight = r.get("evidence", "ndvi_weight", 0.0, kind=float)
        use_ndvi = (ndvi_weight or 0.0) > 0
        ndvi_method = r.get("evidence", "ndvi_method", _REQUIRED if use_ndvi else None, choices=NDVI_METHODS)
        ndvi_tophat = r.get("evidence", "ndvi_tophat_size_m", _REQUIRED if ndvi_method == "white_tophat" else None, kind=float)

        # [template]
        indices = r.get("template", "indices", kind=list)
        weights = r.get("template", "weights", kind=list)
        line_length = r.get("template", "line_length_m", kind=float)
        band_width = r.get("template", "band_width_px", kind=float)

        # [kernels]
        coverage = r.get("kernels", "coverage", choices=COVERAGE_METHODS)
        samples = r.get("kernels", "subpixel_samples", _REQUIRED if coverage == "sampled" else None, kind=int)
        use_cache = r.get("kernels", "cache", True, kind=bool)
        cache_dir = r.get("kernels", "cache_dir", ".cache/kernels", kind=str)

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
        unavailable = r.get("scoring", "unavailable_bands", choices=UNAVAILABLE_BAND_POLICIES)
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

        # [validation], [storage], [export]
        val = [r.get("validation", k, kind=float) for k in ("spacing_min_m", "spacing_max_m", "tolerance_m")]
        sample_step = r.get("validation", "sample_step_m", 1.0, kind=float)
        review_keys = ("max_bridged_m", "max_bridged_frac", "max_spacing_out_frac", "min_support_frac", "min_length_m")
        review = {k: r.get("review", k, kind=float) for k in review_keys}
        surfaces = r.get("storage", "surfaces", choices=SURFACE_STORAGE)
        export = [r.get("export", k, True, kind=bool) for k in ("shapefile", "geopackage", "overlay")]
        export.append(r.get("export", "drop_hanging_rows", False, kind=bool))

        r.raise_if_failed()

        # Cross-key checks, once every value is present.
        problems = []
        if len(indices) != len(weights):
            problems.append("[template] indices and weights differ in length")
        elif not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            problems.append(f"[template] weights sum to {sum(weights)}, not 1.00 (PROJECT.md §6)")
        if 0 not in indices:
            problems.append("[template] indices must contain the centre line 0")
        if band_width <= 0:
            problems.append("[template] band_width_px must be > 0 for rasterisation")
        if line_length <= 0:
            problems.append("[template] line_length_m must be > 0")
        if not set(seed_lines) <= set(indices):
            problems.append(f"[seeds] lines {seed_lines} must be template indices {indices}")
        if not 0 < min_cov <= 1:
            problems.append("[scoring] min_valid_coverage must be in (0, 1]")
        if rot[2] <= 0 or rot[1] <= rot[0]:
            problems.append("[rotation] needs start_deg < stop_deg and step_deg > 0")
        if centre_spacing <= 0:
            problems.append("[centres] spacing_m must be > 0")
        if pt_theta < 0 or pt_spacing < 0 or pt_rings < 0:
            problems.append("[centroid_pretest] windows and neighbour_rings must be >= 0")
        if val[0] > val[1]:
            problems.append("[validation] spacing_min_m is above spacing_max_m")
        if nir is not None and nir_bands is not None and len(set(nir_bands)) != len(nir_bands):
            problems.append("[input] nir_bands has duplicate names")
        if not 0 <= ndvi_weight <= 1:
            problems.append("[evidence] ndvi_weight must be in [0, 1]")
        if use_ndvi and ndvi is None:
            problems.append("[evidence] ndvi_weight > 0 needs [input] ndvi")
        try:
            spacing = SpacingRange(s_min, s_max, s_step)
        except ValueError as exc:
            problems.append(f"[spacing] {exc}")
        if problems:
            raise ConfigError("Invalid configuration:\n  " + "\n  ".join(problems))

        return cls(
            input=InputSettings(
                rgb=resolve_path(rgb),
                rgb_bands=tuple(rgb_bands),
                nir=resolve_path(nir) if nir else None,
                nir_bands=tuple(nir_bands) if nir_bands else None,
                nodata_value=nodata_value,
                nodata_rule=nodata_rule,
                ndvi=resolve_path(ndvi) if ndvi else None,
            ),
            vineyards=VineyardSettings(
                source=resolve_path(source),
                layer=layer,
                id_field=id_field,
                blocks=tuple(int(b) for b in blocks) if blocks is not None else None,
            ),
            evidence=EvidenceSettings(
                method=method,
                bands=tuple(ev_bands),
                tophat_size_m=tophat,
                scale_percentile=scale_pct,
                stretch_percentiles=(float(stretch[0]), float(stretch[1])),
                scale_region=scale_region,
                ndvi_weight=ndvi_weight,
                ndvi_method=ndvi_method,
                ndvi_tophat_size_m=ndvi_tophat,
            ),
            template=TemplateSettings(
                indices=tuple(int(i) for i in indices),
                weights=tuple(float(w) for w in weights),
                line_length_m=line_length,
                band_width_px=band_width,
            ),
            kernels=KernelSettings(coverage=coverage, subpixel_samples=samples, cache_dir=resolve_path(cache_dir) if use_cache else None),
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
                unavailable_bands=unavailable,
            ),
            selection=SelectionSettings(method=sel_method, consensus_radius_m=sel_radius, consensus_window_deg=sel_window),
            seed_lines=tuple(int(k) for k in seed_lines),
            seed_min_band_mean=seed_min_mean,
            reconstruction=ReconstructionSettings(**rec),
            validation=ValidationSettings(*val, sample_step_m=sample_step),
            review=ReviewSettings(**review),
            store_surfaces=surfaces,
            export=ExportSettings(*export),
            raw=config,
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
            lines.append("Missing keys (OPEN choices must be set explicitly, PROJECT.md §15):")
            lines += [f"  {m}" for m in self.missing]
        if self.invalid:
            lines.append("Invalid values:")
            lines += [f"  {i}" for i in self.invalid]
        if lines:
            raise ConfigError("\n".join(lines))
