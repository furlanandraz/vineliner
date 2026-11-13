"""Pipeline inputs: the raster layers on one analysis grid, and the block polygons.

The plugin reads the layers the user picked and puts them on one projected,
north-up analysis grid (vineliner/qgis_io). This module checks them and turns
them into the components the row evidence is built from:

  rgb   the RGB layer        bands red, green, blue
  nir   the NIR-R-G layer    bands nir, red, green
  ndvi  the NDVI layer       one band; when it is missing and the NIR-R-G layer
                             is there, NDVI is computed from its NIR and red bands

So RGB alone gives the RGB component; RGB with NIR-R-G gives RGB, NIR and a
computed NDVI; NDVI alone gives NDVI. A supplied NDVI layer is used rather than
one computed from NIR-R-G.

Rasters hold float32 arrays with NaN where there is no data. A pixel is valid
where every band of every layer is finite, and an NDVI value lies in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geo import Grid
from ..vineyards import Block, block_mask

LAYER_BANDS = {"rgb": ("red", "green", "blue"), "nir": ("nir", "red", "green"), "ndvi": ("ndvi",)}
LAYER_NAMES = {"rgb": "RGB", "nir": "NIR-R-G", "ndvi": "NDVI"}


@dataclass
class Raster:
    data: np.ndarray  # (bands, rows, cols) float32, NaN where there is no data
    grid: Grid
    band_names: tuple[str, ...]  # meaning of each band
    metadata: dict = field(default_factory=dict)  # source, CRS, pixel size, resampling, ...

    def band(self, name: str) -> np.ndarray:
        return self.data[self.band_names.index(name)]


@dataclass
class Inputs:
    layers: dict[str, Raster]  # evidence components: "rgb", "nir", "ndvi" (supplied or computed)
    grid: Grid
    pixel_m: float
    valid: np.ndarray
    blocks: list[Block]
    masks: dict[int, np.ndarray]
    ndvi_source: str | None  # "supplied", "computed from NIR-R-G" or None

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(name for name in LAYER_BANDS if name in self.layers)

    @property
    def band_rasters(self) -> list[Raster]:
        """The rasters the RGB component's gray bands are looked up in: RGB, then NIR-R-G."""
        return [self.layers[k] for k in ("rgb", "nir") if k in self.layers]


class InputError(ValueError):
    """The input layers cannot be used together."""


def ndvi_from(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """(NIR - red) / (NIR + red), NaN where the sum is not positive or a band has no data."""
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    total = nir + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / total
    ndvi[~(np.isfinite(total) & (total > 0))] = np.nan
    return ndvi.astype(np.float32)


def assemble_inputs(layers: dict[str, Raster], blocks: list[Block]) -> Inputs:
    """Check the layers (at least one, one north-up metric grid) and derive the evidence components."""
    given = {k: v for k, v in layers.items() if v is not None}
    unknown = set(given) - set(LAYER_BANDS)
    if unknown:
        raise InputError(f"unknown input layers {sorted(unknown)}; expected {', '.join(LAYER_BANDS)}")
    if not given:
        raise InputError("select at least one input layer: RGB, NIR-R-G or NDVI")
    if not blocks:
        raise InputError("the vineyard polygon layer holds no polygons")
    first = next(k for k in LAYER_BANDS if k in given)
    grid = given[first].grid
    for name, raster in given.items():
        if raster.data.ndim != 3 or raster.data.shape[0] != len(LAYER_BANDS[name]):
            raise InputError(f"the {LAYER_NAMES[name]} layer needs {len(LAYER_BANDS[name])} band(s) {LAYER_BANDS[name]}, got {raster.data.shape}")
        if tuple(raster.band_names) != LAYER_BANDS[name]:
            raise InputError(f"the {LAYER_NAMES[name]} bands must be {LAYER_BANDS[name]}, got {raster.band_names}")
        if not raster.grid.same_grid(grid):
            raise InputError(f"the {LAYER_NAMES[name]} layer is not on the {LAYER_NAMES[first]} layer's grid")
    grid.require_north_up()
    pixel_m = grid.require_square_pixels()
    if "ndvi" in given:
        band = given["ndvi"].data[0]
        finite = band[np.isfinite(band)]
        if finite.size and np.mean((finite >= -1.0) & (finite <= 1.0)) < 0.5:
            raise InputError(
                f"most NDVI values lie outside -1..1 (range {finite.min():.3g} to {finite.max():.3g}); "
                "the NDVI layer must hold NDVI itself, not a scaled or colour-rendered version"
            )

    layers_out = dict(given)
    ndvi_source = "supplied" if "ndvi" in given else None
    if "ndvi" not in given and "nir" in given:
        nir = given["nir"]
        layers_out["ndvi"] = Raster(
            ndvi_from(nir.band("nir"), nir.band("red"))[np.newaxis], grid, ("ndvi",),
            {"computed": "(NIR - red) / (NIR + red) from the NIR-R-G layer"},
        )
        ndvi_source = "computed from NIR-R-G"
    layers_out = {k: layers_out[k] for k in LAYER_BANDS if k in layers_out}

    valid = np.ones(grid.shape, dtype=bool)
    for raster in layers_out.values():
        valid &= np.isfinite(raster.data).all(axis=0)
    if "ndvi" in layers_out:
        band = layers_out["ndvi"].data[0]
        with np.errstate(invalid="ignore"):
            valid &= (band >= -1.0) & (band <= 1.0)
    masks = {b.id: block_mask(b, grid) for b in blocks}
    if not any((mask & valid).any() for mask in masks.values()):
        inside = sum(int(mask.sum()) for mask in masks.values())
        raise InputError(
            "no input layer has data inside the vineyard polygons"
            + (f" (they cover {inside} pixels, all without data)" if inside else " (no pixel centre lies inside them)")
            + "; check that the layers and the polygons cover the same place, and that each layer's CRS matches its coordinates"
        )
    return Inputs(layers_out, grid, pixel_m, valid, blocks, masks, ndvi_source)
