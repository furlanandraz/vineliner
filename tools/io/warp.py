"""Reprojection and resampling onto one analysis grid (PROJECT.md §3).

The pipeline needs every input layer on one north-up grid in a projected CRS
with metre units. Drone mosaics often come in geographic WGS 84 (EPSG:4326),
at a pixel size far finer than the search needs, and with each sensor on its
own grid. This adapter makes the alignment an explicit step:

  - target_grid() chooses one north-up grid in the target CRS that covers
    every source, with its corners on whole multiples of the pixel size;
  - warp_to_grid() reprojects one source onto it with a stated resampling
    method and writes the result with its provenance in the GeoTIFF tags,
    so every run's report.json repeats where its inputs came from.

The datum transformation is whatever PROJ picks for the CRS pair; its
description and accuracy are recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from ..geo import Grid

RESAMPLING = {
    "average": Resampling.average,  # area mean: the right choice when making pixels coarser
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
}


@dataclass(frozen=True)
class SourceLayer:
    name: str  # output layer, e.g. "rgb"; written to <out_dir>/<name>.tif
    path: Path
    bands: tuple[int, ...] | None = None  # 1-based band numbers to keep; None = every band


def target_grid(paths, crs: str, pixel_m: float) -> Grid:
    """North-up grid in crs covering the footprint of every source, its corners on multiples of pixel_m."""
    target = CRS.from_user_input(crs)
    if not target.is_projected or target.axis_info[0].unit_name not in ("metre", "meter"):
        raise ValueError(f"{crs} is not a projected CRS in metres (PROJECT.md §3)")
    if pixel_m <= 0:
        raise ValueError("pixel size must be > 0")
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{path} has no CRS; it cannot be reprojected (PROJECT.md §3)")
            b = transform_bounds(src.crs, target, *src.bounds, densify_pts=21)
        xmin, ymin, xmax, ymax = min(xmin, b[0]), min(ymin, b[1]), max(xmax, b[2]), max(ymax, b[3])
    # Rounded to micrometres, so a corner reads 5092044.6 rather than 5092044.600000001.
    x0 = round(math.floor(xmin / pixel_m) * pixel_m, 6)
    y1 = round(math.ceil(ymax / pixel_m) * pixel_m, 6)
    width = int(math.ceil((xmax - x0) / pixel_m - 1e-9))
    height = int(math.ceil((y1 - ymin) / pixel_m - 1e-9))
    return Grid(pixel_m, 0.0, x0, 0.0, -pixel_m, y1, width, height, target.to_wkt())


def warp_to_grid(layer: SourceLayer, dst_path: str | Path, grid: Grid, resampling: str = "average") -> dict:
    """Reproject layer onto grid and write it. Returns the provenance written to the file's tags.

    NoData: the source's declared NoData is excluded from the resampling. Float
    output marks NoData as NaN; integer output keeps the source's NoData value.
    """
    if resampling not in RESAMPLING:
        raise ValueError(f"unknown resampling {resampling!r}; choose one of {', '.join(RESAMPLING)}")
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_crs = rasterio.crs.CRS.from_wkt(grid.crs_wkt)
    with rasterio.open(layer.path) as src:
        if src.crs is None:
            raise ValueError(f"{layer.path} has no CRS; it cannot be reprojected (PROJECT.md §3)")
        bands = layer.bands or tuple(range(1, src.count + 1))
        if not set(bands) <= set(range(1, src.count + 1)):
            raise ValueError(f"{layer.path} has {src.count} bands; cannot keep bands {bands}")
        dtype = np.dtype(src.dtypes[bands[0] - 1])
        is_float = np.issubdtype(dtype, np.floating)
        src_nodata = src.nodata
        dst_nodata = np.nan if is_float else src_nodata
        out = np.full((len(bands), grid.height, grid.width), np.nan if is_float else (src_nodata or 0), dtype=dtype)
        reproject(
            source=rasterio.band(src, list(bands)),
            destination=out,
            src_nodata=src_nodata,
            dst_transform=rasterio.Affine(*grid.affine),
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=RESAMPLING[resampling],
        )
        src_px = (abs(src.transform.a), abs(src.transform.e))
        provenance = {
            "VINELINER_SOURCE": str(layer.path),
            "VINELINER_SOURCE_BANDS": ",".join(map(str, bands)),
            "VINELINER_SOURCE_CRS": src.crs.to_string(),
            "VINELINER_SOURCE_PIXEL": f"{src_px[0]:.6g} x {src_px[1]:.6g} {src.crs.linear_units or 'degree'}",
            "VINELINER_SOURCE_NODATA": str(src_nodata),
            "VINELINER_RESAMPLING": resampling,
            "VINELINER_TRANSFORMATION": _transformation(src.crs, dst_crs),
            "VINELINER_PREPARED": date.today().isoformat(),
        }
    profile = {
        "driver": "GTiff", "width": grid.width, "height": grid.height, "count": len(bands), "dtype": dtype.name,
        "transform": rasterio.Affine(*grid.affine), "crs": dst_crs, "nodata": dst_nodata,
        "compress": "deflate", "tiled": True,
    }
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out)
        dst.update_tags(**provenance)
    return provenance


def _transformation(src_crs, dst_crs) -> str:
    """The coordinate operation PROJ uses for this pair, with its stated accuracy."""
    t = Transformer.from_crs(CRS.from_wkt(src_crs.to_wkt()), CRS.from_wkt(dst_crs.to_wkt()), always_xy=True)
    accuracy = f"{t.accuracy} m" if t.accuracy is not None and t.accuracy >= 0 else "unknown"
    return f"{t.description} (accuracy {accuracy})"
