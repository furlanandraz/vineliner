"""GeoTIFF reading and writing through rasterio (PROJECT.md §3).

This is a file adapter. A QGIS plugin can replace it with one built on the
QGIS/GDAL API, as long as it returns the same Raster structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio

from ..geo import Grid


@dataclass
class Raster:
    data: np.ndarray  # (bands, rows, cols), file dtype
    grid: Grid
    band_names: tuple[str, ...]  # meaning of each band, from the config
    nodata: float | None  # NoData value declared in the file
    path: Path
    metadata: dict = field(default_factory=dict)

    def band(self, name: str) -> np.ndarray:
        return self.data[self.band_names.index(name)]


def read_raster(path: str | Path, band_names: tuple[str, ...]) -> Raster:
    """Read every band. Band meaning comes from band_names, never from array position alone."""
    path = Path(path)
    with rasterio.open(path) as src:
        if len(band_names) != src.count:
            raise ValueError(f"{path} has {src.count} bands but the config names {len(band_names)}: {band_names}")
        if src.crs is None:
            raise ValueError(f"{path} has no CRS; assign one explicitly before use (PROJECT.md §3)")
        if not src.crs.is_projected:
            raise ValueError(f"{path} CRS is not projected; reproject to a metric CRS first (PROJECT.md §3)")
        grid = Grid.from_affine(tuple(src.transform)[:6], src.width, src.height, src.crs.to_wkt())
        data = src.read()
        metadata = describe(src)
    return Raster(data=data, grid=grid, band_names=tuple(band_names), nodata=metadata["nodata"], path=path, metadata=metadata)


def describe_raster(path: str | Path) -> dict:
    with rasterio.open(path) as src:
        return describe(src)


def describe(src) -> dict:
    units = src.crs.linear_units if src.crs is not None else None
    return {
        "path": str(src.name),
        "width": src.width,
        "height": src.height,
        "count": src.count,
        "dtypes": list(src.dtypes),
        "nodata": src.nodata,
        "transform": list(tuple(src.transform)[:6]),
        "pixel_size": [src.transform.a, -src.transform.e],
        "bounds": list(src.bounds),
        "crs_epsg": src.crs.to_epsg() if src.crs is not None else None,
        "crs_units": units,
        "crs_wkt": src.crs.to_wkt() if src.crs is not None else None,
        "band_descriptions": list(src.descriptions),
        "colorinterp": [c.name for c in src.colorinterp],
        "compression": src.compression.name if src.compression else None,
        "tags": src.tags(),
    }


def write_raster(path: str | Path, array: np.ndarray, grid: Grid, nodata: float | None = None) -> Path:
    """Write a 2-D or (bands, rows, cols) array with the grid's transform and CRS preserved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array[np.newaxis] if array.ndim == 2 else array
    if data.dtype == bool:
        data = data.astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": data.shape[0],
        "dtype": data.dtype.name,
        "transform": rasterio.Affine(*grid.affine),
        "crs": rasterio.crs.CRS.from_wkt(grid.crs_wkt) if grid.crs_wkt else None,
        "compress": "deflate",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path
