"""Valid-data mask (PROJECT.md §3, §8, §14)."""

from __future__ import annotations

import numpy as np


def valid_mask(arrays: list[np.ndarray], nodata_value: float | None, rule: str) -> np.ndarray:
    """Pixels usable as image evidence, as a (rows, cols) bool array.

    arrays: one (bands, rows, cols) array per input file, all on the same grid.
    rule:
      "any_band"  - a pixel is invalid when any band of any file equals nodata_value.
                    On the Bordeaux pair, 255 marks NoData and saturated outliers
                    (water glint, white roofs), which are never row evidence.
      "all_bands" - a pixel is invalid when every band of one file equals nodata_value.
    Non-finite values (NaN, the usual NoData of float rasters, and infinities)
    count as NoData under the same rule, whatever nodata_value is.
    """
    if rule not in ("any_band", "all_bands"):
        raise ValueError(f"unknown nodata rule {rule!r}")
    shape = arrays[0].shape[1:]
    valid = np.ones(shape, dtype=bool)
    for data in arrays:
        if data.shape[1:] != shape:
            raise ValueError("input rasters must share one grid")
        hit = np.zeros(data.shape, dtype=bool)
        if nodata_value is not None and not np.isnan(nodata_value):
            hit |= data == nodata_value
        if np.issubdtype(data.dtype, np.floating):
            hit |= ~np.isfinite(data)
        valid &= ~(hit.any(axis=0) if rule == "any_band" else hit.all(axis=0))
    return valid
