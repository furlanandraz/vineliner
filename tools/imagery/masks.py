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
    """
    shape = arrays[0].shape[1:]
    valid = np.ones(shape, dtype=bool)
    if nodata_value is None:
        return valid
    for data in arrays:
        if data.shape[1:] != shape:
            raise ValueError("input rasters must share one grid")
        hit = data == nodata_value
        if rule == "any_band":
            valid &= ~hit.any(axis=0)
        elif rule == "all_bands":
            valid &= ~hit.all(axis=0)
        else:
            raise ValueError(f"unknown nodata rule {rule!r}")
    return valid
