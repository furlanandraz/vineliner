"""Row-evidence raster V in [0, 1] (PROJECT.md §5).

Polarity: high V always means stronger row evidence.

V is built from up to two components, each a raw response that is then
scaled to [0, 1] over a scaling region (one block, or all blocks):

  gray component (from the configured RGB/NIR bands, averaged)
    black_tophat   closing(gray) - gray with a square of side tophat_size_m.
                   Keeps narrow dark features (rows on bright soil) and removes
                   broad brightness changes. Scaled by a percentile: 0 -> 0,
                   P(scale_percentile) -> 1.
    gray_stretch   -gray, scaled between the stretch percentiles.

  ndvi component (optional, from an NDVI raster; vines are greener than soil)
    white_tophat   ndvi - opening(ndvi): narrow bright (green) features. Scaled
                   like black_tophat.
    stretch        ndvi, scaled between the stretch percentiles.

  V = (1 - ndvi_weight) * V_gray + ndvi_weight * V_ndvi

Invalid pixels are replaced by the median valid value before filtering, so
they cannot leak into neighbours, and V is 0 there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

GRAY_METHODS = ("black_tophat", "gray_stretch")
NDVI_METHODS = ("white_tophat", "stretch")


@dataclass
class Response:
    """An unscaled response (higher = more row-like) and how to scale it."""

    name: str
    raw: np.ndarray
    scaling: str  # "percentile" (0 -> 0, P -> 1) or "stretch" (P_lo -> 0, P_hi -> 1)
    info: dict = field(default_factory=dict)


def gray_from_bands(bands: list[np.ndarray]) -> np.ndarray:
    """Mean of the selected bands, as float32."""
    return np.mean([b.astype(np.float32) for b in bands], axis=0, dtype=np.float32)


def _filled(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32, copy=True)
    out[~valid] = np.median(image[valid])
    return out


def _size_px(size_m: float | None, pixel_size_m: float, method: str) -> int:
    if not size_m:
        raise ValueError(f"{method} needs a window size in metres")
    return max(3, int(round(size_m / pixel_size_m)) | 1)  # odd, at least 3


def gray_response(gray: np.ndarray, valid: np.ndarray, method: str, pixel_size_m: float, tophat_size_m: float | None) -> Response:
    img = _filled(gray, valid)
    if method == "black_tophat":
        size = _size_px(tophat_size_m, pixel_size_m, method)
        raw = ndimage.grey_closing(img, size=(size, size)) - img
        return Response("gray", raw, "percentile", {"method": method, "tophat_size_m": tophat_size_m, "tophat_size_px": size})
    if method == "gray_stretch":
        return Response("gray", -img, "stretch", {"method": method})
    raise ValueError(f"unknown gray evidence method {method!r}")


def ndvi_response(ndvi: np.ndarray, valid: np.ndarray, method: str, pixel_size_m: float, tophat_size_m: float | None) -> Response:
    img = _filled(ndvi, valid)
    if method == "white_tophat":
        size = _size_px(tophat_size_m, pixel_size_m, method)
        raw = img - ndimage.grey_opening(img, size=(size, size))
        return Response("ndvi", raw, "percentile", {"method": method, "tophat_size_m": tophat_size_m, "tophat_size_px": size})
    if method == "stretch":
        return Response("ndvi", img, "stretch", {"method": method})
    raise ValueError(f"unknown NDVI evidence method {method!r}")


def scale_response(resp: Response, region: np.ndarray, scale_percentile: float, stretch_percentiles: tuple[float, float]) -> tuple[np.ndarray, dict]:
    """Map a response to [0, 1] using statistics over region (valid pixels only)."""
    values = resp.raw[region]
    if values.size == 0:
        raise ValueError("no valid pixels in the scaling region")
    if resp.scaling == "percentile":
        lo, hi = 0.0, float(np.percentile(values, scale_percentile))
    else:
        lo, hi = (float(v) for v in np.percentile(values, stretch_percentiles))
    if hi <= lo:
        raise ValueError(f"{resp.name} evidence has no contrast in the scaling region")
    v = np.clip((resp.raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return v, {"lo": lo, "hi": hi}


def combine(components: list[tuple[np.ndarray, float]], valid: np.ndarray) -> np.ndarray:
    v = np.zeros_like(components[0][0])
    for array, weight in components:
        if weight:
            v += np.float32(weight) * array
    v[~valid] = 0.0
    return np.clip(v, 0.0, 1.0)


def row_evidence(
    gray: np.ndarray,
    valid: np.ndarray,
    region: np.ndarray,
    method: str,
    pixel_size_m: float,
    tophat_size_m: float | None = None,
    scale_percentile: float = 99.0,
    stretch_percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[np.ndarray, dict]:
    """Single-component V from gray, scaled over region. Kept for simple uses and tests."""
    resp = gray_response(gray, valid, method, pixel_size_m, tophat_size_m)
    v, scale = scale_response(resp, region & valid, scale_percentile, stretch_percentiles)
    v[~valid] = 0.0
    info = resp.info | scale | {"polarity": "high = row", "mean_in_region": float(v[region & valid].mean())}
    return v, info
