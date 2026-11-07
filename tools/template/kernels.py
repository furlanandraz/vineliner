"""Template rasterisation into kernels (PROJECT.md §6.1).

Each template band is rasterised onto a local grid with the analysis raster's
pixel size and axes. The template centre sits on pixel (0, 0), a pixel
centre. Kernel value K(x) is the fraction of pixel x covered by the band.

Coverage methods:
  exact        fractional coverage by polygon-pixel clipping (Shapely). Default.
  sampled      fractional coverage by s x s sub-pixel point sampling.
  centre       binary: 1 where the pixel centre is inside the band (comparison only).
  all_touched  binary: 1 where the pixel overlaps the band (comparison only).

Kernels depend only on (theta, d, k, L, w), not on location, so they are
built once per run. Line 0 does not depend on d and is shared across spacings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from ..geo import across, along
from .geometry import Template

_EPS = 1e-12


@dataclass(frozen=True)
class Kernel:
    """Sparse kernel: offsets from the template-centre pixel, with coverage weights."""

    drow: np.ndarray  # int32
    dcol: np.ndarray  # int32
    weight: np.ndarray  # float32 in (0, 1]

    @property
    def total(self) -> float:
        return float(self.weight.sum(dtype=np.float64))

    @property
    def radius(self) -> int:
        if self.drow.size == 0:
            return 0
        return int(max(np.abs(self.drow).max(), np.abs(self.dcol).max()))


def rasterise_band(
    template: Template,
    theta_deg: float,
    spacing_m: float,
    k: int,
    px: float,
    py: float,
    coverage: str = "exact",
    subpixel_samples: int | None = None,
) -> Kernel:
    corners = template.band_corners(theta_deg, spacing_m, k)  # world offsets from the centre pixel
    cols = corners[:, 0] / px
    rows = -corners[:, 1] / py  # rows increase southward
    r_lo, r_hi = int(np.floor(rows.min() - 0.5)), int(np.ceil(rows.max() + 0.5))
    c_lo, c_hi = int(np.floor(cols.min() - 0.5)), int(np.ceil(cols.max() + 0.5))
    rr, cc = np.meshgrid(np.arange(r_lo, r_hi + 1), np.arange(c_lo, c_hi + 1), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    if coverage in ("exact", "all_touched"):
        boxes = shapely.box((cc - 0.5) * px, -(rr + 0.5) * py, (cc + 0.5) * px, -(rr - 0.5) * py)
        area = shapely.area(shapely.intersection(boxes, shapely.Polygon(corners))) / (px * py)
        weight = area if coverage == "exact" else (area > 1e-9).astype(float)
    elif coverage in ("sampled", "centre"):
        s = 1 if coverage == "centre" else int(subpixel_samples or 0)
        if s < 1:
            raise ValueError("sampled coverage needs subpixel_samples >= 1")
        sub = (np.arange(s) + 0.5) / s - 0.5
        sr, sc = np.meshgrid(sub, sub, indexing="ij")
        pr = rr[:, None] + sr.ravel()[None, :]
        pc = cc[:, None] + sc.ravel()[None, :]
        centre = template.band_centre(theta_deg, spacing_m, k)
        dx = pc * px - centre[0]
        dy = -pr * py - centre[1]
        u, n = along(theta_deg), across(theta_deg)
        a = dx * u[0] + dy * u[1]
        b = dx * n[0] + dy * n[1]
        inside = (np.abs(a) <= template.line_length_m / 2 + _EPS) & (np.abs(b) <= template.band_width_m / 2 + _EPS)
        weight = inside.mean(axis=1)
    else:
        raise ValueError(f"unknown coverage method {coverage!r}")

    keep = weight > 1e-9
    return Kernel(
        drow=rr[keep].astype(np.int32),
        dcol=cc[keep].astype(np.int32),
        weight=np.minimum(weight[keep], 1.0).astype(np.float32),
    )


class KernelBank:
    """Every kernel for every (theta, d, k) of a run, built once."""

    def __init__(
        self,
        template: Template,
        thetas: tuple[float, ...],
        spacings: tuple[float, ...],
        px: float,
        py: float,
        coverage: str,
        subpixel_samples: int | None = None,
    ):
        self.template = template
        self.thetas = tuple(thetas)
        self.spacings = tuple(spacings)
        self.px, self.py = px, py
        self.coverage = coverage
        self.subpixel_samples = subpixel_samples
        K = len(template.indices)
        self._kernels: list[list[list[Kernel]]] = []
        for theta in self.thetas:
            centre_kernel = rasterise_band(template, theta, self.spacings[0], 0, px, py, coverage, subpixel_samples)
            per_theta = []
            for d in self.spacings:
                per_theta.append([
                    centre_kernel if k == 0 else rasterise_band(template, theta, d, k, px, py, coverage, subpixel_samples)
                    for k in template.indices
                ])
            self._kernels.append(per_theta)
        self.n_bands = K
        self.radius = max(kern.radius for per_theta in self._kernels for per_d in per_theta for kern in per_d)

    def get(self, t: int, j: int) -> list[Kernel]:
        """The template's band kernels for rotation index t and spacing index j."""
        return self._kernels[t][j]

    def totals(self, j: int) -> np.ndarray:
        """(n_theta, n_bands) total kernel weight for spacing index j."""
        return np.array([[kern.total for kern in self._kernels[t][j]] for t in range(len(self.thetas))])

    def packed(self, j: int):
        """Concatenate all kernels of spacing j: (drow, dcol, weight, segment), segment = t * n_bands + band."""
        drow, dcol, weight, seg = [], [], [], []
        for t in range(len(self.thetas)):
            for b, kern in enumerate(self._kernels[t][j]):
                drow.append(kern.drow)
                dcol.append(kern.dcol)
                weight.append(kern.weight)
                seg.append(np.full(kern.weight.size, t * self.n_bands + b, dtype=np.int64))
        return np.concatenate(drow), np.concatenate(dcol), np.concatenate(weight), np.concatenate(seg)

    def provenance(self) -> dict:
        nnz = [kern.weight.size for per_theta in self._kernels for per_d in per_theta for kern in per_d]
        return {
            "coverage": self.coverage,
            "subpixel_samples": self.subpixel_samples,
            "line_length_m": self.template.line_length_m,
            "band_width_m": self.template.band_width_m,
            "pixel_size_m": [self.px, self.py],
            "n_kernels": len(nnz),
            "kernel_radius_px": self.radius,
            "nonzero_per_kernel": {"min": int(min(nnz)), "mean": float(np.mean(nnz)), "max": int(max(nnz))},
            "centre_pixel": "template centre on pixel (0, 0) centre",
        }
