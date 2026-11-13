"""On-disk kernel cache (PROJECT.md §6.1).

Kernels depend only on the template geometry (line indices, line length, band
width), the rotations, spacings, pixel size and coverage method. They are
saved once under a key made from exactly those values and loaded on later runs
with the same values, which skips the build (about 20 s for the Bordeaux
settings). Template weights do not affect kernels and are not part of the key.

The key also contains CACHE_VERSION. Bump it whenever kernel rasterisation
changes, so stale files are never read. A file that fails to load or whose
stored key does not match is rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .geometry import Template
from .kernels import Kernel, KernelBank

CACHE_VERSION = 1


def cache_key(template: Template, thetas, spacings, px: float, py: float, coverage: str, subpixel_samples) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "indices": list(template.indices),
        "line_length_m": round(float(template.line_length_m), 9),
        "band_width_m": round(float(template.band_width_m), 9),
        "thetas": [round(float(t), 9) for t in thetas],
        "spacings": [round(float(d), 9) for d in spacings],
        "pixel_size_m": [round(float(px), 12), round(float(py), 12)],
        "coverage": coverage,
        "subpixel_samples": subpixel_samples if coverage == "sampled" else None,
    }


def cache_path(cache_dir: Path, key: dict) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"kernels_{digest}.npz"


def save_bank(path: Path, bank: KernelBank, key: dict) -> None:
    drow, dcol, weight, sizes = [], [], [], []
    for per_theta in bank._kernels:
        for per_d in per_theta:
            for kern in per_d:
                drow.append(kern.drow)
                dcol.append(kern.dcol)
                weight.append(kern.weight)
                sizes.append(kern.weight.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        key=np.array(json.dumps(key, sort_keys=True)),
        drow=np.concatenate(drow).astype(np.int32),
        dcol=np.concatenate(dcol).astype(np.int32),
        weight=np.concatenate(weight).astype(np.float32),
        sizes=np.asarray(sizes, dtype=np.int64),
    )
    tmp.replace(path)  # never leave a half-written cache file behind


def load_bank(path: Path, key: dict, template: Template, thetas, spacings, px, py, coverage, subpixel_samples) -> KernelBank | None:
    try:
        with np.load(path) as f:
            if str(f["key"]) != json.dumps(key, sort_keys=True):
                return None
            drow, dcol, weight, sizes = f["drow"], f["dcol"], f["weight"], f["sizes"]
    except (OSError, KeyError, ValueError):
        return None
    n_t, n_d, n_k = len(thetas), len(spacings), len(template.indices)
    if sizes.size != n_t * n_d * n_k:
        return None
    starts = np.concatenate([[0], np.cumsum(sizes)])
    centre = template.indices.index(0)
    kernels, i = [], 0
    for _t in range(n_t):
        per_theta = []
        for j in range(n_d):
            per_d = []
            for b in range(n_k):
                s, e = starts[i], starts[i + 1]
                i += 1
                if b == centre and j > 0:
                    per_d.append(per_theta[0][b])  # line 0 does not depend on d: share it, as a fresh build does
                else:
                    per_d.append(Kernel(drow=drow[s:e].copy(), dcol=dcol[s:e].copy(), weight=weight[s:e].copy()))
            per_theta.append(per_d)
        kernels.append(per_theta)
    return KernelBank(template, thetas, spacings, px, py, coverage, subpixel_samples, kernels=kernels)


def load_or_build(template: Template, thetas, spacings, px: float, py: float, coverage: str, subpixel_samples, cache_dir: Path | None):
    """Return (KernelBank, info). With cache_dir None the bank is always built."""
    t0 = time.perf_counter()
    if cache_dir is None:
        bank = KernelBank(template, thetas, spacings, px, py, coverage, subpixel_samples)
        return bank, {"cache": "off", "seconds": round(time.perf_counter() - t0, 2)}
    key = cache_key(template, thetas, spacings, px, py, coverage, subpixel_samples)
    path = cache_path(cache_dir, key)
    if path.exists():
        bank = load_bank(path, key, template, thetas, spacings, px, py, coverage, subpixel_samples)
        if bank is not None:
            return bank, {"cache": "hit", "path": str(path), "seconds": round(time.perf_counter() - t0, 2)}
    bank = KernelBank(template, thetas, spacings, px, py, coverage, subpixel_samples)
    save_bank(path, bank, key)
    return bank, {"cache": "miss (built and saved)", "path": str(path), "seconds": round(time.perf_counter() - t0, 2)}
