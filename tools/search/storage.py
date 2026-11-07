"""Score-surface storage (PROJECT.md §7: phase scores stay available).

Modes ([storage] surfaces):
  all              every surface: S_<jj>[centre, theta, iy, ix] for every spacing j
  winning_spacing  per centre, only the surfaces of its winning spacing:
                   S_<jj>[centre, theta, iy, ix] with centre_ids_<jj> naming the centres
  none             nothing; per-centre summaries are still exported

Layout under <run>/scores/:
    manifest.json                     axes, offsets and conventions
    block_<id>_batch_<n>.npz          one file per search batch
Offsets ox (east-positive) and oy (north-positive) are whole pixels; the
translation is (ox * px, oy * py) metres from the centre pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .hypotheses import SearchSpace

BYTES_PER_SCORE = 4  # float32, before compression


class SurfaceStore:
    def __init__(self, directory: str | Path, space: SearchSpace, mode: str):
        self.directory = Path(directory)
        self.space = space
        self.mode = mode
        self.files: list[str] = []
        self.bytes_written = 0
        if mode != "none":
            self.directory.mkdir(parents=True, exist_ok=True)

    def sink(self, block_id: int, centre_ids: np.ndarray):
        """Callback for LocalSearch.run that writes each batch of one block."""

        def write(batch_index, centre_slice, surfaces, best_spacing):
            if self.mode == "none":
                return
            ids = centre_ids[centre_slice]
            arrays = {}
            if self.mode == "all":
                arrays["centre_ids"] = ids
                arrays.update({f"S_{j:02d}": s for j, s in surfaces.items()})
            else:
                for j, s in surfaces.items():
                    sel = best_spacing == j
                    if sel.any():
                        arrays[f"centre_ids_{j:02d}"] = ids[sel]
                        arrays[f"S_{j:02d}"] = s[sel]
            name = f"block_{block_id:03d}_batch_{batch_index:04d}.npz"
            path = self.directory / name
            np.savez_compressed(path, **arrays)
            self.files.append(name)
            self.bytes_written += path.stat().st_size

        return write

    def estimate_bytes(self, n_centres: int) -> int:
        """Uncompressed size of the surfaces this mode keeps for n_centres."""
        per_spacing = [len(self.space.thetas) * self.space.n_translations(j) for j in range(len(self.space.spacings))]
        if self.mode == "all":
            return n_centres * sum(per_spacing) * BYTES_PER_SCORE
        if self.mode == "winning_spacing":
            return int(n_centres * np.mean(per_spacing) * BYTES_PER_SCORE)
        return 0

    def write_manifest(self) -> None:
        if self.mode == "none":
            return
        space = self.space
        manifest = {
            "mode": self.mode,
            "array": "S_<jj>[centre, theta, iy, ix], float32, NaN where the centre band is unavailable",
            "thetas_deg": list(space.thetas),
            "spacings_m": list(space.spacings),
            "pixel_size_m": [space.px, space.py],
            "translation_box": space.box,
            "offsets": {
                f"{j:02d}": {"ox": space.offsets(j)[0].tolist(), "oy": space.offsets(j)[1].tolist()}
                for j in range(len(space.spacings))
            },
            "offset_convention": "ox east-positive, oy north-positive, whole pixels from the centre pixel",
            "files": self.files,
            "bytes_on_disk": self.bytes_written,
        }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
