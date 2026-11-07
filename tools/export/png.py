"""Minimal PNG writer (standard library only), for quick-look images."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, image: np.ndarray) -> Path:
    """Write a (rows, cols) gray or (rows, cols, 3) RGB uint8 array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ascontiguousarray(image, dtype=np.uint8)
    if img.ndim == 2:
        colour_type, channels = 0, 1
    elif img.ndim == 3 and img.shape[2] == 3:
        colour_type, channels = 2, 3
    else:
        raise ValueError("expected a gray or RGB image")
    h, w = img.shape[:2]
    raw = np.zeros((h, 1 + w * channels), dtype=np.uint8)  # filter byte 0 per scanline
    raw[:, 1:] = img.reshape(h, w * channels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))
    return path
