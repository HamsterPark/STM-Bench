"""Frames as images for an agent: a fixed, documented rendering, PNG-encoded without extra
dependencies (numpy + zlib).

Every condition of the benchmark shows a frame the same way, so a difference between agents is
not a difference in how their images were prepared:

1. topography (``z``): each line's median is subtracted (removes the slow-axis offsets a
   drifting, creeping scanner leaves) — current frames are shown as recorded;
2. the 0.5th–99.5th percentile range is mapped onto 0–255 (a few hot pixels do not wash it out);
3. the image is enlarged by an integer nearest-neighbour factor to at least ``min_px`` per side;
4. row 0 is the top of the frame (high y), column 0 its left (low x).

No scale bar or text is drawn into the image; the geometry travels as metadata.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

MIN_PX = 512


def display_array(a: np.ndarray, channel: str = "z") -> np.ndarray:
    """The 8-bit image of a frame array (steps 1–2 above)."""
    x = np.asarray(a, dtype=np.float64)
    if channel == "z":
        x = x - np.nanmedian(x, axis=1, keepdims=True)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(x.shape, np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1e-30
    y = np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.round(y * 255.0).astype(np.uint8)


def enlarge(img: np.ndarray, min_px: int = MIN_PX) -> tuple[np.ndarray, int]:
    """Integer nearest-neighbour enlargement to at least ``min_px`` on the shorter side."""
    k = max(1, int(np.ceil(min_px / max(1, min(img.shape)))))
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1), k


def png_gray(img: np.ndarray) -> bytes:
    """An 8-bit greyscale PNG of a 2-D uint8 array."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def frame_png(a: np.ndarray, channel: str = "z", min_px: int = MIN_PX) -> tuple[bytes, int]:
    """PNG bytes of a frame array and the enlargement factor used."""
    big, k = enlarge(display_array(a, channel), min_px)
    return png_gray(big), k


def read_png_gray(data: bytes) -> np.ndarray:
    """Decode a PNG written by :func:`png_gray` (for tests and checks)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, w, h = 8, 0, 0
    idat = b""
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[r * (w + 1) + 1:(r + 1) * (w + 1)] for r in range(h)]
    return np.frombuffer(b"".join(rows), np.uint8).reshape(h, w)
