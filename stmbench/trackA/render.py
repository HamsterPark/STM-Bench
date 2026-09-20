"""Track A rendering: one ``.sxm`` frame → 8-bit PNG bytes + a header summary.

The picture a model sees is the guide's three-step rendering (data guide §5/§6):

1. **plane fit** — least-squares plane over the finite pixels, subtracted;
2. **row-median align** — every acquired row minus its own median (kills the
   line-to-line DC jumps that dominate raw Z);
3. **MAD stretch** — clip to ``median ± k·MAD`` and map to 0…255.

Rows controller never acquired are NaN in the file (``frame_validity.acquired_row_mask``
convention) and render **black**; ``acq_rows`` / ``n_rows`` in the summary say how
much of the frame is real. Nothing here needs PIL: the PNG is written with ``zlib``.

Reading goes through ``mast.io.nanonis_files`` (``read_sxm`` + ``sxm_oriented_frames``)
so backward frames are un-mirrored and ``:SCAN_DIR: up`` files are put top-first
exactly as MAST does — the only orientation rule in the tree.

Header summary (``header_summary``): size (nm), pixels, bias (V), setpoint (A), scan
speed (nm/s and s/line), angle, direction, optional ``rec_time``, optional ``comment``.
For **T4** the caller passes ``include_comment=False`` — 84 % of source comments carry
the working point that T4 asks the model to guess (guide §3.4 / §5 T4).

Data root: big data lives outside the repo; :func:`stmbench.paths.data_root`
(``STM_BENCH_DATA``) is the only place a machine-specific path is allowed to come from.
"""
from __future__ import annotations

import datetime as _dt
import ntpath
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stmbench.paths import data_root, raw_mirror_root  # noqa: F401 — data_root re-exported (cli.py imports it from here)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ── frame references ────────────────────────────────────────────────────────

def resolve_path(p: str | os.PathLike) -> Path:
    """Absolute paths pass through; relative ones — a manifest's ``rel_path`` and its
    ``context_paths_json`` entries — are taken under :func:`stmsim.paths.raw_mirror_root`
    (``$STM_BENCH_RAW``), the root the manifests were built against."""
    raw = str(p)
    q = Path(raw)
    if q.is_absolute():
        return q
    # ``Path.is_absolute`` follows the host OS, so a Windows drive-letter path
    # saved in a manifest must also be recognised when rendering on POSIX.
    if ntpath.isabs(raw):
        return Path(raw.replace("\\", "/"))
    return raw_mirror_root() / q


# ── reading ─────────────────────────────────────────────────────────────────

def load_scan(path: str | os.PathLike) -> dict:
    from mast.io.nanonis_files import read_sxm
    return read_sxm(str(resolve_path(path)))


def load_header(path: str | os.PathLike) -> dict:
    from mast.io.nanonis_files import read_sxm_header
    return read_sxm_header(str(resolve_path(path)))


def oriented(scan: dict, channel: str = "Z") -> dict:
    from mast.io.nanonis_files import sxm_oriented_frames
    return sxm_oriented_frames(scan, channel)


def _first_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return default


def _pair(value: Any) -> tuple[float | None, float | None]:
    parts = str(value if value is not None else "").split()
    a = _first_float(parts[0]) if parts else None
    b = _first_float(parts[1]) if len(parts) > 1 else a
    return a, b


def frame_start_time(header: dict) -> _dt.datetime | None:
    """``REC_DATE`` + ``REC_TIME`` = the frame's **start** (guide §3.3). ``None`` if unreadable
    or the 1904 placeholder (guide §3.4: treat as missing)."""
    d = str(header.get("rec_date", "")).strip()
    t = str(header.get("rec_time", "")).strip()
    if not d or not t:
        return None
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            when = _dt.datetime.strptime(f"{d} {t}", fmt)
            break
        except ValueError:
            continue
    else:
        return None
    if when.year <= 1904:
        return None
    return when


def header_summary(header: dict, *, include_comment: bool = True, include_time: bool = True) -> dict:
    """JSON-friendly summary of what an operator reads off the controller panel."""
    w_m, h_m = _pair(header.get("scan_range"))
    t_fwd, t_bwd = _pair(header.get("scan_time"))
    px = header.get("scan_pixels") or [None, None]
    nx = int(px[0]) if px and px[0] is not None else None
    ny = int(px[1]) if len(px) > 1 and px[1] is not None else None
    out: dict[str, Any] = {
        "size_nm": [round(w_m * 1e9, 4) if w_m is not None else None,
                    round(h_m * 1e9, 4) if h_m is not None else None],
        "pixels": [nx, ny],
        "bias_v": _first_float(header.get("bias")),
        "setpoint_a": _first_float(header.get("z-controller>setpoint")),
        "line_time_s": t_fwd,
        "scan_speed_nm_per_s": (w_m * 1e9 / t_fwd) if (w_m and t_fwd) else None,
        "angle_deg": _first_float(header.get("scan_angle")),
        "scan_dir": str(header.get("scan_dir", "")).strip().lower() or None,
        "z_controller_on": _zctrl_on(header),
    }
    if include_time:
        when = frame_start_time(header)
        out["rec_time"] = when.isoformat(sep=" ") if when else None
    if include_comment:
        out["comment"] = str(header.get("comment", "") or "")
    return out


def _zctrl_on(header: dict) -> bool | None:
    status = str(header.get("z-controller>controller_status", "")).strip().upper()
    if status in ("ON", "OFF"):
        return status == "ON"
    raw = str(header.get("z-controller", ""))
    parts = raw.split("\t")
    if len(parts) >= 2 and parts[1].strip() in ("0", "1"):
        return parts[1].strip() == "1"
    return None


# ── the three steps ─────────────────────────────────────────────────────────

def acquired_rows(a: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that are entirely finite (controller' saved-file convention)."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return np.zeros(0, dtype=bool)
    return np.isfinite(a).all(axis=1)


def plane_fit(a: np.ndarray) -> np.ndarray:
    """Subtract the least-squares plane fitted on the finite pixels. NaNs stay NaN."""
    a = np.asarray(a, dtype=float)
    fin = np.isfinite(a)
    if fin.sum() < 3:
        return a.copy()
    ny, nx = a.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    x = ii[fin].astype(float)
    y = jj[fin].astype(float)
    z = a[fin]
    g = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(g, z, rcond=None)
    plane = coef[0] * ii + coef[1] * jj + coef[2]
    out = a - plane
    out[~fin] = np.nan
    return out


def row_median_align(a: np.ndarray) -> np.ndarray:
    """Subtract each row's median (rows with no finite pixel are left as NaN)."""
    a = np.asarray(a, dtype=float).copy()
    for j in range(a.shape[0]):
        row = a[j]
        fin = np.isfinite(row)
        if fin.any():
            a[j, fin] = row[fin] - np.median(row[fin])
    return a


def mad_stretch(a: np.ndarray, k: float = 3.0) -> tuple[np.ndarray, dict]:
    """Clip to ``median ± k·MAD`` and map onto [0, 1]. Returns (image01, stats)."""
    a = np.asarray(a, dtype=float)
    fin = np.isfinite(a)
    stats: dict[str, float | None] = {"median": None, "mad": None, "lo": None, "hi": None}
    if not fin.any():
        return np.full(a.shape, np.nan), stats
    v = a[fin]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad <= 0:
        # a perfectly flat frame (or a saturated one): fall back to the finite range
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1e-15
    else:
        lo, hi = med - k * mad, med + k * mad
    img = (a - lo) / (hi - lo)
    img = np.clip(img, 0.0, 1.0)
    img[~fin] = np.nan
    stats.update(median=med, mad=mad, lo=lo, hi=hi)
    return img, stats


def flatten_three_step(a: np.ndarray, k_mad: float = 3.0) -> tuple[np.ndarray, dict]:
    img, stats = mad_stretch(row_median_align(plane_fit(a)), k_mad)
    return img, stats


# ── PNG ─────────────────────────────────────────────────────────────────────

def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def png_bytes_gray(img01: np.ndarray, *, nan_value: int = 0) -> bytes:
    """8-bit greyscale PNG from an array in [0, 1]; NaN pixels get ``nan_value``."""
    a = np.asarray(img01, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {a.shape}")
    ny, nx = a.shape
    u8 = np.where(np.isfinite(a), np.clip(np.rint(a * 255.0), 0, 255), nan_value).astype(np.uint8)
    raw = b"".join(b"\x00" + u8[j].tobytes() for j in range(ny))
    ihdr = struct.pack(">IIBBBBB", nx, ny, 8, 0, 0, 0, 0)
    return (PNG_SIGNATURE + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))


def png_shape(png: bytes) -> tuple[int, int]:
    """(height, width) read back from the IHDR — a cheap self-check without PIL."""
    if png[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG")
    w, h = struct.unpack(">II", png[16:24])
    return h, w


# ── public API ──────────────────────────────────────────────────────────────

@dataclass
class RenderedFrame:
    png: bytes
    summary: dict
    shape: tuple[int, int]
    acq_rows: int
    n_rows: int
    stretch: dict = field(default_factory=dict)
    path: str | None = None

    @property
    def acq_frac(self) -> float:
        return self.acq_rows / self.n_rows if self.n_rows else 0.0

    def as_dict(self) -> dict:
        """Everything but the PNG — what goes into a prompt JSONL."""
        return {"summary": self.summary, "shape": list(self.shape), "acq_rows": self.acq_rows,
                "n_rows": self.n_rows, "acq_frac": round(self.acq_frac, 4), "path": self.path}


def render_array(a: np.ndarray, *, k_mad: float = 3.0, crop_to_acquired: bool = False) -> tuple[bytes, dict, int, int]:
    """Three steps + PNG on a bare array. Returns (png, stretch_stats, acq_rows, n_rows)."""
    a = np.asarray(a, dtype=float)
    mask = acquired_rows(a)
    n_rows = int(a.shape[0]) if a.ndim == 2 else 0
    acq = int(mask.sum())
    src = a[mask] if (crop_to_acquired and acq > 0) else a
    img, stats = flatten_three_step(src, k_mad)
    return png_bytes_gray(img), stats, acq, n_rows


def head_band(arr: np.ndarray, scan_dir: str | None, n: int) -> np.ndarray:
    """The first ``n`` **acquired** rows of a top-first frame, as a strip.

    Acquisition starts at the top for ``down`` scans and at the bottom for ``up`` scans, so the
    head band is the top-most or bottom-most ``n`` finite rows. This is the only part of the
    *current* frame a T1 model may see (schema ``head_rows``): showing the rest — or the NaN
    tail — would show it the label.
    """
    a = np.asarray(arr, dtype=float)
    idx = np.where(acquired_rows(a))[0]
    if idx.size == 0:
        return a[:0]
    rows = idx[-n:] if str(scan_dir or "").strip().lower() == "up" else idx[:n]
    return a[rows]


def render_frame(path_or_scan: str | os.PathLike | dict, *, channel: str = "Z", direction: str = "forward",
                 k_mad: float = 3.0, include_comment: bool = True, include_time: bool = True,
                 crop_to_acquired: bool = False, head_rows: int | None = None) -> RenderedFrame:
    """Render one frame. Accepts a path or an already-read ``read_sxm`` dict.

    ``head_rows=n`` renders only the first ``n`` acquired rows (see :func:`head_band`) and
    withholds ``acq_rows`` / ``acq_frac`` from the summary — for the frame whose outcome is
    the label (T1).
    """
    path: str | None
    if isinstance(path_or_scan, dict):
        scan, path = path_or_scan, None
    else:
        path = str(resolve_path(path_or_scan))
        scan = load_scan(path)
    fr = oriented(scan, channel)
    arr = fr.get(direction)
    if arr is None:
        arr = fr.get("forward")
    if arr is None:
        raise ValueError(f"channel {channel!r} has no {direction} frame in {path or '<scan dict>'}")
    header = scan.get("header") or {}
    summary = header_summary(header, include_comment=include_comment, include_time=include_time)
    if head_rows is not None:
        strip = head_band(arr, header.get("scan_dir"), int(head_rows))
        if strip.shape[0] == 0:
            raise ValueError(f"no acquired rows to show in {path or '<scan dict>'}")
        png, stats, _, _ = render_array(strip, k_mad=k_mad)
        n_rows = int(np.asarray(arr).shape[0])
        shown = int(strip.shape[0])
        summary["head_rows_shown"] = shown
        summary["rows_total"] = n_rows
        return RenderedFrame(png=png, summary=summary, shape=(shown, int(strip.shape[1])),
                             acq_rows=shown, n_rows=n_rows, stretch=stats, path=path)
    png, stats, acq, n_rows = render_array(arr, k_mad=k_mad, crop_to_acquired=crop_to_acquired)
    summary["acq_rows"] = acq
    summary["acq_frac"] = round(acq / n_rows, 4) if n_rows else 0.0
    return RenderedFrame(png=png, summary=summary, shape=tuple(int(s) for s in np.asarray(arr).shape),
                         acq_rows=acq, n_rows=n_rows, stretch=stats, path=path)


def render_sequence(paths: Iterable[str | os.PathLike], **kw) -> list[RenderedFrame]:
    return [render_frame(p, **kw) for p in paths]


def render_settings_for_task(task: str) -> dict:
    """Per-task rendering switches. T4 must not see COMMENT (it carries the answer)."""
    task = str(task).upper()
    if task == "T4":
        return {"include_comment": False, "include_time": False}
    if task == "T1":
        # timestamps of the glances seen so far are legitimate input (guide §5 T1);
        # the *next* frame's time / file number never enters — the manifest excludes it.
        return {"include_comment": True, "include_time": True}
    return {"include_comment": True, "include_time": True}
