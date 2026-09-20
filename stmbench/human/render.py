"""Pictures of what the instrument saved, for the person in mode H.

Frames come back through MAST's own reader (``read_sxm`` + ``sxm_oriented_frames``), so
the orientation convention is the one the judge and the analysis skills use: row 0 of the
oriented array is the frame's high-v edge (DESIGN.md §8). :func:`pixel_to_scan_nm` is the
inverse of the judge's footprint test (``trackB.claims.frame_covering``) and is pinned to it
by a test, because a reported position is only worth something in the scan frame the judge
reads.

Rendering is deliberately bare: the frame as an image with no axes or margins, so the
browser can map a mouse position to a pixel and from there to scan coordinates. Scale, unit
and geometry travel separately as JSON (:func:`frame_meta`).
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np

FLATTEN_MODES = ("plane", "line", "none")
DEFAULT_CMAP = "gray"


# ── geometry ────────────────────────────────────────────────────────────────
def _floats(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
        return out
    out = []
    for tok in str(value or "").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def frame_geometry(header: dict) -> dict | None:
    """``{cx_nm, cy_nm, w_nm, h_nm, angle_deg, nx, ny}`` from a parsed .sxm header; None
    when the geometry keys are missing."""
    rng = _floats(header.get("scan_range"))
    off = _floats(header.get("scan_offset"))
    px = header.get("scan_pixels")
    px = [int(v) for v in px] if isinstance(px, (list, tuple)) else [int(v) for v in _floats(px)]
    if len(rng) < 1 or len(off) < 2 or len(px) < 2:
        return None
    w_m = rng[0]
    h_m = rng[1] if len(rng) > 1 else rng[0]
    ang = _floats(header.get("scan_angle"))
    return {"cx_nm": off[0] * 1e9, "cy_nm": off[1] * 1e9, "w_nm": w_m * 1e9, "h_nm": h_m * 1e9,
            "angle_deg": ang[0] if ang else 0.0, "nx": px[0], "ny": px[1]}


def pixel_to_scan_nm(geom: dict, col: float, row: float) -> tuple[float, float]:
    """Oriented-array pixel (``col`` along the fast axis, ``row`` 0 = the frame's high-v
    edge) → scan-frame nm. Pixel centres sit at half-integers; ``col``/``row`` may be
    fractional (a mouse position)."""
    w, h = float(geom["w_nm"]), float(geom["h_nm"])
    nx, ny = int(geom["nx"]), int(geom["ny"])
    u = -w / 2 + (float(col) + 0.5) * w / nx
    v = h / 2 - (float(row) + 0.5) * h / ny
    a = math.radians(float(geom.get("angle_deg", 0.0) or 0.0))
    c, s = math.cos(a), math.sin(a)
    return float(geom["cx_nm"]) + c * u - s * v, float(geom["cy_nm"]) + s * u + c * v


# ── files ───────────────────────────────────────────────────────────────────
def _header(path: Path) -> dict:
    from mast.io.nanonis_files import read_sxm_header
    return read_sxm_header(str(path)) or {}


def list_files(session_dir: str | Path) -> list[dict]:
    """Every ``.sxm`` / ``.dat`` in the session directory, newest first, with the
    metadata a person wants at a glance (geometry, bias, setpoint, time)."""
    d = Path(session_dir)
    if not d.is_dir():
        return []
    out: list[dict] = []
    for p in d.iterdir():
        if p.suffix.lower() not in (".sxm", ".dat"):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rec: dict = {"name": p.name, "kind": p.suffix.lower()[1:], "mtime": st.st_mtime, "bytes": st.st_size}
        try:
            if rec["kind"] == "sxm":
                rec.update(frame_meta(p))
            else:
                rec.update(dat_meta(p))
        except Exception as exc:  # noqa: BLE001 — a half-written file is listed, not fatal
            rec["error"] = f"{type(exc).__name__}: {exc}"
        out.append(rec)
    out.sort(key=lambda r: (r["mtime"], r["name"]), reverse=True)
    return out


def frame_meta(path: str | Path) -> dict:
    h = _header(Path(path))
    geom = frame_geometry(h) or {}
    bias = _floats(h.get("bias"))
    setp = _floats(h.get("z-controller>setpoint"))
    return {**geom, "bias_v": bias[0] if bias else None, "setpoint_a": setp[0] if setp else None,
            "scan_dir": str(h.get("scan_dir", "")).strip().lower() or None,
            "rec_time": f"{str(h.get('rec_date', '')).strip()} {str(h.get('rec_time', '')).strip()}".strip(),
            "channels": list(h.get("channel_names") or [])}


def dat_meta(path: str | Path) -> dict:
    from mast.io.nanonis_files import read_dat
    d = read_dat(str(path))
    hdr = d.get("header") or {}
    cols = d.get("columns") or {}
    names = list(cols)
    n = int(len(next(iter(cols.values())))) if cols else 0

    def _f(key):
        v = _floats(hdr.get(key))
        return v[0] if v else None

    return {"experiment": hdr.get("Experiment"), "columns": names, "n_points": n,
            "x_nm": (_f("X (m)") * 1e9) if _f("X (m)") is not None else None,
            "y_nm": (_f("Y (m)") * 1e9) if _f("Y (m)") is not None else None,
            "bias_v": _f("Bias>Bias (V)"), "setpoint_a": _f("Z-Controller>Setpoint"),
            "date": hdr.get("Date")}


# ── frames ──────────────────────────────────────────────────────────────────
def _oriented(path: Path, channel: str, direction: str) -> tuple[np.ndarray | None, dict]:
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    scan = read_sxm(str(path))
    fr = sxm_oriented_frames(scan, channel)
    img = fr.get("backward" if direction == "backward" else "forward")
    return (None if img is None else np.asarray(img, dtype=float)), fr


def flatten(img: np.ndarray, mode: str = "plane") -> np.ndarray:
    """Plane subtraction (least squares on the finite pixels), per-line median, or none.
    NaN rows (an aborted scan) stay NaN."""
    z = np.array(img, dtype=float)
    if mode == "none" or z.ndim != 2:
        return z
    if mode == "line":
        med = np.nanmedian(z, axis=1, keepdims=True)
        med = np.where(np.isfinite(med), med, 0.0)
        return z - med
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ok = np.isfinite(z)
    if ok.sum() < 3:
        return z
    a = np.column_stack([xx[ok], yy[ok], np.ones(int(ok.sum()))])
    coef, *_ = np.linalg.lstsq(a, z[ok], rcond=None)
    return z - (coef[0] * xx + coef[1] * yy + coef[2])


def _unit_scale(channel: str, span: float) -> tuple[float, str]:
    name = str(channel).strip().lower()
    if name == "z":
        return (1e12, "pm") if span < 1e-9 else (1e9, "nm")
    if name == "current":
        return (1e12, "pA") if span < 1e-9 else (1e9, "nA")
    if name in ("frequency shift",):
        return 1.0, "Hz"
    return 1.0, ""


def _to_rgb(norm: np.ndarray, cmap: str) -> np.ndarray:
    try:
        import matplotlib
        cm = matplotlib.colormaps[cmap]
        rgba = cm(np.where(np.isfinite(norm), norm, 0.0))
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001 — no matplotlib / unknown map: greyscale
        g = (np.where(np.isfinite(norm), norm, 0.0) * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    nan = ~np.isfinite(norm)
    if nan.any():
        rgb[nan] = (40, 40, 60)          # rows never acquired: a flat dark blue, not "black = low"
    return rgb


def array_png(z: np.ndarray, *, cmap: str = DEFAULT_CMAP, max_px: int | None = None,
               clip_pct: float = 0.5, min_span: float = 0.0, limits: tuple[float, float] | None = None,
               mask: np.ndarray | None = None, mask_rgb: tuple[int, int, int] = (20, 20, 28)
               ) -> tuple[bytes, float, float]:
    """PNG bytes of a 2-D array (row 0 at the top; NaN = never acquired) and the colour
    limits ``(lo, hi)`` it used: ``limits`` when given (two pictures on one scale), else the
    ``clip_pct`` percentiles, widened to ``min_span`` so a nearly flat array is not stretched
    into noise. Pixels where ``mask`` is true are painted ``mask_rgb`` (outlines on a map)."""
    from PIL import Image

    z = np.asarray(z, dtype=float)
    finite = z[np.isfinite(z)]
    if limits is not None:
        lo, hi = float(limits[0]), float(limits[1])
    elif finite.size:
        lo, hi = (float(v) for v in np.percentile(finite, [clip_pct, 100.0 - clip_pct]))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max() + 1e-30)
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    norm = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    rgb = _to_rgb(norm, cmap)
    if mask is not None:
        rgb[np.asarray(mask, dtype=bool)] = mask_rgb
    im = Image.fromarray(rgb, mode="RGB")
    ny, nx = z.shape
    if max_px and max(nx, ny) > int(max_px):
        f = int(max_px) / max(nx, ny)
        im = im.resize((max(1, int(nx * f)), max(1, int(ny * f))), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), lo, hi


def frame_array(path: str | Path, *, channel: str = "Z", direction: str = "forward",
                flatten_mode: str = "plane") -> tuple[np.ndarray, dict]:
    """One frame as the flattened oriented array (row 0 = the high-v edge) and MAST's
    frame dict."""
    img, fr = _oriented(Path(path), channel, direction)
    if img is None:
        raise FileNotFoundError(f"no {channel} {direction} frame in {path}")
    return flatten(img, flatten_mode if flatten_mode in FLATTEN_MODES else "plane"), fr


def render_frame(path: str | Path, *, channel: str = "Z", direction: str = "forward",
                 flatten_mode: str = "plane", cmap: str = DEFAULT_CMAP, max_px: int | None = None,
                 clip_pct: float = 0.5) -> tuple[bytes, dict]:
    """PNG bytes of one frame (no axes, no margins) plus its display meta:
    ``{lo, hi, unit, scale, nx, ny, channel, direction, flatten}``. ``lo``/``hi`` are the
    colour limits in display units (after ``clip_pct`` percentile clipping)."""
    z, fr = frame_array(path, channel=channel, direction=direction, flatten_mode=flatten_mode)
    png, lo, hi = array_png(z, cmap=cmap, max_px=max_px, clip_pct=clip_pct)
    ny, nx = z.shape
    scale, unit = _unit_scale(channel, float(hi - lo))
    meta = {"lo": float(lo) * scale, "hi": float(hi) * scale, "unit": unit, "scale": scale,
            "nx": int(nx), "ny": int(ny), "channel": channel, "direction": direction,
            "flatten": flatten_mode, "nm_per_px": fr.get("nm_per_px"), "rec_time": fr.get("rec_time")}
    return png, meta


# ── spectra ─────────────────────────────────────────────────────────────────
def dat_columns(path: str | Path) -> dict:
    from mast.io.nanonis_files import read_dat
    d = read_dat(str(path))
    cols = {k: [float(x) for x in np.asarray(v, dtype=float)] for k, v in (d.get("columns") or {}).items()}
    return {"header": {k: str(v) for k, v in (d.get("header") or {}).items()}, "columns": cols}


def render_dat(path: str | Path, *, x: str | None = None, y: str | None = None,
               width_px: int = 640, height_px: int = 420) -> bytes:
    """One spectrum as a PNG line plot: ``y`` (default: the second column) against ``x``
    (default: the first, the sweep axis)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from mast.io.nanonis_files import read_dat

    d = read_dat(str(path))
    cols = d.get("columns") or {}
    names = list(cols)
    if not names:
        raise ValueError(f"no data columns in {path}")
    xn = x if x in cols else names[0]
    yn = y if y in cols else (names[1] if len(names) > 1 else names[0])
    xs = np.asarray(cols[xn], dtype=float)
    ys = np.asarray(cols[yn], dtype=float)
    fig = Figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, lw=1.2)
    ax.set_xlabel(xn)
    ax.set_ylabel(yn)
    ax.set_title(Path(path).name, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()
