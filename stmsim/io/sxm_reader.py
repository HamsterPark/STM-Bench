"""Read the controller files stmsim writes (``.sxm`` frames, ``.dat`` spectra) without MAST.

Written from the file layout that :mod:`stmsim.io.sxm_writer` and :mod:`stmsim.io.dat_writer`
document, and held to them by round-trip tests; files of the same controller format from a real
instrument read the same way. Two layers:

* :func:`read_sxm_header`, :func:`read_sxm`, :func:`read_dat` return what a file holds: the header
  fields, and the data blocks exactly as stored (acquisition order; backward blocks mirrored);
* :func:`oriented_frames` puts one channel of a frame into image orientation — backward block
  un-mirrored, row 0 the frame's top (high y) whatever the slow-axis direction — with its scale.

Header conventions: ``:KEY:`` lines become lower-case keys with spaces as underscores; a value is
the last non-empty line under its key (multi-line blocks such as ``:Z-CONTROLLER:`` keep their
value row); ``:SCAN_PIXELS:`` becomes ``[nx, ny]``; ``:DATA_INFO:`` becomes ``channel_names`` and
``channel_directions``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_END_MARKERS = (b"\x1a\x04", b"\\1A\\04")


def _split_header(content: bytes) -> tuple[str, bytes] | None:
    best = None
    for m in _END_MARKERS:
        i = content.find(m)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, m)
    if best is None:
        return None
    i, m = best
    return content[:i].decode("utf-8", errors="replace"), content[i + len(m):]


def parse_sxm_header(raw: str) -> dict:
    header: dict = {}
    key = ""
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith(":") and s.endswith(":") and len(s) > 1:
            key = s.strip(":").lower().replace(" ", "_")
            continue
        if not key:
            continue
        if key == "scan_pixels":
            parts = s.split()
            try:
                header[key] = [int(parts[0]), int(parts[1])]
            except (ValueError, IndexError):
                pass
        elif key == "data_info":
            header.setdefault("channel_names", [])
            header.setdefault("channel_directions", [])
            parts = s.split("\t")             # a row: Channel, Name, Unit, Direction, ...
            if len(parts) >= 2 and parts[1].strip() and parts[1].strip() != "Name":
                header["channel_names"].append(parts[1].strip())
                direction = parts[3].strip().lower() if len(parts) >= 4 else ""
                header["channel_directions"].append(direction or "both")
        else:
            header[key] = s
    return header


def read_sxm_header(path: str | Path, *, max_bytes: int = 1_048_576) -> dict:
    """The header alone (``{}`` if the end marker is not within ``max_bytes``)."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(max(0, int(max_bytes)))
    except OSError:
        return {}
    split = _split_header(chunk)
    return parse_sxm_header(split[0]) if split else {}


def read_sxm(path: str | Path) -> dict:
    """``{"header": {...}, "channels": {name: {"forward": 2-D, "backward": 2-D}}}`` as stored."""
    content = Path(path).read_bytes()
    split = _split_header(content)
    if split is None:
        raise ValueError(f"{path}: no header end marker")
    raw, data = split
    header = parse_sxm_header(raw)
    try:
        nx, ny = (int(v) for v in header.get("scan_pixels", [0, 0])[:2])
    except (TypeError, ValueError):
        return {"header": header, "channels": {}}
    if nx <= 0 or ny <= 0:
        return {"header": header, "channels": {}}
    n = nx * ny * 4
    offset = 0
    channels: dict[str, dict] = {}
    names = header.get("channel_names", [])
    dirs = header.get("channel_directions", [])
    for i, name in enumerate(names):
        flag = dirs[i] if i < len(dirs) else "both"
        if "fwd" in flag and "bwd" not in flag:
            order = ["forward"]
        elif "bwd" in flag and "fwd" not in flag:
            order = ["backward"]
        else:
            order = ["forward", "backward"]
        ch: dict[str, np.ndarray] = {}
        for d in order:
            if offset + n > len(data):
                break
            ch[d] = np.frombuffer(data[offset:offset + n], dtype=">f4").reshape(ny, nx).astype(np.float64)
            offset += n
        channels[name] = ch
    return {"header": header, "channels": channels}


def _first_float(value) -> float | None:
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def oriented_frames(scan: dict, channel: str = "Z") -> dict:
    """One channel in image orientation (row 0 = top, column 0 = left), with scale.

    Keys: ``forward`` / ``backward`` (ndarray or None), ``nm_per_px``, ``width_nm``,
    ``height_nm``, ``bias_v``, ``setpoint_a``, ``scan_dir``, ``rec_time``, ``channel``."""
    header = (scan or {}).get("header") or {}
    chans = (scan or {}).get("channels") or {}
    ch = chans.get(channel)
    if ch is None:
        low = {str(k).lower(): k for k in chans}
        key = low.get(str(channel).lower())
        ch = chans.get(key) if key else None
    fwd = bwd = None
    scan_dir = str(header.get("scan_dir", "")).strip().lower()
    if ch:
        fwd, bwd = ch.get("forward"), ch.get("backward")
        if bwd is not None:
            bwd = np.fliplr(np.asarray(bwd))           # acquired along -x, stored mirrored
        if fwd is None and bwd is not None:
            fwd, bwd = bwd, None
        if scan_dir == "up":                          # first acquired line is the bottom edge
            fwd = None if fwd is None else np.flipud(np.asarray(fwd))
            bwd = None if bwd is None else np.flipud(np.asarray(bwd))
    parts = str(header.get("scan_range", "")).split()
    w_m = _first_float(parts[0]) if parts else None
    h_m = _first_float(parts[1]) if len(parts) > 1 else w_m
    nm_per_px = None
    if w_m and fwd is not None and np.asarray(fwd).ndim == 2 and np.asarray(fwd).shape[1] > 0:
        nm_per_px = w_m * 1e9 / np.asarray(fwd).shape[1]
    rec = f"{str(header.get('rec_date', '')).strip()} {str(header.get('rec_time', '')).strip()}".strip()
    return {"channel": channel, "forward": fwd, "backward": bwd, "nm_per_px": nm_per_px,
            "width_nm": w_m * 1e9 if w_m else None, "height_nm": h_m * 1e9 if h_m else None,
            "bias_v": _first_float(header.get("bias")), "setpoint_a": _first_float(header.get("z-controller>setpoint")),
            "scan_dir": scan_dir or None, "rec_time": rec or None}


def read_dat(path: str | Path) -> dict:
    """``{"header": {key: value}, "columns": {name: 1-D array}}`` of a ``.dat`` spectrum."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header: dict[str, str] = {}
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "[DATA]":
            start = i + 1
            break
        if not line.strip():
            continue
        parts = line.split("\t")
        key = parts[0].strip()
        if key:
            header[key] = "\t".join(parts[1:]).strip() if len(parts) > 1 else ""
    if start is None or start >= len(lines):
        return {"header": header, "columns": {}}
    names = [c.strip() for c in lines[start].split("\t") if c.strip()]
    rows: list[list[float]] = []
    for line in lines[start + 1:]:
        s = line.strip()
        if not s:
            continue
        try:
            rows.append([float(x) for x in s.split("\t")])
        except ValueError:
            continue
    if not rows:
        return {"header": header, "columns": {}}
    width = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    arr = np.array([(r + [np.nan] * width)[:width] for r in rows], dtype=np.float64)
    cols = {name: arr[:, j] for j, name in enumerate(names) if j < width}
    for j in range(len(names), width):
        cols[f"column_{j}"] = arr[:, j]
    return {"header": header, "columns": cols}
