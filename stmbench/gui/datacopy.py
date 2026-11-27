"""Plain-text copies of the data the window lists, for an account that reads files (no Qt).

Why. An analysis such as a dispersion fit or an inversion of Delta-f(z) curves needs the measured arrays, which the
window only shows as pictures and a handful of pinned readings. With ``--data-dir DIR`` the console writes what its Data
tab lists as CSV with ``#`` header lines: one file per spectrum or move trace, one per channel and direction of a frame.
Without the option nothing is written.

What it reads. Only :class:`~stmbench.gui.session.GuiSession` (``data``, ``frame``, ``spectrum``): exactly what the window
can show, and never a raw file or an instrument internal. What it writes. Only the whitelisted header keys below, and
the values in the units the window uses; the drift offsets, the tip's state and everything the judge holds stay where they
are. The files are written by a thread of their own, each to a temporary name and then renamed, so that a reader never
meets half a file; a frame is written once its scan reply has arrived (a frame is rendered whole but revealed row by row
on screen, and the copy never holds rows the screen has not reached).
"""
from __future__ import annotations

import os
import queue
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import fmt

#: what a file's header may say, by kind of file: nothing else is written there
FRAME_KEYS = ("source", "kind", "channel", "direction", "slow_axis", "unit", "centre_x_nm", "centre_y_nm", "size_nm", "pixels",
              "angle_deg", "line_time_s", "start_time_s", "start_time", "bias_v", "setpoint_pa", "values", "pixel_mapping",
              "rows", "columns")
SPECTRUM_KEYS = ("source", "kind", "tip_x_nm", "tip_y_nm", "time_s", "time", "sweep", "v_start_v", "v_end_v", "sweep_nm",
                 "bias_v", "setpoint_pa", "points")
TRACE_KEYS = ("source", "kind", "from_x_nm", "from_y_nm", "to_x_nm", "to_y_nm", "speed_nm_s", "bias_v", "setpoint_pa",
              "feedback", "time_s", "time", "points")
HEADER_KEYS = frozenset(FRAME_KEYS) | frozenset(SPECTRUM_KEYS) | frozenset(TRACE_KEYS)

#: the columns of a spectrum or a move trace, with the unit the window plots them in (the factor from the stored unit):
#: the same table as the viewer's (tests/test_gui_datacopy.py compares them)
COLUMNS = {
    "v": ("V", 1.0, "V"), "i": ("I", 1e12, "pA"), "didv": ("dI/dV", 1e9, "nS"), "z_rel": ("z rel", 1e12, "pm"),
    "z_abs": ("z", 1e9, "nm"), "df": ("Δf", 1.0, "Hz"), "amp": ("amplitude", 1e12, "pm"),
    "dist": ("distance", 1e9, "nm"), "t": ("time", 1.0, "s"),
}
#: a frame channel's unit in the window and the factor from its stored SI unit (z in m, current and lock-in X in A)
FRAME_UNITS = {"z": ("pm", 1e12), "current": ("pA", 1e12), "didv": ("pA", 1e12)}

#: how a pixel maps to scan coordinates, as :func:`stmbench.render.pixel_to_scan_nm` has it (tests/test_gui_datacopy.py)
PIXEL_MAPPING = ("row 0 is the top (the high-v edge) and column 0 the left; pixel centres at half-integers; "
                 "u = -size/2 + (col+0.5)*size/pixels, v = h/2 - (row+0.5)*h/rows with h = size*rows/pixels; "
                 "x_nm = centre_x + cos(a)*u + sin(a)*v, y_nm = centre_y - sin(a)*u + cos(a)*v, a = angle_deg in radians "
                 "(clockwise positive)")
#: a name the window shows for a file: nothing that could leave the directory
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,100}")
_DIRECTIONS = ("forward", "backward")


def _num(v: Any) -> str:
    return f"{float(v):.10g}"


def _header(lines: list[tuple[str, Any]], allowed: tuple[str, ...]) -> str:
    out = []
    for key, value in lines:
        if key not in allowed:
            raise KeyError(f"header key {key!r} is not allowed here")
        out.append(f"# {key}: {value}")
    return "\n".join(out) + "\n"


def base_name(name: str) -> str:
    """The file name the Data tab shows, without its extension (what a reader cites is ``name``)."""
    base = Path(str(name)).name.rsplit(".", 1)[0]
    if not _SAFE_NAME.fullmatch(base):
        raise ValueError("not a name the window shows")
    return base


def spectrum_text(spec: dict, junction: dict | None = None) -> str:
    """One spectrum or move trace as CSV text: ``#`` header lines, the column names with their units, the rows."""
    kind = spec.get("kind")
    cols = [k for k in COLUMNS if k in spec and np.ndim(spec[k]) == 1]
    n = len(spec[cols[0]]) if cols else 0
    if kind == "move":
        lines: list[tuple[str, Any]] = [("source", spec["name"]), ("kind", "move")]
        for key, field in (("from", "from_nm"), ("to", "to_nm")):
            if field in spec:
                lines += [(f"{key}_x_nm", _num(spec[field][0])), (f"{key}_y_nm", _num(spec[field][1]))]
        for key, field in (("speed_nm_s", "speed_nm_s"), ("bias_v", "bias_v"), ("setpoint_pa", "setpoint_pa")):
            if field in spec:
                lines.append((key, _num(spec[field])))
        if "feedback" in spec:
            lines.append(("feedback", "on" if spec["feedback"] else "off"))
        allowed = TRACE_KEYS
    else:
        x, y = spec["tip_nm"]
        lines = [("source", spec["name"]), ("kind", kind), ("tip_x_nm", _num(x)), ("tip_y_nm", _num(y))]
        j = junction or {}
        if fmt.sweep_text(j):
            lines.append(("sweep", fmt.sweep_text(j)))
        for key, field in (("v_start_v", "v_start"), ("v_end_v", "v_end"), ("sweep_nm", "sweep_nm"), ("bias_v", "bias_v"),
                           ("setpoint_pa", "setpoint_pa")):
            if field in j:
                lines.append((key, _num(j[field])))
        allowed = SPECTRUM_KEYS
    lines += [("time_s", _num(spec["t_s"])), ("time", fmt.hms(spec["t_s"])), ("points", n)]
    names = ",".join(f"{COLUMNS[k][0]} ({COLUMNS[k][2]})" if COLUMNS[k][0] != COLUMNS[k][2] else COLUMNS[k][0] for k in cols)
    data = np.column_stack([np.asarray(spec[k], float) * COLUMNS[k][1] for k in cols]) if cols else np.zeros((0, 0))
    rows = "\n".join(",".join(_num(v) for v in row) for row in data)
    return _header(lines, allowed) + names + "\n" + rows + ("\n" if rows else "")


def frame_text(frame: dict, slow_axis: str | None = None) -> str:
    """One channel and direction of a frame as CSV text: header, then the matrix (row 0 on top), in the window's units."""
    unit, factor = FRAME_UNITS[frame["channel"]]
    arr = np.asarray(frame["array"], float) * factor
    ny, nx = arr.shape
    cx, cy = frame["center_nm"]
    lines: list[tuple[str, Any]] = [
        ("source", frame["frame"]), ("kind", "frame"), ("channel", frame["channel"]), ("direction", frame["direction"])]
    if slow_axis:
        lines.append(("slow_axis", slow_axis))
    lines += [("unit", unit), ("centre_x_nm", _num(cx)), ("centre_y_nm", _num(cy)), ("size_nm", _num(frame["size_nm"])),
              ("pixels", nx), ("angle_deg", _num(frame.get("angle_deg") or 0.0)), ("line_time_s", _num(frame["line_time_s"])),
              ("start_time_s", _num(frame["t_start_s"])), ("start_time", fmt.hms(frame["t_start_s"]))]
    for key in ("bias_v", "setpoint_pa"):
        if key in frame:
            lines.append((key, _num(frame[key])))
    lines += [("values", "as acquired (no flattening), in " + unit),
              ("pixel_mapping", PIXEL_MAPPING), ("rows", ny), ("columns", nx)]
    body = "\n".join(",".join(_num(v) for v in row) for row in arr)
    return _header(lines, FRAME_KEYS) + body + "\n"


# ── writing ──
def write_atomically(directory: Path, name: str, text: str) -> bool:
    """``directory/name`` with ``text``, 0644, through a temporary name and a rename. Never follows a symlink (the directory
    itself, the temporary file or the target); an existing file of that name is left as it is (saved data does not change).
    False when nothing was written."""
    if directory.is_symlink():
        raise OSError("the data directory is a symlink")
    directory.mkdir(mode=0o755, parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(directory, 0o755)
    target = directory / name
    if target.is_symlink():
        raise OSError(f"{name} is a symlink")
    if target.exists():
        return False
    tmp = directory / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        if os.name == "posix":
            os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True


class DataCopier:
    """Writes the copies of what an action saved, on a thread of its own. ``save`` is called from the window with the reply of a
    successful action; the files appear a moment later (:meth:`wait`)."""

    def __init__(self, session, directory: str | Path):
        self._session = session
        self.directory = Path(directory)
        self.errors: list[str] = []
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="datacopy", daemon=True)
        self._thread.start()

    def save(self, reply: Any, junctions: dict[str, dict] | None = None) -> None:
        """Queue the copies of the frame, spectrum or move trace that ``reply`` names (nothing for any other reply)."""
        if not isinstance(reply, dict):
            return
        if reply.get("frame"):
            self._queue.put(("frame", str(reply["frame"]), None))
        for key in ("spectrum", "trace"):
            if reply.get(key):
                name = str(reply[key])
                self._queue.put(("spectrum", name, dict((junctions or {}).get(name) or {})))

    def wait(self, timeout: float = 30.0) -> bool:
        """Until everything queued so far is written (for tests and for a clean exit)."""
        done = threading.Event()
        self._queue.put(("done", "", done))
        return done.wait(timeout)

    def close(self) -> None:
        self.wait()
        self._queue.put(("stop", "", None))
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            kind, name, extra = self._queue.get()
            if kind == "stop":
                return
            if kind == "done":
                extra.set()
                continue
            try:
                if kind == "frame":
                    self._write_frame(name)
                else:
                    self._write_spectrum(name, extra)
            except Exception as exc:  # noqa: BLE001 - a copy that cannot be written is told in ``errors``; the window goes on
                self.errors.append(f"{name}: {type(exc).__name__}: {exc}")

    def _write_spectrum(self, name: str, junction: dict) -> None:
        spec = self._session.spectrum(name)
        write_atomically(self.directory, base_name(spec["name"]) + ".csv", spectrum_text(spec, junction))

    def _write_frame(self, name: str) -> None:
        listed = next((f for f in self._session.data()["frames"] if f["frame"] == name), None)
        if listed is None:
            return
        base = base_name(name)
        for channel in listed.get("channels", ["z", "current"]):
            if channel not in FRAME_UNITS:
                continue
            for direction in _DIRECTIONS:
                frame = self._session.frame(name, channel, direction)
                write_atomically(self.directory, f"{base}_{channel}_{direction}.csv", frame_text(frame, listed.get("direction")))
