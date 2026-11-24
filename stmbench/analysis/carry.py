"""Did a 1.0 nm drag take the atom to the end point? (P7's outcome, read from what the instrument shows.)

P7 counts a drag as a **full carry** when the atom rests within 0.6 site spacings (0.153 nm on
Cu(111)) of where the holding apex stopped. That is a finer outcome than :mod:`stmbench.analysis.trace`
reads: ``read_drag`` tells whether an atom hopped at all and was built for drags of several nm. On a
drag of 1.0 nm at 0.1 nm/s (21 samples, about four hops for an atom that follows) an atom dropped
at the second, third or last site leaves a fall in Z of a few picometres to a few tens of them, which
the noise, the lattice ripple and a tip that changes in the middle can all imitate.

**What the reader does.** The decisive observation is a small slow frame after the drag: where the
atom is, against where the tip stopped. The frame images through the same apex as the frame that
placed the tip over the atom, so an atom that the tip carried sits where the tip stopped, to within
the frame's own placement error, and one dropped at the last site sits 0.25 nm short of it. The trace
is a cross-check and a fallback: Z at the end of the path against Z at its start (a carried atom is
under the tip at both ends; one left behind makes the trace fall), used where no frame was taken or
the atom was not found.

**How well it reads** is in ``tests/test_analysis_carry.py`` (a small version) and in the P7 baseline's
report, against the judge's record of each drag (``truth()["drags"]["records"][k]["followed"]``,
which only the checks read, never the reader).
"""
from __future__ import annotations

import math

import numpy as np

from .adatoms import locate

#: the distance inside which P7 counts an atom as carried to the end point: 0.6 of the site spacing
#: (stmsim.physics.drag_threshold.WITH_TIP_FRACTION × Cu(111)'s 0.255 nm), as nm
CARRY_RADIUS_NM = 0.153
#: how far from the drag's path (nm) a bump in the after-frame is still taken for the atom: an atom
#: that stayed or was dropped lies on the path, and any other atom is 3 nm away
PATH_BAND_NM = 1.3
#: what is added to the radius for the frame's reading. A carried atom sits 0.10 nm from the end point
#: at the median (behind the apex: it hops when the apex is 0.6 spacings ahead), up to the radius by
#: definition, and a frame places it to about 0.06 nm per axis; one dropped at the last site is 0.16-0.3
#: nm short. Against the judge's record (2026-11-17, seeds 0-39, 60 drags each, ~2200 drags each time):
#: with 5 nm / 50 px frames the radius alone read 96 of 1202 carries as drops and 1 of 1043 misses as
#: carried, 0.18 gave about 19 and 18, 0.21 gave 7 and 35; with the baseline's 3.2 nm / 32 px frames
#: (0.1 nm per pixel) 0.153 gave 83 of 1204 and 4 of 1014, 0.165 gave 31 and 12, 0.17 gave 21 and 15,
#: 0.18 gave 9 and 23 (false drops, false carries). The two populations touch and 0.168 balances them.
FRAME_MARGIN_NM = 0.015
#: trace fallback: Z over the last samples against the first (pm). A carried atom is under the tip at
#: both ends, so Z rises or holds (+8 pm median, the lattice ripple and the tip's drift); an atom left
#: behind makes it fall (median -30 pm). Nothing here is exact: see the checks.
TRACE_DROP_PM = -8.0
TRACE_EDGE_SAMPLES = 3


def trace_fall_pm(spectrum: dict) -> float | None:
    """Z over the last samples of a move's trace minus Z over the first (pm), or ``None`` when the
    record has no usable Z (no feedback, too short)."""
    z = spectrum.get("z_abs")
    if z is None or not spectrum.get("feedback", True):
        return None
    z = np.asarray(z, float) * 1e12
    if z.size < 2 * TRACE_EDGE_SAMPLES or not np.all(np.isfinite(z)):
        return None
    return float(z[-TRACE_EDGE_SAMPLES:].mean() - z[:TRACE_EDGE_SAMPLES].mean())


def find_atom(frame: dict, start, end, *, band_nm: float = PATH_BAND_NM) -> tuple[float, float] | None:
    """The atom of an after-frame: the bump within ``band_nm`` of the segment ``start``-``end``
    (scan coordinates, nm) nearest its end, ``None`` when there is none (the atom is on the tip, or
    the frame does not show it)."""
    (ax, ay), (bx, by) = start, end
    ux, uy = bx - ax, by - ay
    uu = ux * ux + uy * uy or 1e-12
    best, best_d = None, None
    for x, y, _ in locate(frame):
        t = min(max(((x - ax) * ux + (y - ay) * uy) / uu, 0.0), 1.0)
        if math.hypot(x - (ax + t * ux), y - (ay + t * uy)) > band_nm:
            continue
        d = math.hypot(x - bx, y - by)
        if best_d is None or d < best_d:
            best, best_d = (float(x), float(y)), d
    return best


def read_carry(spectrum: dict | None, *, end, atom=None, framed: bool = False) -> dict:
    """What a drag did: ``{"carried": bool | None, "how": "frame" | "trace" | None, "end_offset_nm":
    float | None, "fall_pm": float | None}``.

    ``end``: where the tip stopped (scan coordinates, nm); ``atom``: where the after-frame put the
    atom (:func:`find_atom`), or ``None`` for no frame or no atom found. With an atom the verdict is
    its distance from ``end`` against :data:`CARRY_RADIUS_NM`; without one it is the trace's fall,
    and ``None`` (unreadable) when that is missing too. ``framed``: a frame was taken and showed no
    atom on the path, so the atom is not at the end (it is on the tip): not carried, whatever the
    trace says (a lifted atom steps Z down by its height, and a trace that ends high can follow it)."""
    fall = None if spectrum is None else trace_fall_pm(spectrum)
    out = {"carried": None, "how": None, "end_offset_nm": None, "fall_pm": fall}
    if atom is None and framed:
        out.update(carried=False, how="frame")
    elif atom is not None:
        off = math.hypot(atom[0] - end[0], atom[1] - end[1])
        out.update(carried=bool(off <= CARRY_RADIUS_NM + FRAME_MARGIN_NM), how="frame", end_offset_nm=float(off))
    elif fall is not None:
        out.update(carried=bool(fall >= TRACE_DROP_PM), how="trace")
    return out
