"""P1 — Barth et al. 1990: the Au(111) herringbone, measured.

The baseline route is a grid of high-resolution windows, with the reconstruction measured on
each, the step-free ones kept, and the period taken as their median. A second rotational domain
is found by walking to patches several hundred nanometres away until a window's stripes run
60° from the first.

Three things decide whether a right measurement is scored right, and all three are handled
here explicitly rather than by luck:

* **Frame vs stage.** ``AssessHerringbone`` reports ``stripe_angle_deg`` in image
  coordinates, where 0° is the fast scan axis and the angle grows the way the row index
  does. The reader puts row 0 at the high-y edge of the window, so the row axis runs along
  −y of the scan frame and the stage-frame angle is the negative of it, modulo 180°.
* **Steps.** A step edge is 200-odd pm and the reconstruction 10, so a window with an edge in
  it has its stripe peak dragged off by the step and reads several nanometres too long. Such
  a window also has many times the corrugation of a step-free one, which is how they are told
  apart here: everything above twice the flattest window seen is dropped before anything is
  averaged.
* **Domain walls and FFT bins.** A window straddling a wall carries two stripe directions at
  once, and a 120 nm window resolves the period only to about a third of a nanometre per FFT
  bin, so a single reading can snap to a bin. Both show up as outliers among otherwise
  agreeing windows, so the period reported is the median over the windows, never one frame.

The baseline may only use what the instrument shows it. It never reads ``world.truth()`` —
``tests/test_papers.py`` checks that by inspection, because a baseline that peeks is not a
solvability proof.
"""
from __future__ import annotations

from ._drift import null_the_drift
from ._runner import Baseline

#: windows whose stripe directions differ by at least this are different rotational domains.
#:
#: The three orientations are 60° apart, but a single domain shows TWO directions — the arms of
#: its own zigzag, 2·alpha apart, and alpha is drawn up to 18°. So anything under 36° can be one
#: domain measured twice, and a 30° threshold reported exactly that: two windows 31.5° apart,
#: both in the domain at 105.5°, whose arms sit at 89.4° and 121.6°. The judge caught it —
#: ``distinct_from`` compares the two claims' truths, not the two reported numbers.
DOMAIN_SEPARATION_DEG = 45.0
#: a window this wide holds ~19 stripe periods
WINDOW_NM = 120.0
#: 0.31 nm/px, inside the claim's 0.4 limit. 512 px measures no better (the stripe peak is
#: already resolved) and costs the simulator 2.4 s a frame against a 5 s wire reply budget.
WINDOW_PX = 384
GRID_STEP_NM = 160.0
#: registration frames for the drift measurement — small and fast, since they only have to
#: correlate. A third of the line count of a measurement frame, and at 1 nm/min the sample
#: still moves three or four pixels between two of them.
#: registration frames. Wide enough to contain a defect or two at this surface's density —
#: the herringbone alone is a grating and pins only one component, modulo its period.
DRIFT_NM, DRIFT_PX = 100.0, 256
#: reconnaissance frames: wide enough to hold six stripe periods, cheap enough that nine of
#: them cost a third of one measurement grid
SCOUT_NM, SCOUT_PX = 50.0, 160
#: a window carrying a step edge has many times the corrugation of a step-free one, so this
#: multiple of the flattest window seen is the line between "the reconstruction" and "a step"
RMS_FACTOR = 2.0
#: patches to walk until two domains are in hand; rotational domains are 200-500 nm across
PATCHES = ((0.0, 0.0), (600.0, 600.0), (-600.0, -600.0), (600.0, -600.0))
LINE_TIME_S = 0.3


def run_p1(host, scenario, *, seed: int = 0, extra: dict | None = None) -> dict:
    b = Baseline(host, scenario, seed)
    extra = dict(extra or {})
    step = float(extra.get("grid_step_nm", GRID_STEP_NM))

    # Stabilise on the first window that shows the reconstruction, then measure. Over this
    # scenario's two-hour budget the sample walks ~120 nm, and within a single frame it shears
    # the image — which stretches the very stripe period being measured. But the drift has to
    # be measured somewhere with something to register on: two frames of a bare patch correlate
    # to approximately zero, which reads exactly like a stable machine. The first measurement
    # window doubles as that anchor, so nothing is spent on a separate reconnaissance pass.
    anchor = None
    for dx, dy in [(gx * step, gy * step) for gx in (0, -1, 1) for gy in (0, -1, 1)]:
        got = _window(b, dx, dy, scenario.material)
        if got:
            anchor = got
            break
    if anchor is None:
        return b.finish(n_windows=0, note="no window showed the reconstruction")
    drift = null_the_drift(b, (anchor["cx_nm"] * 1e-9, anchor["cy_nm"] * 1e-9),
                           frame_nm=DRIFT_NM, pixels=DRIFT_PX)

    windows: list[dict] = []
    for n, (px, py) in enumerate(PATCHES):
        offsets = ([(gx * step, gy * step) for gx in (-1, 0, 1) for gy in (-1, 0, 1)] if n == 0
                   else [(gx * step, 0.0) for gx in (-1, 0, 1)])
        for dx, dy in offsets:
            got = _window(b, px + dx, py + dy, scenario.material)
            if got:
                windows.append(got)
        if len(_flat(windows)) >= 4 and _two_domains(_flat(windows)) is not None:
            break
    if not windows:
        windows = [anchor]
    flat = _flat(windows)

    # ── the period: the median over the flat windows, so one bin-snapped or step-contaminated
    # reading cannot carry it ──
    periods = sorted(w["period_nm"] for w in flat)
    b.report("stripe_period_nm", periods[len(periods) // 2], "nm")

    # ── the orientation, reported at the place it was measured ──
    best = max(flat, key=lambda w: w["snr"])
    b.report("stripe_orientation_deg", best["stripe_deg"], "deg",
             x_nm=best["cx_nm"], y_nm=best["cy_nm"])

    pair = _two_domains(flat)
    if pair is not None:
        a, c = pair
        b.report("domain_a_orientation_deg", a["stripe_deg"], "deg",
                 x_nm=a["cx_nm"], y_nm=a["cy_nm"])
        b.report("domain_b_orientation_deg", c["stripe_deg"], "deg",
                 x_nm=c["cx_nm"], y_nm=c["cy_nm"])
    return b.finish(n_windows=len(windows), n_flat=len(flat), drift=drift,
                    stripe_directions=[round(w["stripe_deg"], 1) for w in flat],
                    periods_nm=[round(x, 3) for x in periods],
                    found_two_domains=pair is not None)


def _flat(windows: list[dict]) -> list[dict]:
    """The windows without a step edge in them, told apart by corrugation alone."""
    if not windows:
        return []
    lo = min(w["rms_m"] for w in windows)
    return [w for w in windows if w["rms_m"] <= RMS_FACTOR * lo] or windows


def _window(b: Baseline, cx_nm: float, cy_nm: float, material: str, *,
            size_nm: float = WINDOW_NM, pixels: int = WINDOW_PX) -> dict | None:
    """One frame, measured. The defaults are measurement grade; the reconnaissance pass asks
    for something cheaper, because all it needs to know is where the reconstruction is."""
    b.run("ScanAt", {"center_x_m": cx_nm * 1e-9, "center_y_m": cy_nm * 1e-9,
                     "size_m": size_nm * 1e-9, "pixels": pixels,
                     "line_time_s": LINE_TIME_S}, optional=True)
    b.run("SaveScan", {}, optional=True)
    frame = b.latest_frame()
    return _assess(b, frame, material) if frame is not None else None


def _assess(b: Baseline, path, material: str) -> dict | None:
    """One frame through ``AssessHerringbone``, in stage-frame terms."""
    geom = _frame_geometry(path)
    if geom is None:
        return None
    res = b.run("AssessHerringbone", {"scan_path": str(path), "channel": "Z",
                                      "substrate": material}, optional=True)
    data = res.data if (res is not None and isinstance(res.data, dict)) else {}
    stripe, period = data.get("stripe_angle_deg"), data.get("period_nm")
    if stripe is None or period is None:
        return None
    return {"path": str(path), "period_nm": float(period),
            # image angle → stage angle: the row axis runs along −y (see the module docstring)
            "stripe_deg": (-float(stripe)) % 180.0,
            "rms_m": float(data.get("corrugation_rms_m") or 1e9),
            "snr": float(data.get("snr") or 0.0),
            "verdict": data.get("verdict"),
            "cx_nm": geom[0], "cy_nm": geom[1]}


def _two_domains(tiles: list[dict]) -> tuple[dict, dict] | None:
    """The pair of windows whose stripe directions are furthest apart, if far enough."""
    best, best_sep = None, 0.0
    for i, a in enumerate(tiles):
        for c in tiles[i + 1:]:
            d = abs(a["stripe_deg"] - c["stripe_deg"]) % 180.0
            sep = min(d, 180.0 - d)
            if sep > best_sep:
                best, best_sep = (a, c), sep
    return best if best_sep >= DOMAIN_SEPARATION_DEG else None


def _frame_geometry(path) -> tuple[float, float] | None:
    """Scan centre (nm) from the .sxm header — the frame says where it was taken."""
    try:
        from mast.io.nanonis_files import read_sxm

        header = read_sxm(str(path)).get("header", {})
    except Exception:  # noqa: BLE001
        return None
    off = header.get("scan_offset") or header.get("SCAN_OFFSET")
    if isinstance(off, str):
        off = off.split()
    try:
        return float(off[0]) * 1e9, float(off[1]) * 1e9
    except (TypeError, ValueError, IndexError):
        return None
