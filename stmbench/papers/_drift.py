"""Measuring the thermal drift and telling the instrument to cancel it.

Every one of these papers takes hours of instrument time, and the rig walks about a nanometre
a minute. Over a full budget the sample moves 90–180 nm under the tip — a quantum corral is
15 nm across and an adatom is 0.7. Nothing here works unless the drift is tracked, which is
why the scenarios do not turn it down.

The instrument's own answer is ``Piezo.DriftCompSet``: a ramp on the piezo that follows the
sample. It takes the velocity at which features are seen to move in the scan frame — exactly
what a drift measurement reports. Reversing the sign doubles the drift, so the sequence ends
by measuring again.

Two ways to measure it are offered here because the right one depends on what is in the frame:

``by_frame_correlation``
    Cross-correlate successive frames of the same window. Needs 2-D structure — a
    reconstruction, a lattice, scattered adsorbates. This is the general method.

``by_step_edge``
    Track one step edge. A straight feature constrains nothing along its own length: the
    correlation of two frames of a bare step gives a ridge, not a peak, and the shift comes
    back as whatever the noise picked. Tracking the edge measures the one component that is
    observable — its normal — which is also the only one a step-referenced measurement needs.

**What makes a usable reference.** Both of the above are the same lesson from opposite ends: a
drift measurement can only recover the components its feature constrains. A straight edge gives
one. A periodic pattern gives one too, and only modulo its period — a 40 nm window of Au(111)
herringbone is a 6.3 nm grating, and correlating two of them recovers the across-stripe
component wrapped into one period and nothing along the stripes. Measured on P1 seed 0: the
sample moved (−0.78, −1.09) nm between two frames and the correlation reported (−0.13, +0.75).
Only a non-periodic two-dimensional feature — an adsorbate, a defect, a corner where two steps
meet — pins both axes.

The sequence therefore ends with *measure again*, and if the
residual is not much smaller than what went in, the compensation is switched back off: one
built on an unsupported reading points the piezo outside the measured displacement and can be
less reliable than leaving the sample to drift predictably.

Compensation cancels the steady part. It does not cancel the creep that follows every move, so
a long run still re-registers on its own feature between steps.

**A drift reading has to be credible before it is worth acting on.** Correlating two frames of
a patch with nothing in it returns approximately zero, which is indistinguishable from a stable
machine — and acting on it programs a compensation of nothing while the sample keeps walking.
The measurement is taken where there is something to register on, and the skill's own
consistency verdict is carried through: ``MeasureFrameDrift`` says how many pairs it actually
measured and warns when they disagree with one another. When it warns, nothing is set.

**Register on the thing you are measuring.** Where the window goes is not a free choice: it
should sit on the feature the experiment is about. P3 puts it on the corral — forty Fe atoms in
a ring, aperiodic and two-dimensional, and the ring is what has to still be there when the
spectrum is taken. P4 and P5 put it on the adatom they are about to move or push on. Those
scenarios have a nearly clean surface (3 defects per square micron) and it does not matter,
because they never register on the bare surface.

P1 is the one paper where this fails, and the reason is instructive: its subject *is* the
periodic thing. A window of herringbone and nothing else pins one component modulo 6.3 nm and
nothing at all along the stripes, so P1 has to widen the window until it holds a defect.

**Cheap frames, but not so cheap they hold nothing.** These frames are for registration, not
for the measurement the paper is about: they want enough structure to correlate and enough time
between them for the drift to show, not resolution. Spending measurement-grade frames on this
is how the drift work eats the scenario's budget — four P1 seeds once verified every claim and
still failed, over budget. But the two requirements pull against each other: a 40 nm window on
Au(111) is cheap and contains nothing but the grating, so it cannot measure drift at all. The
caller picks the size, and the right size is the smallest one that still holds an aperiodic
feature at that surface's defect density.
"""
from __future__ import annotations

import math


def null_the_drift(b, centre_m: tuple[float, float], *, frame_nm: float, pixels: int,
                   method: str = "correlation", n_frames: int = 3) -> dict:
    """Measure the drift, program the compensation, and check that it shrank.

    Returns what happened, for the ledger: the rate going in, the rate left afterwards, and
    whether the sign had to be flipped.
    """
    measure = _BY_STEP_EDGE if method == "step_edge" else _BY_CORRELATION
    out: dict = {"method": method, "applied": False}

    first, why = measure(b, centre_m, frame_nm, pixels, n_frames)
    if first is None:
        return {**out, "note": why or "could not measure a drift rate"}
    out["measured_nm_per_min"] = _nm_per_min(first)
    if why:
        # measured, but the measurement does not hang together — setting a compensation from
        # it would act on noise, so the compensation is left disabled
        return {**out, "note": why, "trusted": False}

    _set(b, first)
    after, why_after = measure(b, centre_m, frame_nm, pixels, n_frames)
    out["applied"] = True
    if after is None:
        # It was set and cannot be checked. Same call as a check that came back unchanged: an
        # unverified ramp on the piezo is unsupported by a corresponding sample measurement.
        b.run("SetDriftCompensation", {"enable": False, "vx": 0.0, "vy": 0.0, "vz": 0.0},
              optional=True)
        return {**out, "applied": False,
                "note": f"set, but the check could not measure ({why_after}) — switched off"}
    out["residual_nm_per_min"] = _nm_per_min(after)

    if _speed(after) > _speed(first):
        # it grew: the sign was the other way round. This is the failure the check is for.
        flipped = (-first[0], -first[1])
        _set(b, flipped)
        out["sign_flipped"] = True
        third, _ = measure(b, centre_m, frame_nm, pixels, n_frames)
        after = third if third is not None else after
        if third is not None:
            out["residual_nm_per_min"] = _nm_per_min(third)

    # It has to have actually helped. A compensation built on a reading the feature could not
    # support points the piezo outside the measured displacement — less reliable than a sample
    # that drifts predictably. Neither sign helping is the signature of that condition.
    if _speed(after) > KEEP_IF_BELOW * _speed(first):
        b.run("SetDriftCompensation", {"enable": False, "vx": 0.0, "vy": 0.0, "vz": 0.0},
              optional=True)
        out["applied"] = False
        out["note"] = ("compensation did not reduce the drift and was switched off — "
                       "this window does not constrain both axes")
    return out


#: the residual has to fall to this fraction of what went in for the compensation to be kept
KEEP_IF_BELOW = 0.5


def _set(b, v) -> None:
    b.run("SetDriftCompensation",
          {"enable": True, "vx": float(v[0]), "vy": float(v[1]), "vz": 0.0}, optional=True)


def _speed(v) -> float:
    return math.hypot(v[0], v[1])


def _nm_per_min(v) -> list[float]:
    return [round(float(x) * 6e10, 3) for x in v[:2]]


def _frames_at(b, centre_m, frame_nm: float, pixels: int, n: int) -> list[str]:
    out = []
    for _ in range(n):
        b.run("ScanAt", {"center_x_m": centre_m[0], "center_y_m": centre_m[1],
                         "size_m": frame_nm * 1e-9, "pixels": pixels}, optional=True)
        b.run("SaveScan", {}, optional=True)
        frame = b.latest_frame()
        if frame is not None:
            out.append(str(frame))
    return out


def _BY_CORRELATION(b, centre_m, frame_nm: float, pixels: int, n_frames: int):
    """(velocity, complaint) from successive frames of one window."""
    paths = _frames_at(b, centre_m, frame_nm, pixels, n_frames)
    if len(paths) < 2:
        return None, "fewer than two frames to compare"
    # newline-separated: the skill splits on commas, so a JSON array is torn at its own
    # separators and every path comes back "file not found"
    res = b.run("MeasureFrameDrift", {"scan_paths": chr(10).join(paths)}, optional=True)
    if b.data(res, "verdict") != "measured":
        return None, "nothing in these frames registered — no feature to track"
    secs = b.data(res, "frame_seconds") or acq_seconds(paths[-1])
    # the skill's feature_* fields are how far the features moved in the scan frame (x right,
    # y up) — the velocity Piezo.DriftCompSet wants. Its dx/dy are the register-shift in array
    # axes (x reversed, rows downward); the fallback keeps the older skill working, and the
    # measure-again check below still catches a sign that came out wrong.
    dx, dy = b.data(res, "feature_dx_median_nm"), b.data(res, "feature_dy_median_nm")
    if dx is None or dy is None:
        dx, dy = b.data(res, "dx_median_nm"), b.data(res, "dy_median_nm")
        if dx is not None:
            dx = -float(dx)
    if not secs or dx is None or dy is None:
        return None, "no frame duration to turn a displacement into a rate"
    v = (float(dx) * 1e-9 / float(secs), float(dy) * 1e-9 / float(secs))
    if b.data(res, "consistency_warning"):
        return v, "the pairs disagree with one another — this median is not a drift"
    if int(b.data(res, "n_measured") or 0) < 2:
        return v, "only one pair of frames registered"
    return v, ""


def _BY_STEP_EDGE(b, centre_m, frame_nm: float, pixels: int, n_frames: int):
    """(velocity, complaint) from where one step edge sits in two frames."""
    obs = []
    for path in _frames_at(b, centre_m, frame_nm, pixels, max(2, n_frames - 1)):
        res = b.run("LocateStepEdge", {"scan_path": path, "channel": "Z"}, optional=True)
        if b.data(res, "verdict") != "step_edge":
            continue
        ex, ey = b.data(res, "edge_x_m"), b.data(res, "edge_y_m")
        ang = b.data(res, "edge_angle_scan_deg")
        if ex is None or ey is None or ang is None:
            continue
        obs.append((float(ex), float(ey), float(ang), acq_seconds(path)))
    if len(obs) < 2:
        return None, "the step edge was not found in two frames running"
    (x0, y0, a0, _), (x1, y1, _a1, secs) = obs[0], obs[-1]
    if not secs:
        return None, "no frame duration to turn a displacement into a rate"
    span = secs * (len(obs) - 1)
    nx, ny = -math.sin(math.radians(a0)), math.cos(math.radians(a0))
    along_normal = ((x1 - x0) * nx + (y1 - y0) * ny) / span
    return (along_normal * nx, along_normal * ny), ""


def acq_seconds(path) -> float | None:
    """How long a frame took, from its own header — the instrument's clock, not ours."""
    try:
        from mast.io.nanonis_files import read_sxm

        header = read_sxm(str(path)).get("header") or {}
    except Exception:  # noqa: BLE001
        return None
    for key in ("acq_time", "scan_time_s", "acquisition_time"):
        try:
            v = float(str(header.get(key)).split()[0])
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        if v > 0:
            return v
    return None
