"""P2 — Crommie/Hasegawa 1993: the surface-state dispersion from standing waves.

Survey, locate the step edge, take a line of dI/dV spectra running away from it along its
normal, and fit the dispersion. ``SpectroscopyAtPositions`` does the acquisition: it writes
one ``.dat`` per point with a per-point basename and checks that each file really is that
point's, which is what makes the set safe to hand to a fit.

A step edge, not an adsorbate, for two reasons. It is what the 1993 papers used, and the
physics behind a straight reflector is the J0 the fit assumes. Measured on the generator, the
step route recovers E0 to about 6 meV and m* to about 0.7 %, while the point-scatterer route
biases m* by twenty percent: the LDOS round a point defect goes as J0 squared with a Y0 cross
term, which is not the ``cos(2kd)/kd`` a point fit assumes.

The edge comes from ``LocateStepEdge``. Its angle carries a few degrees of bias on a real
frame, because creep shears successive rows and rotates the apparent edge — but the spectra
are placed along the *measured* normal and the distances are taken to the *measured* line, so
the two agree and only the cosine of that angle survives, half a percent on the distance scale.

**Drift is the thing that decides whether this works.** Forty spectra take the better part of
an hour, and at the rig's measured ~1 nm/min the sample slides under the tip by more than the
1.5 nm the ripple repeats over. A distance axis built from one edge located once is scrambled
by the end. Two procedures address it, in the following order:

1. **Null it at the instrument.** Track the step edge across two frames for the rate,
   ``SetDriftCompensation`` to ramp the piezo along with the sample, then measure again to
   check it actually shrank — and flip the sign if it grew, which is the failure the
   verification step exists to catch. Measured on seed 0: 0.57 / 0.62 nm/min going in,
   0.07 / 0.07 nm/min left afterwards.
2. **Track what is left.** Compensation cancels the steady part, not the creep that follows
   every move, so the line is still taken in batches with the edge re-located before each one,
   and each spectrum's distance is measured against the edge as it was at that moment.

Neither step alone is enough, and turning the drift down instead of tracking it would remove
the part of this experiment that is actually hard.
"""
from __future__ import annotations

import json
import math

from ._drift import null_the_drift
from ._runner import Baseline

#: the half-wavelength at the Fermi level is about 1.5 nm, so the spectra have to sit closer
#: together than 0.75 nm or the ripple is aliased away. Forty points over 1–13 nm gives three
#: per half-wavelength across four hundred millivolts of band.
N_NEAR, N_FAR = 40, 5
D_MIN_NM, D_NEAR_NM, D_FAR_NM = 1.0, 13.0, 24.0
#: spectra per batch. Between batches the edge is re-located, so the distance axis follows the
#: drift instead of being fixed once at the start.
BATCH = 9
V_LO, V_HI = -0.6, 0.4
E_MAX_FIT_V = 0.15
FRAME_NM, FRAME_PX = 60.0, 256
#: the drift measurement only has to see where the edge is, and a 200 pm step is unmissable at
#: half a nanometre per pixel — no reason to spend measurement-grade frames on registration
DRIFT_PX = 128


def run_p2(host, scenario, *, seed: int = 0, extra: dict | None = None) -> dict:
    b = Baseline(host, scenario, seed)
    extra = dict(extra or {})
    frame_nm = float(extra.get("frame_nm", FRAME_NM))

    edge = _find_edge(b, frame_nm)
    if edge is None:
        return b.finish(note="no step edge found to measure standing waves against")

    # a straight step tells you nothing along its own length, so track the edge itself
    drift = null_the_drift(b, (edge[0], edge[1]), frame_nm=frame_nm, pixels=DRIFT_PX,
                           method="step_edge")

    wanted = [D_MIN_NM + (D_NEAR_NM - D_MIN_NM) * i / (N_NEAR - 1) for i in range(N_NEAR)]
    wanted += [D_NEAR_NM + (D_FAR_NM - D_NEAR_NM) * (i + 1) / N_FAR for i in range(N_FAR)]

    paths: list[str] = []
    dists: list[float] = []
    batches = 0
    for start in range(0, len(wanted), BATCH):
        chunk = wanted[start:start + BATCH]
        if start:                                    # re-locate the edge before every batch
            again = _find_edge(b, frame_nm, near=edge)
            if again is not None:
                edge = again
        ex, ey, angle_deg, _ = edge
        nx, ny = -math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))
        positions = [{"x_m": ex + nx * d * 1e-9, "y_m": ey + ny * d * 1e-9,
                      "label": f"d={d:.1f}nm"} for d in chunk]
        res = b.run("SpectroscopyAtPositions", {
            "positions": json.dumps(positions),
            "expected_coord_epoch": _coord_epoch(),
            "run_tag": f"sw{batches}",
            "assess": False,
            "position_tol_nm": 2.0,
        }, optional=True)
        got = b.data(res, "dat_paths") or [str(q) for q in b.dats(f"sw{batches}_p*.dat")]
        # the distance each spectrum was actually taken at, against the edge as it was then
        for path, d in zip(got, chunk):
            paths.append(str(path))
            dists.append(float(d))
        batches += 1

    if len(paths) < 8:
        return b.finish(n_spectra=len(paths), note="too few spectra to fit a dispersion")

    fit = b.run("FitDispersion", {
        "dat_paths": json.dumps(paths),
        "distances_nm": json.dumps(dists),
        "scatterer_kind": "step",
        "min_distance_nm": 1.0,
        # above about +0.15 V the ripple is under a nanometre across and the tip's own DOS
        # carries more of the signal than the surface state does; those energies add scatter
        # to the parabola without adding reach
        "energy_min_v": V_LO + 0.05, "energy_max_v": E_MAX_FIT_V,
    }, optional=True)
    e0 = b.data(fit, "e0_mev")
    m = b.data(fit, "m_eff")
    if e0 is not None:
        b.report("e0_mev", float(e0), "meV")
    if m is not None:
        b.report("m_eff", float(m), "m_e")
    return b.finish(n_spectra=len(paths), n_batches=batches, drift=drift,
                    edge_nm=[edge[0] * 1e9, edge[1] * 1e9], edge_angle_deg=edge[2],
                    step_height_pm=edge[3],
                    fit_verdict=b.data(fit, "verdict"), r2=b.data(fit, "r2"))


def _find_edge(b: Baseline, frame_nm: float,
               near: tuple | None = None) -> tuple[float, float, float, float] | None:
    """A point on the dominant step edge and the direction it runs, in scan coordinates.

    If the first window is one flat terrace there is nothing to scatter off, so the search
    walks outward a terrace or two at a time until an edge turns up. Once an edge is known,
    ``near`` re-images that same place, which is how the edge is tracked as the sample drifts."""
    if near is not None:
        centres = [(near[0] * 1e9, near[1] * 1e9)]
    else:
        centres = [(0.0, 0.0), (frame_nm, 0.0), (0.0, frame_nm), (-frame_nm, 0.0)]
    for dx, dy in centres:
        b.run("ScanAt", {"center_x_m": dx * 1e-9, "center_y_m": dy * 1e-9,
                         "size_m": frame_nm * 1e-9, "pixels": FRAME_PX}, optional=True)
        b.run("SaveScan", {}, optional=True)
        frame = b.latest_frame()
        if frame is None:
            continue
        res = b.run("LocateStepEdge", {"scan_path": str(frame), "channel": "Z"}, optional=True)
        if b.data(res, "verdict") != "step_edge":
            continue
        ex, ey = b.data(res, "edge_x_m"), b.data(res, "edge_y_m")
        if ex is None or ey is None:
            continue
        # the scan-frame angle: the skill reports the image-frame one alongside it, and the
        # positions below are scan-frame coordinates
        ang = b.data(res, "edge_angle_scan_deg")
        return float(ex), float(ey), float(ang or 0.0), b.data(res, "step_height_pm")
    return None


def _coord_epoch() -> int:
    try:
        from mast.core.coord_epoch import read_current_epoch

        return int(read_current_epoch() or 0)
    except Exception:  # noqa: BLE001
        return 0
