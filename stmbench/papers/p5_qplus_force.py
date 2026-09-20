"""P5 — Sader–Jarvis 2004 / Huber 2019: the force between the tip and one atom.

Start the oscillation, put the tip over the atom, take Δf(z) there and again on clean copper,
invert the pair. The background curve is not optional: the van der Waals attraction between
the tip cone and the surface is the same order as the bond, and only the difference of the two
curves is the short-range force the paper is about.
"""
from __future__ import annotations

import math

from ._clusters import bright_blobs
from ._drift import null_the_drift
from ._runner import Baseline

# an adatom is 70 pm on a surface whose steps are 208: without a plane subtraction and
# a threshold near the noise, the segmentation never sees it. Measured 2026-09-07: the
# defaults find nothing but frame-corner artefacts, these find every atom to 0.1 nm.
CLUSTER_THRESHOLD_MAD = 2.0

#: the amplitude the force curves are taken at
AMPLITUDE_M = 50e-12
#: how long to keep asking whether the sensor has rung up. tau = Q/(pi f0) is about 0.2 s at
#: Q = 2e4 and f0 = 3e4 Hz, so a couple of seconds is many time constants; the poll is there
#: because an enabled output does not guarantee that the sensor is oscillating.
RING_UP_TIMEOUT_S = 8.0

SURVEY_NM = 70.0
#: cheap registration frames for the drift measurement
DRIFT_NM, DRIFT_PX = 20.0, 128
#: wide enough to cover the whole area the task points at (Surface.CLEAR_AREA_REACH_M)
#: 0.18 nm/px: an adatom is 0.7 nm across, so it has to be several pixels wide
SURVEY_PX = 384
#: how many bright features to try before giving up on finding an atom
N_CANDIDATES = 3
Z_OFFSET_M = 0.3e-9
Z_SWEEP_M = 0.5e-9
N_POINTS = 256
BACKGROUND_OFFSET_NM = 6.0


def run_p5(host, scenario, *, seed: int = 0, extra: dict | None = None) -> dict:
    b = Baseline(host, scenario, seed)
    extra = dict(extra or {})

    status = b.run("GetPLLStatus", {"modulator_index": 1}, optional=True)
    f0 = b.data(status, "center_freq_hz") or b.data(status, "f0_hz")
    b.run("SetPLLAmpCtrlSetpnt", {"modulator_index": 1, "setpoint_m": 50e-12}, optional=True)
    b.run("PLLOnOff", {"modulator_index": 1, "output_on": True,
                       "amp_ctrl_on": True, "phase_ctrl_on": True}, optional=False)
    _wait_for_oscillation(b, float(extra.get("amplitude_m", AMPLITUDE_M)))
    b.run("PLLFreqShiftAutoCenter", {"modulator_index": 1}, optional=True)

    cands = _candidates(b, float(extra.get("survey_nm", SURVEY_NM)))
    if not cands:
        return b.finish(note="no adatom found to measure")

    # A bright blob in topography can be an adatom, an adsorbate or a step edge, and nothing
    # in the image says which. What identifies the force signal: only a surface adatom gives a deep
    # short-range well. So try the candidates and keep the one whose curve goes deepest.
    # every curve has to land on the same atom, and the sweep pair takes many minutes
    drift = null_the_drift(b, cands[0], frame_nm=DRIFT_NM, pixels=DRIFT_PX)

    best = None
    for n, (ax, ay) in enumerate(cands):
        b.run("MoveToXY", {"x_m": ax, "y_m": ay, "wait": True}, optional=True)
        got = b.run("AcquireDeltaFCurve", {
            "save_basename": f"P5_try{n}", "z_sweep_distance_m": Z_SWEEP_M,
            "z_offset_m": Z_OFFSET_M, "num_points": N_POINTS}, optional=True)
        path, df_min = b.data(got, "path"), b.data(got, "df_min_hz")
        if path and df_min is not None and (best is None or float(df_min) < best[2]):
            best = (ax, ay, float(df_min), str(path))
    if best is None:
        return b.finish(note="no Δf curve on the atom", drift=drift, f0_hz=f0,
                        n_candidates=len(cands))
    ax, ay, df_min, atom_path = best

    b.run("MoveToXY", {"x_m": ax + BACKGROUND_OFFSET_NM * 1e-9, "y_m": ay, "wait": True},
          optional=True)
    bg = b.run("AcquireDeltaFCurve", {
        "save_basename": "P5_bg", "z_sweep_distance_m": Z_SWEEP_M,
        "z_offset_m": Z_OFFSET_M, "num_points": N_POINTS}, optional=True)
    bg_path = b.data(bg, "path")

    inv = b.run("InvertForceSaderJarvis", {
        "dat_path": str(atom_path),
        "background_dat_path": str(bg_path or ""),
        "k_n_per_m": _stiffness(host),
    }, optional=True)
    f_min = b.data(inv, "f_min_pn")
    decay = b.data(inv, "decay_length_pm")
    e_bind = b.data(inv, "e_bind_mev")
    if f_min is not None:
        b.report("f_min_pn", float(f_min), "pN")
    if decay is not None:
        b.report("f_decay_pm", float(decay), "pm")
    if e_bind is not None:
        b.report("e_bind_mev", float(e_bind), "meV")
    return b.finish(drift=drift, f0_hz=f0, atom_dat=atom_path, background_dat=bg_path,
                    n_candidates=len(cands), atom_nm=[ax * 1e9, ay * 1e9], df_min_hz=df_min,
                    inversion=b.data(inv, "verdict"),
                    well_posedness=b.data(inv, "well_posedness"),
                    forward_residual=b.data(inv, "forward_residual"))


def _candidates(b: Baseline, survey_nm: float) -> list[tuple[float, float]]:
    """The bright features nearest the origin, closest first."""
    b.run("ScanAt", {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": survey_nm * 1e-9,
                     "pixels": SURVEY_PX}, optional=True)
    b.run("SaveScan", {}, optional=True)
    frame = b.latest_frame()
    if frame is None:
        return []
    # compact bumps above the LOCAL background (``_clusters``): a plane-levelled MAD threshold
    # put the frame's creep-bowed corners above the atoms and sent the force curves to bare
    # copper (2026-09-13 calibration gate)
    clusters = list(bright_blobs(frame))
    clusters.sort(key=lambda q: math.hypot(q[0], q[1]))
    return clusters[:N_CANDIDATES]


def _stiffness(host) -> float:
    """The sensor's spring constant, from the tip registration the host declared.

    It is in no controller header — on instrument hardware it comes from the sensor's own calibration,
    which is why the harness registers it and the skill asks for it."""
    facts = getattr(host, "facts", {}) or {}
    tip = facts.get("tip") if isinstance(facts.get("tip"), dict) else {}
    for src in (tip, facts):
        val = src.get("qplus_k_n_per_m")
        if val:
            return float(val)
    return 1800.0


def _wait_for_oscillation(b: Baseline, want_m: float) -> bool:
    """Poll the amplitude until it reaches its setpoint.

    Turning the output on does not make the sensor oscillate; it rings up over a few Q/(pi f0).
    A force curve taken before that is a curve of the undriven noise floor, and it looks like
    a perfectly ordinary curve."""
    import time

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < RING_UP_TIMEOUT_S:
        res = b.run("ReadTipOscillationAmplitude", {}, optional=True)
        amp = b.data(res, "amplitude")
        if amp is not None and float(amp) >= 0.8 * want_m:
            return True
        time.sleep(0.4)
    return False
