"""P2 privileged reference solution: the Cu(111) surface state from standing waves at a step edge.

The truth navigates: which step edge to image (a wide terrace on the side the wave runs into,
a place where the edge runs straight, no point scatterer near), how far the sample has drifted
at the moment a frame starts, what state the tip is in. The reported numbers come from the
dI/dV maps the instrument measured, through :mod:`stmbench.analysis.standing_waves`.

Protocol:

1. a tip that is unstable, carries an adsorbate, or is blunt or multiple (a double tip puts a
   ghost of the edge into every map) is conditioned >= 150 nm from the work area: a 4 V pulse
   sprays clusters ~40 nm around. Back at the work area it waits only 30 s: every frame is
   centred on the drift predicted for its middle (read from the world), and the alignment of
   each row at its own edge absorbs what the creep of the long move adds;
2. the work area is chosen after the conditioning, on the sample as it then is: a step edge
   with a terrace >= 60 nm wide on the side the ripples run into (a narrower terrace puts the
   next edge's ripples into the profile and biases m*), at a bend of the edge where it runs
   along the mean step direction, with no point scatterer within 25 nm;
3. the lock-in is switched on (10 mV), the fast scan direction is turned across the edge
   (toward the wide terrace) so that every row is one profile, and eight 24 nm frames at
   0.2 nm/px are taken from -0.30 V to +0.12 V, each re-centred for the drift expected at its
   middle so that the edge sits 4 nm inside the frame's left side every time;
4. in each frame the edge is found in every row of the topograph, the rows of the dI/dV map are
   aligned at their own edge and averaged, and k comes from a standing-wave fit with a free
   phase. A second pass refits each k around the dispersion the first pass found;
5. a straight line in k^2 against E gives E0 and m*.

A tip changes spontaneously during a session (on seed 0 the conditioned single apex turned
double within ~700 s, and a double tip puts a ghost of the edge into the maps, which read k 1 %
high): after every frame the tip is checked and a frame taken with a changed tip is retaken
after conditioning. Only the forward lines are used (:func:`_profile`). A frame whose dI/dV
channel is blank is retaken.

The edge is taken where it runs along the mean step direction (``MAX_SLOPE``, 1.5 deg), so
the rows cross it along its local normal as well as the mean one, and the profile reads the
true wavelength whichever of the two distances the step's wave is a function of (the two differ
by sqrt(1 + w'^2) - 1 < 0.04 % there). The simulator uses the distance across the local edge
since 2026-11-27 and the distance along the mean normal before; across that change the results
moved within their noise (|m* error| p90 0.65 % before, 0.66 % after).

On seeds 0-29, run locally at df705e6 (2026-11-27, the simulator with both surface-state fixes
of that day): 30/30 reproduced, |E0 error| p90 2.0 meV (max 2.3), |m* error| p90 0.7 % (max
0.85 %), 1213 s of instrument time (median; p90 1401 s), about 25 s of wall clock per seed.
Every terrace was >= 60 nm wide. The validation run on khwu is recorded in docs/DESIGN.md
(section 9.10) and docs/results/p2_reference_seeds_0-29.json.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis.standing_waves import (edge_positions, fit_dispersion, fit_step_wave, k_guess,
                                       step_profile)
from ._common import Session, claims_ok, condition_tip, tip_needs_poke

SCENARIO = "P2_standing_waves_cu111"
ENERGIES_V = (-0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12)
FOV_NM, PIXELS, LINE_S = 24.0, 120, 0.6
#: the edge sits this far inside the frame's left side, so the profile can reach ~18 nm and the
#: frame's centre is (12 - EDGE_AT) nm from the edge (the evidence rule asks for < 10 nm)
EDGE_AT_NM = 4.0
MIN_TERRACE_NM, MIN_LOWER_NM = 60.0, 8.0
CLEAR_NM = 25.0
#: the edge runs along the mean step direction within ~1.5 deg (|w'| of its wander)
MAX_SLOPE = 0.027
D_MAX_NM = 17.0
LOCKIN_MV = 10.0
#: after a long move: only the drift prediction matters (it is read from the world), not the creep
SETTLE_S = 30.0
#: blank or edgeless maps retaken in one run, over all its energies together
MAX_RETAKES = 2


def _wander(site, v):
    """The step lines' wander at along-step coordinate ``v`` (m): ``Surface._staircase_u``."""
    a, p = site.wander_amp, site.wander_period
    return a * np.sin(2 * np.pi * v / p) + 0.4 * a * np.sin(2 * np.pi * v / (0.37 * p) + 1.3)


def pick_site(w, origin_xy, *, min_terrace_nm: float = MIN_TERRACE_NM) -> dict:
    """The work area nearest ``origin_xy`` (sample frame, m): an edge, the side of the wide
    terrace, the along-step position where the edge is straightest. Returns its frame centre
    (sample frame, m), the scan angle (deg, clockwise positive; the fast axis then points into
    the wide terrace) and the terrace width and local tilt for the record."""
    site = w.surface.site
    al = float(site.step_angle)
    n = np.array([math.cos(al), math.sin(al)])
    t = np.array([-math.sin(al), math.cos(al)])
    b = np.asarray(site.boundaries, float)
    ox, oy = origin_xy
    v_o = float(ox * t[0] + oy * t[1])
    half = 0.5 * FOV_NM * 1e-9
    best = None
    vs = v_o + np.arange(-400e-9, 400e-9, 1e-9)
    for j in range(1, len(b) - 1):
        for side in (+1, -1):
            wide = (b[j + 1] - b[j]) if side > 0 else (b[j] - b[j - 1])
            low = (b[j] - b[j - 1]) if side > 0 else (b[j + 1] - b[j])
            if wide < min_terrace_nm * 1e-9 or low < MIN_LOWER_NM * 1e-9:
                continue
            for v in vs:
                # how far the edge strays from a line along t over the frame's extent
                vv = v + np.linspace(-half, half, 9)
                stray = float(np.max(np.abs(_wander(site, vv) - _wander(site, v))))
                if stray > 0.5e-9:
                    continue
                dv = 1e-9
                slope = float((_wander(site, v + dv) - _wander(site, v - dv)) / (2 * dv))
                if abs(slope) > MAX_SLOPE:
                    continue
                u_edge = b[j] - float(_wander(site, v))
                e_pt = n * u_edge + t * v
                c = e_pt + side * n * (half - EDGE_AT_NM * 1e-9)
                dist = math.hypot(c[0] - ox, c[1] - oy)
                if best is not None and dist >= best[0]:
                    continue
                if w.surface.scatterers_near(float(c[0]), float(c[1]), CLEAR_NM * 1e-9):
                    continue
                best = (dist, dict(centre=(float(c[0]), float(c[1])), edge=(float(e_pt[0]), float(e_pt[1])),
                                   side=side, angle_deg=_wrap_deg(-math.degrees(math.atan2(side * n[1], side * n[0]))),
                                   terrace_nm=wide * 1e9, tilt_deg=math.degrees(math.atan(slope)), edge_index=j))
    if best is None:
        raise RuntimeError("no step edge with a wide clean terrace near the tip")
    return best[1]


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _centre_for_frame(s: Session, pick: dict, duration_s: float) -> tuple[float, float]:
    """Scan coordinates (nm) of the frame centre such that the sample point ``pick['centre']`` is
    at the middle of the frame as the drift will have moved it by then (privileged)."""
    ox, oy, _ = s.w.drift_offset(s.w.clock.sim() + 0.5 * duration_s)
    cx, cy = pick["centre"]
    return (cx - ox) * 1e9, (cy - oy) * 1e9


def _truth_value(inst, path: str) -> float:
    node = inst.world.truth()
    for key in path.split("."):
        node = node[key]
    return float(node)


def _profile(inst, frame: str, nm_per_px: float) -> dict | None:
    """The averaged profile of one dI/dV frame, from its forward lines. The backward lines are
    displaced by the scanner's hysteresis loop, a position error that grows along the line (up to
    ~0.3 nm in a 24 nm frame) and so stretches the distances by a few percent; the forward lines
    are the undistorted ones (comparing the two directions shows it). Pooling the directions
    only averages the distortion in."""
    z = inst.frame(frame, "z", "forward")["array"]
    a = inst.frame(frame, "didv", "forward")["array"]
    if not np.isfinite(a).all():
        return None                                  # a blank (NaN) map: the caller retakes it
    p = step_profile(a, edge_positions(z, nm_per_px), nm_per_px, side=+1, d_max_nm=D_MAX_NM)
    return p if p["n_rows"] >= 20 else None


def _tip_fit(w) -> bool:
    """Privileged: the tip still images a single clean apex (spontaneous changes happen during
    a session and turn a good tip into a double or an unstable one)."""
    return not tip_needs_poke(w, clean_ldos=True, imaging=True)


def run(seed: int, out_dir: str | Path | None = None) -> dict:
    """One P2 episode by the reference protocol; returns what it reported and how it scored."""
    s = Session(SCENARIO, seed, out_dir)
    inst, w = s.inst, s.w
    tx, ty = (v * 1e-9 for v in inst.status()["tip_nm"])
    ox, oy, _ = w.drift_offset(w.clock.sim())
    pick = pick_site(w, (tx + ox, ty + oy))
    rounds = tip_changes = blank = 0
    inst.act("lockin", on=True, amplitude_mv=LOCKIN_MV)
    nm_per_px = FOV_NM / PIXELS
    frame_s = PIXELS * 2 * LINE_S
    maps: dict[float, tuple[str, dict]] = {}
    pending = list(ENERGIES_V)
    for _ in range(3 * len(ENERGIES_V)):
        if not pending:
            break
        if not _tip_fit(w):
            r = condition_tip(s, *pick["centre"], clean_ldos=True, imaging=True, r_min=160e-9,
                              r_max=200e-9, clear=5e-9)
            rounds += r
            pick = pick_site(w, pick["centre"])      # the pulses sprayed clusters: look again
            s.go(*pick["centre"])
            inst.act("wait", seconds=SETTLE_S)
        e = pending[0]
        inst.act("set_bias", bias_v=e)
        cx, cy = _centre_for_frame(s, pick, frame_s)
        r = inst.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=FOV_NM, pixels=PIXELS,
                     angle_deg=pick["angle_deg"], line_time_s=LINE_S)
        if not _tip_fit(w):                          # it changed during the frame: retake it
            tip_changes += 1
            continue
        prof = _profile(inst, r["frame"], nm_per_px)
        if prof is None:
            # A blank (NaN) map: scipy 1.18.1's struve(0, z) has a numerical hole at z = 25.765,
            # and one pixel landing on it blanked the whole channel (the lock-in reference is the
            # frame mean). The simulator steps over the hole since 2026-11-27
            # (surface_state._struve0), so this no longer fires; it is kept, capped, for a map
            # that is blank or edgeless for another reason, so that such a map cannot loop.
            blank += 1
            if blank > MAX_RETAKES:
                raise RuntimeError(f"map at {e:+.2f} V unusable: {blank} blank or edgeless maps in "
                                   f"this run, more than the {MAX_RETAKES} retakes allowed")
            continue
        maps[e] = (r["frame"], prof)
        pending.pop(0)
    if pending:
        raise RuntimeError(f"could not complete the maps (missing {pending})")
    es = np.array(ENERGIES_V)
    names = [maps[e][0] for e in ENERGIES_V]
    profs = [maps[e][1] for e in ENERGIES_V]

    def fit_all(guess):
        return [fit_step_wave(p["d_nm"], p["y"], k_guess=g, sem=p["sem"]) for p, g in zip(profs, guess)]

    fits = fit_all([k_guess(e) for e in es])
    ok = np.array([f["ok"] for f in fits])
    disp = fit_dispersion(es[ok], [f["k"] for f, o in zip(fits, ok) if o],
                          [f["k_err"] for f, o in zip(fits, ok) if o])
    # second pass: start from the dispersion just measured
    fits = fit_all([k_guess(e, e0_ev=disp["e0_ev"], m_star=disp["m_star"]) for e in es])
    ok = np.array([f["ok"] for f in fits])
    disp = fit_dispersion(es[ok], [f["k"] for f, o in zip(fits, ok) if o],
                          [f["k_err"] for f, o in zip(fits, ok) if o])
    got = {"e0_mev": disp["e0_ev"] * 1e3, "m_eff": disp["m_star"]}
    units = {"e0_mev": "meV", "m_eff": "m_e"}
    for c in inst.scenario.claims:
        cid = str(c["id"])
        inst.report(cid, got[cid], unit=units[cid], evidence=names)
    out = inst.finish()
    truth = {str(c["id"]): _truth_value(inst, c["truth"]) for c in inst.scenario.claims}
    used_idx = np.flatnonzero(ok)
    dropped = None if disp["dropped"] is None else float(es[used_idx[disp["dropped"]]])
    return {"paper": "P2", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)),
            "values": got, "truth": truth,
            "abs_err": {"e0_mev": float(got["e0_mev"] - truth["e0_mev"])},
            "rel_err": {"m_eff": float(got["m_eff"] / truth["m_eff"] - 1.0)},
            "claims_ok": claims_ok(inst),
            "instrument_s": float(inst.used_s), "actions": int(inst.n_actions), "tip_rounds": rounds,
            "tip_changes": tip_changes, "retaken_maps": blank,
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"),
            "terrace_nm": float(pick["terrace_nm"]), "tilt_deg": float(pick["tilt_deg"]),
            "maps": [{"e_v": float(e), "k": f["k"], "k_err": f["k_err"], "n_rows": p["n_rows"],
                      "chi2_red": f["chi2_red"], "ok": f["ok"],
                      "k_true": float(w.surface.surface_state.k_of_e(e))}
                     for e, f, p in zip(es, fits, profs)],
            "dispersion": {"e0_mev": float(disp["e0_ev"] * 1e3), "m_star": float(disp["m_star"]),
                           "e0_err_mev": float(disp["e0_err"] * 1e3), "m_err": float(disp["m_err"]),
                           "dropped_e_v": dropped}}
