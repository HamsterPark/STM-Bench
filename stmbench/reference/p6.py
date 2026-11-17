"""P6 privileged reference solution: the Kondo resonance of one Co atom on Cu(111).

Privileged means the simulator's truth is used to *navigate* — which atom to use, where it is
now (drift included), where the leading apex has to be for the tip to sit over it, which nearby
spot is bare surface, whether the tip needs conditioning first — and never for a reported number.
Γ, q and ε_K come from bias sweeps the instrument measured, through
:mod:`stmbench.analysis.kondo` (the fit knows the rig temperature the instrument reports and the
modulation amplitude the operator set; it does not know how much of that amplitude reaches the
junction).

Protocol, as settled on seeds 0-29 (2026-11-17; docs/results/p6_reference_seeds_0-29.json and the
scenario's notes have the numbers and the variants that were measured):

1. the isolated Co atom nearest the scan origin (no other atom within 3 nm, ≥ 1.5 nm from a step,
   a bare spot ≥ 2.5 nm away on its terrace);
2. travel and conditioning at 20 mV and 20 pA. The setting dates from the draft, which started at
   100 mV on a rig whose tip sat on a qPlus fork: a conditioning poke above ~50 mV rang the sensor
   up and sprayed clusters that wiped every adatom within 15 nm (``World._apply_poke_outcome``; P5,
   which keeps the fork, pokes at 20 mV for the same reason), and the first version lost the target
   atom on 5 of 10 seeds. Since the decision of 2026-11-17 the STM rig's tip is an etched W wire,
   whose poke does not depend on the bias, and P6 starts at 50 mV; the setting is kept, and the
   reference re-run on seeds 0-29 with the etched tip and the new start is identical.
   An unstable tip, or one carrying an adsorbate, is poked on bare surface ~10 nm away. Blunt and
   double tips are *not* conditioned (``imaging=True`` does it): the pulses displace the apex and
   it reproduced 22/30 instead of 28/30, while the blunt and double tips of the unconditioned
   runs were measured as well as the good ones;
3. two frames 30 s apart give the lateral drift, which the compensation follows; the loop's Z drift
   is compensated too (it costs about 2 minutes; without it 27/30);
4. lock-in on at 3 mV zero-to-peak, feedback stabilised at +50 mV and 100 pA, outside the
   resonance. The modulation is a compromise the data show: the noise falls as 1/V_p, the line
   broadens as V_p (1 mV 15/30, 2 mV 26/30, 3 mV 28/30, 4 mV 28/30: 3 mV is the smaller broadening for the same score, and the fit relies less on
   the unknown attenuation, 0.8-1.0);
5. before every sweep the *leading apex* is put over the atom, at the position the atom will have
   half way through the sweep (the tip images and tunnels through that apex), moving again until
   it is within 0.03 nm; then a −50…+50 mV sweep, 201 points, 5 ms settle, 100 ms integration
   (200 ms or 300 ms did not help: the longer the sweep, the more a tip change or the drift
   residual costs). Six sweeps on the atom and four on a bare spot, alternating so that a slow
   change in the tip or the loop affects both alike;
6. the sweeps of one kind are averaged, the atom's divided by the bare reference, the Fano
   line fitted with the 3 mV kernel (the attenuation is not read), and Γ (meV), q and ε_K (meV)
   reported citing every sweep.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._common import Session, bare_spot, claims_ok, compensate_z_drift, condition_tip, lateral_drift

SCENARIO = "P6_kondo_co_cu111"
#: sweep window (V) and points: covers the claim's ±20 mV with margin for the baseline
SWEEP_V = (-0.05, 0.05)
POINTS = 201
#: stabilisation outside the resonance: ±50 mV is the sweep's end, the resonance is within ~10 mV
STABILISE_V, SETPOINT_PA = 0.05, 100.0
SETTLE_MS = 5.0
TRAVEL_V = 0.02


def _isolated_atom(w):
    """The atom of the field nearest the scan origin with no other atom within 3 nm and ≥ 1.5 nm
    from a step, and a bare spot ≥ 2.5 nm away on its terrace."""
    reg = w.surface.site.adatoms
    xy = reg.positions()
    cands = []
    for a in reg.atoms.values():
        x, y = reg.site_xy(a.i, a.j, a.sub)
        if sorted(np.hypot(xy[:, 0] - x, xy[:, 1] - y))[1] < 3.0e-9:
            continue
        if not w.surface.step_free_window(x, y, 3e-9):
            continue
        cands.append((math.hypot(x, y), x, y))
    for _, x, y in sorted(cands):
        try:
            bare_spot(w, x, y)
        except RuntimeError:
            continue
        return x, y
    raise RuntimeError("no isolated atom with a bare spot nearby")


def _lead_offset(w) -> tuple[float, float]:
    """The leading apex's lateral offset (m), chosen as the world's sweep chooses it
    (``World._sts_curve_resolved``): the apex with the largest current weight at the point the
    sweep starts from, ``a.w * exp(2 kappa (h_smooth + a.dz))``. With several apexes of unequal
    weight that is not always the one closest to the surface (privileged: read from the tip)."""
    apexes = getattr(w.tip, "apexes", None) or []
    if not apexes:
        return 0.0, 0.0
    if len(apexes) == 1:
        return float(apexes[0].dx), float(apexes[0].dy)
    ox, oy, _oz = w.drift_offset(w.clock.sim())
    px, py = np.array([w.tip_x + ox]), np.array([w.tip_y + oy])
    kappa = w.kappa_m()
    hs = np.array([float(w.surface.height_smooth(px + a.dx, py + a.dy)[0]) + a.dz for a in apexes])
    lead = apexes[int(np.argmax(np.array([a.w for a in apexes]) * np.exp(2.0 * kappa * (hs - hs.max()))))]
    return float(lead.dx), float(lead.dy)


def _apex_error_nm(w, sx: float, sy: float, lead_s: float) -> float:
    """How far (nm) the leading apex will be from the sample-frame point (sx, sy) ``lead_s``
    seconds from now, given the tip's position and the drift (privileged)."""
    dx, dy = _lead_offset(w)
    ox, oy, _ = w.drift_offset(w.clock.sim() + lead_s)
    return 1e9 * math.hypot(w.tip_x + ox + dx - sx, w.tip_y + oy + dy - sy)


def _put_apex_over(s: Session, sx: float, sy: float, lead_s: float, *, tol_nm: float = 0.03, moves: int = 4) -> float:
    """Move so that the leading apex is over the sample-frame point (sx, sy) ``lead_s`` seconds
    from now (half a sweep ahead: whatever drift the compensation leaves has carried the sample by
    then), correcting until it is within ``tol_nm`` (a move leaves creep, the drift goes on and a
    tip change shifts the apex: one move was off by 0.15 nm on some seeds). Returns the error left."""
    for _ in range(moves):
        dx, dy = _lead_offset(s.w)
        ox, oy, _oz = s.w.drift_offset(s.w.clock.sim() + lead_s)
        s.inst.act("move_tip", x_nm=(sx - dx - ox) * 1e9, y_nm=(sy - dy - oy) * 1e9)
        if _apex_error_nm(s.w, sx, sy, lead_s) < tol_nm:
            break
    return _apex_error_nm(s.w, sx, sy, lead_s)


def _sweep(s: Session, sx: float, sy: float, settle_ms: float, integration_ms: float, points: int) -> tuple[str, dict]:
    lead_s = 0.5 * points * (settle_ms + integration_ms) * 1e-3
    _put_apex_over(s, sx, sy, lead_s + 1.0)
    r = s.inst.act("sts", v_start=SWEEP_V[0], v_end=SWEEP_V[1], points=points, settle_ms=settle_ms,
                   integration_ms=integration_ms, backward=False)
    return r["spectrum"], s.inst.spectrum(r["spectrum"])


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str | Path = SCENARIO,
        amplitude_mv: float = 3.0, fit_amplitude_mv: float | None = None, integration_ms: float = 100.0,
        points: int = POINTS, atom_sweeps: int = 6, bare_sweeps: int = 4, drift_comp: bool = True,
        z_comp: bool = True, imaging: bool = False, condition: bool = True, travel_pa: float = 20.0,
        setpoint_pa: float = SETPOINT_PA) -> dict:
    """One P6 episode by the reference protocol; returns what it reported and how it scored.

    ``fit_amplitude_mv`` is the modulation amplitude the fit assumes (default: the one set), what
    the contrast run with the instrument's defaults changes. ``drift_comp=False`` and
    ``condition=False`` switch the corresponding steps off for the contrast runs."""
    from ..analysis import kondo      # imported here: the module arrives with the analysis branch

    s = Session(scenario, seed, out_dir)
    inst, w = s.inst, s.w
    temp_k = float(inst.info()["temperature_k"])
    ax, ay = _isolated_atom(w)
    bx, by = bare_spot(w, ax, ay)
    # travel and condition at TRAVEL_V, a setting from the qPlus fork the draft had (point 2 of the
    # module text; the etched tip's poke does not depend on the bias). The setpoint is lowered too
    # (travel_pa) as a precaution: the tip starts on top of the atom
    # and a drag at a low junction resistance could take it along, so every travel before the sweeps runs at
    # a high junction resistance
    inst.act("set_bias", bias_v=TRAVEL_V)
    inst.act("set_setpoint", current_pa=travel_pa)
    rounds = condition_tip(s, ax, ay, clean_ldos=True, imaging=imaging, r_min=10e-9, r_max=12e-9) if condition else 0
    vx = vy = vz = 0.0
    if drift_comp:
        vx, vy = lateral_drift(s, ax, ay)
        inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
        if z_comp:
            s.go(bx, by)
            vz = compensate_z_drift(inst, vx, vy)
    inst.act("set_bias", bias_v=STABILISE_V)
    inst.act("set_setpoint", current_pa=setpoint_pa)
    inst.act("lockin", on=True, amplitude_mv=amplitude_mv)
    atom, bare, names = [], [], []
    for i in range(max(atom_sweeps, bare_sweeps)):
        if i < atom_sweeps:
            n, sp = _sweep(s, ax, ay, SETTLE_MS, integration_ms, points)
            atom.append(np.asarray(sp["didv"], float))
            names.append(n)
            v = np.asarray(sp["v"], float)
        if i < bare_sweeps:
            n, sp = _sweep(s, bx, by, SETTLE_MS, integration_ms, points)
            bare.append(np.asarray(sp["didv"], float))
            names.append(n)
    a_mean, _ = kondo.mean_sweep(atom)
    b_mean, _ = kondo.mean_sweep(bare)
    vp = (fit_amplitude_mv if fit_amplitude_mv is not None else amplitude_mv) * 1e-3
    fit = kondo.fit_fano(v, a_mean, temp_k=temp_k, vmod_peak_v=vp, reference=b_mean)
    got = {"gamma_mev": fit.gamma_v * 1e3, "q_fano": fit.q, "e_k_mev": fit.e_k_v * 1e3}
    units = {"gamma_mev": "meV", "q_fano": "", "e_k_mev": "meV"}
    for c in inst.scenario.claims:
        cid = str(c["id"])
        inst.report(cid, float(got[cid]), unit=units[cid], evidence=names)
    out = inst.finish()
    truth = w.truth()["kondo"]
    t_of = {"gamma_mev": float(truth["gamma_mev"]), "q_fano": float(truth["q"]), "e_k_mev": float(truth["e_k_mev"])}
    lead_dx, lead_dy = _lead_offset(w)
    near = [{k: r[k] for k in ("near_kind", "near_nm", "near_kind_apex", "near_nm_apex", "near_nm_apex_end", "apex_offset_nm") if k in r}
            for r in w.sts_records]                       # what sts_pair reads
    return {"paper": "P6", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)),
            "values": {k: float(v_) for k, v_ in got.items()}, "truth": t_of,
            "rel_err": {"gamma_mev": float(got["gamma_mev"] / t_of["gamma_mev"] - 1.0)},
            "abs_err": {"q_fano": float(got["q_fano"] - t_of["q_fano"]),
                        "e_k_mev": float(got["e_k_mev"] - t_of["e_k_mev"])},
            "claims_ok": claims_ok(inst),
            "evidence_ok": {k: bool(d.get("evidence_ok", True))
                            for k, d in (inst.verdict.details or {}).get("claims", {}).items()},
            "fit_ok": bool(fit.success), "fit_r2": float(fit.r2),
            "instrument_s": float(inst.used_s), "tip_conditioning_rounds": rounds,
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"),
            "attenuation": float(w.lockin_attenuation), "sweeps": len(names),
            "sweep_near": near,
            "drift_comp": {"vx_nm_per_min": float(vx), "vy_nm_per_min": float(vy), "vz_pm_per_min": float(vz)},
            "protocol": {"amplitude_mv": amplitude_mv, "fit_amplitude_mv": vp * 1e3,
                         "integration_ms": integration_ms, "points": points, "atom_sweeps": atom_sweeps,
                         "bare_sweeps": bare_sweeps, "drift_comp": drift_comp, "z_comp": z_comp,
                         "imaging": imaging, "condition": condition, "travel_pa": travel_pa, "setpoint_pa": setpoint_pa},
            "lead_apex_offset_nm": [lead_dx * 1e9, lead_dy * 1e9]}
