"""P3 privileged reference solution: the resonances at the centre of a quantum corral.

The truth navigates — where the ring's centre is, how far the sample has drifted, what state
the tip is in — and the reported numbers come from what the instrument measured: the energies
from spectra (:mod:`stmbench.analysis.spectra`), the radius from a topograph
(:mod:`stmbench.analysis.corral`).

Protocol:

1. a tip that is unstable, carries an adsorbate (its own states put peaks into every
   spectrum), or is blunt or multiple (it blurs or doubles the ring) is conditioned ~150 nm
   from the ring: a 4 V pulse sprays clusters ~40 nm around;
   back at the ring, five minutes for the creep of that long move to settle;
2. lateral drift from two frames 30 s apart, compensated; then at once one 28 nm frame of the
   ring at 0.1 nm/px and 0.1 s/line, and the radius from it (a circle through the bright
   pixels, outliers dropped). Measured right before imaging so that the decaying creep is the
   same during both;
3. the junction is stabilised at 0.1 V / 1 nA. At the 10 mV imaging point the gap is so small
   that a sweep to -0.55 V runs the current into the preamp's clamp, and the clamp's onset
   looks like a pair of peaks near ±60 mV. A sweep reaching far above the resonances (to
   +0.55 V, stabilised at +0.6 V) is worse: the transmission makes dI/dV grow with bias, the
   noise scales with the curve's maximum, and the band edge and the low resonances drown;
4. one spectrum on bare terrace well outside the ring gives the band bottom E0: there the 2-D
   band's onset is the steepest rise. Inside the ring a resonance's flank is steeper, and
   reading E0 there shifts every peak's rank;
5. three spectra at the centre, -0.55 to +0.15 V, 20 ms per point, combined by the median;
6. the resonances are the local maxima at least 5 mV above E0 whose prominence is at least 8 %
   of the spectrum's range there (the truth's own definition), lowest first. They are labelled
   with the box's quantum numbers from the measured radius and E0 and Cu(111)'s textbook
   m* = 0.38 (:func:`stmbench.analysis.corral.label_levels`); the two lowest-resonance claims
   get n = 2 and n = 3.

On seeds 0–29 of the 2026-10-13 draws: 30/30 reproduced, |error| p90 2.7 meV (lower peak),
2.5 meV (upper), 0.04 nm (radius); 485 s of instrument time (median). A seed whose tip needed
conditioning spends five of those minutes waiting out the creep of the move back.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analysis.corral import corral_radius, label_levels
from ..analysis.spectra import band_edge, resonance_peaks
from ._common import Session, bare_spot, claims_ok, condition_tip, lateral_drift

SCENARIO = "P3_corral_cu111"
SWEEP_V = (-0.55, 0.15)
STAB_V, STAB_PA = 0.1, 1000.0
POINTS = 351


def _sts(s: Session, integration_ms: float) -> tuple[str, dict]:
    r = s.inst.act("sts", v_start=SWEEP_V[0], v_end=SWEEP_V[1], points=POINTS, settle_ms=5.0,
                   integration_ms=integration_ms, backward=False)
    return r["spectrum"], s.inst.spectrum(r["spectrum"])


def _peak_values(claims: list[dict], labels: dict[int, float], peaks_v: list[float]) -> dict[str, float]:
    """The lowest-resonance claims (``peak_in_list``: any of the three lowest, distinct,
    ascending) get the box levels n = 2, 3, … in the order they are listed. n = 1 sits so close
    to the band bottom that whether it is a resolvable peak varies with the seed; n = 2 and 3
    are among the lowest three resonances either way."""
    out: dict[str, float] = {}
    taken: list[float] = []
    k = 0
    for c in claims:
        if c.get("kind") != "peak_in_list":
            continue
        v = labels.get(2 + k)
        if v is None or v in taken:                     # fall back on the next unclaimed peak
            v = next((p for p in sorted(peaks_v) if p not in taken and (not taken or p > taken[-1])), None)
        if v is not None:
            out[str(c["id"])] = v
            taken.append(v)
        k += 1
    return out


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO, repeats: int = 3,
        integration_ms: float = 20.0) -> dict:
    """One P3 episode by the reference protocol; returns what it reported and how it scored."""
    s = Session(scenario, seed, out_dir)
    inst, w = s.inst, s.w
    ring = w.surface.corral_snapshot()
    cx, cy = (v * 1e-9 for v in ring["centre_nm"])
    radius_true = float(ring["radius_nm"]) * 1e-9
    outside = radius_true + 8e-9
    rounds = condition_tip(s, cx, cy, clean_ldos=True, imaging=True, r_min=radius_true + 150e-9,
                           r_max=radius_true + 200e-9, clear=5e-9)
    if rounds:
        s.go(cx, cy)
        inst.act("wait", seconds=300.0)
    vx, vy = lateral_drift(s, cx, cy)
    inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
    s.go(cx, cy)
    tx, ty = inst.status()["tip_nm"]
    fr = inst.act("scan", center_x_nm=tx, center_y_nm=ty, size_nm=28.0, pixels=280, line_time_s=0.1)
    radius_nm = corral_radius(inst.frame(fr["frame"])["array"], fr["nm_per_px"])["radius_nm"]
    inst.act("set_bias", bias_v=STAB_V)
    inst.act("set_setpoint", current_pa=STAB_PA)
    s.go(*bare_spot(w, cx, cy, r_min=outside, r_max=outside + 30e-9, clear=6e-9, same_terrace=False))
    _, bare = _sts(s, integration_ms)
    e0 = band_edge(bare["v"], bare["didv"])
    names, curves = [], []
    for _ in range(repeats):
        s.go(cx, cy)
        n, sp = _sts(s, integration_ms)
        names.append(n)
        curves.append(sp["didv"])
    v = sp["v"]
    peaks = resonance_peaks(v, np.median(np.vstack(curves), axis=0), e0)
    peaks_v = [p["v"] for p in peaks]
    labels = label_levels(peaks_v, e0, radius_nm)
    got = {cid: v * 1e3 for cid, v in _peak_values(inst.scenario.claims, labels, peaks_v).items()}
    for cid, mev in got.items():
        inst.report(cid, mev, unit="meV", evidence=names)
    radius_claims = [str(c["id"]) for c in inst.scenario.claims if c.get("truth") == "corral.radius_nm"]
    for cid in radius_claims:
        got[cid] = radius_nm
        inst.report(cid, radius_nm, unit="nm", evidence=[fr["frame"]])
    out = inst.finish()
    snap = w.truth()["corral"]
    truth = snap["peaks_mev"]
    want = {}
    for c in inst.scenario.claims:            # as the judge matches: the nearest of the lowest rank_max
        cid = str(c["id"])
        if c.get("kind") == "peak_in_list" and cid in got:
            cands = truth[: int(c.get("rank_max", len(truth)))]
            if cands:
                want[cid] = float(min(cands, key=lambda t: abs(t - got[cid])))
    want.update({cid: float(snap["radius_nm"]) for cid in radius_claims})
    return {"paper": "P3", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)),
            "values": got, "truth": want,
            "abs_err": {k: float(got[k] - want[k]) for k in want if k in got},
            "claims_ok": claims_ok(inst), "e0_mev": e0 * 1e3,
            "radius_nm": float(ring["radius_nm"]), "radius_measured_nm": float(radius_nm),
            "peaks_found": len(peaks), "instrument_s": float(inst.used_s), "tip_rounds": rounds,
            "spectra": len(names) + 1, "drift_comp_nm_per_min": [float(vx), float(vy)],
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition")}
