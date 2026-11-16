"""P3 scripted baseline: the resonances at the centre of a quantum corral with nothing but what
the instrument shows — no truth.

It runs the reference's measurement protocol (:mod:`stmbench.reference.p3`: drift compensated,
one 28 nm frame at 0.1 nm/px for the radius, spectra stabilised at 0.1 V / 1 nA, a bare-terrace
spectrum for the band bottom, three spectra at the centre combined by the median, the
resonances labelled with the box levels). What the reference reads from the truth, this reads
from observations, through :class:`stmbench.baselines._common.Blind` only:

* **where the corral is**: a 60 nm survey at 0.1 nm/px around the origin, where the task
  promises it; the circle through its bright pixels
  (:func:`stmbench.analysis.corral.locate_corral`). A survey that does not hold a whole ring
  is followed by a wider one at 0.25 nm/px;
* **the tip**: three shallow pokes 130 nm away come first whatever it shows (an unstable or
  dirty tip does not show in one frame, and a shallow contact only stabilises, sharpens and
  cleans an apex), then the survey judges it:

  - a *double* tip draws every feature twice: the copy of the ring shows as a second local
    maximum of the ring's autocorrelation (:func:`~stmbench.analysis.corral.ghost_strength`),
    and the spare atoms of the survey have partners at one displacement
    (:func:`stmbench.analysis.tip.diagnose`, on the survey with the ring blanked out);
  - a *blunt* tip widens the ring's cross-section
    (:func:`~stmbench.analysis.corral.ring_width_nm`) and smears the spare atoms;
  - a *dirty* tip puts its own peaks into every spectrum: the spectrum on bare terrace, which
    should show only the band onset, is where they show;
  - an *unstable* tip steps the rows of a frame (:class:`stmbench.baselines._common.Ops`);

  a blunt or double tip is pulsed 170 nm from the ring and judged again, a dirty or unstable
  one poked again; after any of them the creep of the long move is waited out;
* **a bare spot** for the band bottom: from the survey, the place farthest from every bump,
  step and the ring (:func:`bare_site`);
* **the drift**: two frames of the ring 30 s apart, cross-correlated.

What the baseline reaches is what a script reaches from the observations alone
(``stmbench.cli baseline --paper P3``).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..analysis.adatoms import locate
from ..analysis.corral import corral_radius, ghost_strength, label_levels, locate_corral, ring_width_nm
from ..analysis.drift import frame_shift
from ..analysis.spectra import band_edge, resonance_peaks
from ..analysis.tip import diagnose
from ..instrument import Instrument
from ..reference.p3 import _peak_values, truth_errors
from ._common import Blind, Ops

SCENARIO = "P3_corral_cu111"
SWEEP_V = (-0.55, 0.15)
STAB_V, STAB_PA = 0.1, 1000.0
POINTS = 351
#: the survey that finds the ring and judges the tip: 60 nm at 0.1 nm/px, 0.15 s/line (180 s)
SURVEY_NM, SURVEY_PX, SURVEY_LINE_S = 60.0, 600, 0.15
#: the wide survey when the first one shows no whole ring: 160 nm at 0.25 nm/px
WIDE_NM, WIDE_PX = 160.0, 640
#: the frame the radius is measured on, as in the reference (28 nm, 0.1 nm/px)
RADIUS_NM, RADIUS_PX, RADIUS_LINE_S = 28.0, 280, 0.1
#: a ring is whole when this share of its circle, inside the frame, is bright
COVERAGE_MIN, INSIDE_MIN = 0.8, 0.95
#: ring cross-section above this (nm, FWHM) is a blunt tip: sharp tips 0.8-1.1, blunt 1.2-1.6
BLUNT_FWHM_NM = 1.15
#: ghost-ring autocorrelation at or above this is a double tip: copies 0.09-0.27, structure of a
#: single ring at most 0.08
GHOST_MIN = 0.085
#: where the shallow pokes go and where a pulse is applied, from home (nm)
POKE_AWAY_NM, PULSE_AWAY_NM = 130.0, 170.0
MAX_ROUNDS = 12
#: the bare spot lies at least this far beyond the ring (nm) and this far from anything else
BARE_BEYOND_NM, BARE_CLEAR_NM = 8.0, 6.0
#: a bare-terrace spectrum this far from flat above the band onset (relative) carries a tip feature
FEATURE_MAX = 0.2


class NoRing(Exception):
    """The survey shows no whole ring."""


def _blank(frame: dict, centre, radius: float, pad_nm: float = 4.0) -> dict:
    """The frame with the ring (and a copy of it a double tip draws within ``pad_nm``) replaced
    by its median, so the tip is judged on the spare atoms outside."""
    z = np.asarray(frame["array"], float).copy()
    npx = float(frame["nm_per_px"])
    ny, nx = z.shape
    cx0, cy0 = frame["center_nm"]
    size = float(frame["size_nm"])
    yy, xx = np.mgrid[0:ny, 0:nx]
    x, y = cx0 - size / 2 + (xx + 0.5) * npx, cy0 + size / 2 - (yy + 0.5) * npx
    z[np.hypot(x - centre[0], y - centre[1]) <= radius + pad_nm] = np.nanmedian(z)
    out = dict(frame)
    out["array"] = z
    return out


def _to_scan(frame: dict, c) -> tuple[float, float]:
    """Scan coordinates of a point of a frame given as (x from its left edge, y up from its
    bottom edge), nm."""
    size = float(frame["size_nm"])
    return float(frame["center_nm"][0]) - size / 2 + float(c[0]), float(frame["center_nm"][1]) - size / 2 + float(c[1])


def find_corral(frame: dict) -> dict:
    """The ring in a survey frame: ``{centre (scan nm), radius_nm, coverage, ...}``; raises
    :class:`NoRing` when no whole ring is in it."""
    z = np.asarray(frame["array"], float)
    npx = float(frame["nm_per_px"])
    b = max(1, int(round(0.2 / npx)))                    # bin to 0.2 nm: the search over centres is the slow part
    ny, nx = z.shape[0] // b * b, z.shape[1] // b * b
    zb = np.where(np.isfinite(z), z, np.nanmedian(z))[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
    try:
        L = locate_corral(zb, npx * b)
    except ValueError as exc:
        raise NoRing(str(exc)) from exc
    if L["coverage"] < COVERAGE_MIN or L["inside"] < INSIDE_MIN:
        raise NoRing(f"coverage {L['coverage']:.2f}, inside {L['inside']:.2f}")
    L["centre_frame_nm"] = list(L["centre_nm"])
    L["centre"] = _to_scan(frame, L["centre_nm"])
    return L


def judge(frame: dict, L: dict) -> dict:
    """The tip as the survey shows it: ``{"verdict": "ok" | "blunt" | "double", ...}``."""
    c, r, npx = L["centre_frame_nm"], L["radius_nm"], float(frame["nm_per_px"])
    z = frame["array"]
    width = ring_width_nm(z, npx, c, r)
    ghost, lag = ghost_strength(z, npx, c, r)
    field = diagnose(_blank(frame, L["centre"], r))
    out = {"width_nm": width, "ghost": ghost, "ghost_lag_nm": [round(v, 2) for v in lag], "field": field["verdict"],
           "field_atoms": field["isolated"], "ghost_fraction": field["ghost_fraction"]}
    if ghost >= GHOST_MIN or field["verdict"] == "double":
        out["verdict"] = "double"
    elif width > BLUNT_FWHM_NM or field["verdict"] in ("blunt", "unknown"):
        out["verdict"] = "blunt"
    else:
        out["verdict"] = "ok"
    return out


def tip_feature(v: np.ndarray, didv: np.ndarray, e0_v: float) -> float:
    """How far a bare-terrace spectrum departs from flat above the band onset: the largest
    deviation of the smoothed curve from its median over the sweep from E0 + 50 mV, relative
    to that median. A 2-D band is a plateau there (good tips: at most ~0.1); a peak in the tip's
    density of states stands 0.3-0.8 above it."""
    from ..analysis.spectra import smooth

    v = np.asarray(v, float)
    y = smooth(didv, 21)
    sel = v >= e0_v + 0.05
    if sel.sum() < 20:
        return 0.0
    med = float(np.median(y[sel]))
    return float(np.max(np.abs(y[sel] - med)) / med) if med > 0 else 0.0


def _levelled(frame: dict) -> np.ndarray:
    """The topograph with the rows levelled against each other (a tip change steps every row after it)."""
    z = np.asarray(frame["array"], float)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    return z - np.concatenate([[0.0], np.cumsum(np.median(np.diff(z, axis=0), axis=1))])[:, None]


def bare_site(frame: dict, centre, radius: float, *, beyond_nm: float = BARE_BEYOND_NM,
              clear_nm: float = BARE_CLEAR_NM, edge_nm: float = 4.0) -> tuple[tuple[float, float], float]:
    """A bare spot of a survey for the band bottom: ((x, y) scan nm, its clearance nm). Anything
    that is not flat — atoms, clusters, steps, the ring — makes a pixel structure (the gradient of
    the topograph at 0.5 nm resolution, over 40 pm/nm: a Cu(111) step is 208 pm over ~1 nm, an
    atom 50-90 pm over 0.7 nm, a tilted terrace a few pm/nm); the spot is the pixel farthest
    from structure and from the frame's edge, at least ``beyond_nm`` beyond the ring, nearest the
    ring among those clear of everything by ``clear_nm``."""
    from scipy.ndimage import distance_transform_edt, gaussian_gradient_magnitude

    npx = float(frame["nm_per_px"])
    z = _levelled(frame)
    f = max(1, int(round(0.5 / npx)))
    ny, nx = z.shape[0] // f * f, z.shape[1] // f * f
    zc = z[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))
    px = npx * f
    grad = gaussian_gradient_magnitude(zc, 1.0) / px
    structure = grad > 40e-12
    cy_n, cx_n = zc.shape
    yy, xx = np.mgrid[0:cy_n, 0:cx_n]
    size = float(frame["size_nm"])
    x = float(frame["center_nm"][0]) - size / 2 + (xx + 0.5) * px
    y = float(frame["center_nm"][1]) + size / 2 - (yy + 0.5) * px
    d_ring = np.hypot(x - centre[0], y - centre[1])
    structure |= d_ring <= radius + 1.5
    # the frame's edge is not known to be bare: count it as structure
    e = max(1, int(round(edge_nm / px)))
    structure[:e, :] = structure[-e:, :] = True
    structure[:, :e] = structure[:, -e:] = True
    clear = distance_transform_edt(~structure) * px
    ok = d_ring >= radius + beyond_nm
    if not ok.any():
        raise RuntimeError("no room beyond the ring in the survey")
    score = np.where(ok, np.minimum(clear, clear_nm) - 0.001 * d_ring, -np.inf)
    j, i = np.unravel_index(int(np.argmax(score)), score.shape)
    return (float(x[j, i]), float(y[j, i])), float(clear[j, i])


def _sts(v: Blind, ops: Ops, integration_ms: float = 20.0) -> tuple[str, dict]:
    r = v.act("sts", v_start=SWEEP_V[0], v_end=SWEEP_V[1], points=POINTS, settle_ms=5.0,
              integration_ms=integration_ms, backward=False)
    return r["spectrum"], v.spectrum(r["spectrum"])


def _stabilise(v: Blind) -> None:
    v.act("set_bias", bias_v=STAB_V)
    v.act("set_setpoint", current_pa=STAB_PA)


def _image(v: Blind, ops: Ops) -> None:
    v.act("set_setpoint", current_pa=ops.imaging_pa)
    v.act("set_bias", bias_v=ops.imaging_v)


class Corral:
    """One session: the ring, where it is now, the tip, and the steps of the protocol."""

    def __init__(self, v: Blind, log: dict):
        self.v, self.log = v, log
        self.ops = Ops(v, log)
        self.centre = (0.0, 0.0)
        self.L = None
        self.frame = None
        self.tip: dict = {}

    # ── the tip ──
    def poke(self, n: int = 3) -> None:
        self.ops.poke_near(n, away_nm=POKE_AWAY_NM)
        self.v.act("wait", seconds=300.0)

    def pulse_round(self, k: int) -> None:
        """A 4 V, 50 ms pulse ``PULSE_AWAY_NM`` from the ring on fresh ground (a pulse sprays
        clusters ~40 nm around), two shallow pokes beside it (a pulse leaves the apex
        metastable), back to the ring."""
        cx, cy = self.centre
        x0, y0 = cx + 8.0 * k, cy + PULSE_AWAY_NM
        self.ops.go(x0, y0)
        self.v.act("pulse", bias_v=4.0, width_ms=50.0)
        for j in (1, 2):
            self.ops.go(x0 + 2.0 * j, y0)
            self.v.act("poke", depth_nm=0.4, dwell_s=0.5)
        self.ops.go(cx, cy)
        self.ops.jumps.clear()

    # ── the ring ──
    def survey(self) -> dict:
        """A survey around the last known centre; the ring in it. A frame without a whole ring
        is followed by the wide one."""
        cx, cy = self.centre
        f, _ = self.ops.frame(cx, cy, SURVEY_NM, SURVEY_PX, SURVEY_LINE_S, retakes=1)
        try:
            L = find_corral(f)
        except NoRing as exc:
            self.log.setdefault("wide_survey", []).append(str(exc))
            f, _ = self.ops.frame(cx, cy, WIDE_NM, WIDE_PX, WIDE_NM / 400.0, retakes=1)
            L = find_corral(f)
        self.frame, self.L = f, L
        self.centre = L["centre"]
        return L

    def find_and_fix(self) -> None:
        """Survey, judge the tip, and condition until it images well."""
        rounds = 0
        for k in range(MAX_ROUNDS + 1):
            L = self.survey()
            j = judge(self.frame, L)
            self.log.setdefault("judged", []).append({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in j.items()})
            if j["verdict"] == "ok" and not self.ops.unstable():
                break
            if k == MAX_ROUNDS:
                raise RuntimeError("the tip could not be conditioned")
            if j["verdict"] == "ok":                    # unstable
                self.ops.poke_near(3, away_nm=POKE_AWAY_NM)
                self.ops.go(*self.centre)
            else:
                self.pulse_round(k)
            rounds += 1
        self.log["tip_rounds"] = rounds
        if rounds:
            self.v.act("wait", seconds=300.0)

    # ── the measurement ──
    def relocate(self) -> None:
        """The ring again, from a 40 nm frame at 0.2 nm/px around where it was: the sample has
        crept and drifted by nanometres since the survey (0.3-0.6 nm/min was seen)."""
        cx, cy = self.centre
        f, _ = self.ops.frame(cx, cy, 40.0, 200, 0.1)
        try:
            L = find_corral(f)
        except NoRing:
            L = self.survey()
            return
        self.centre = L["centre"]

    def drift(self) -> tuple[float, float]:
        """Two frames of the ring 30 s apart, with the compensation off; the features' velocity,
        and the ring's centre in the second."""
        self.v.act("drift_comp", on=False)
        cx, cy = self.centre
        frames = []
        for i in range(2):
            if i:
                self.v.act("wait", seconds=30.0)
            f, _ = self.ops.frame(cx, cy, 28.0, 224, 0.07)
            frames.append(f)
        dx, dy = frame_shift(frames[0]["array"], frames[1]["array"], frames[0]["nm_per_px"])
        dt_min = (frames[1]["t_start_s"] - frames[0]["t_start_s"]) / 60.0
        vx, vy = dx / dt_min, dy / dt_min
        self.v.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
        self.centre = _to_scan(frames[1], corral_radius(frames[1]["array"], frames[1]["nm_per_px"])["centre_nm"])
        return vx, vy

    def bare_spectrum(self) -> tuple[float, float]:
        """The band bottom E0 from a spectrum on bare terrace, and how far the spectrum above it
        departs from flat (a tip that carries an adsorbate puts its own peak there)."""
        (bx, by), clear = bare_site(self.frame, self.L["centre"], self.L["radius_nm"])
        rel = (bx - self.L["centre"][0], by - self.L["centre"][1])
        self.log["bare_clear_nm"] = round(clear, 1)
        cx, cy = self.centre
        self.ops.go(cx + rel[0], cy + rel[1])
        _stabilise(self.v)
        _, sp = _sts(self.v, self.ops)
        e0 = band_edge(sp["v"], sp["didv"])
        return e0, tip_feature(sp["v"], sp["didv"], e0)

    def measure(self) -> dict:
        v, ops = self.v, self.ops
        e0, dev = self.bare_spectrum()
        for _ in range(2):
            if dev < FEATURE_MAX:
                break
            self.log["dirty_peaks"] = self.log.get("dirty_peaks", 0) + 1
            _image(v, ops)
            self.poke()
            e0, dev = self.bare_spectrum()
        self.log["bare_feature"] = round(dev, 3)
        _image(v, ops)
        self.relocate()
        vx, vy = self.drift()
        cx, cy = self.centre
        ops.go(cx, cy)
        f, _ = ops.frame(cx, cy, RADIUS_NM, RADIUS_PX, RADIUS_LINE_S)
        res = corral_radius(f["array"], f["nm_per_px"])
        centre = _to_scan(f, res["centre_nm"])
        self.log["centre_shift_nm"] = [round(centre[0] - cx, 2), round(centre[1] - cy, 2)]
        _stabilise(v)
        names, curves = [], []
        for _ in range(3):
            ops.go(*centre)
            n, sp = _sts(v, ops)
            names.append(n)
            curves.append(sp["didv"])
        peaks = resonance_peaks(sp["v"], np.median(np.vstack(curves), axis=0), e0)
        peaks_v = [p["v"] for p in peaks]
        labels = label_levels(peaks_v, e0, res["radius_nm"])
        return {"radius_nm": float(res["radius_nm"]), "radius_frame": f["frame"], "spectra": names, "e0_mev": e0 * 1e3,
                "peaks_v": peaks_v, "labels": labels, "drift_comp_nm_per_min": [float(vx), float(vy)],
                "centre": centre}


def claim_ids(text: str) -> list[str]:
    """The results the rule card lists, in order."""
    return re.findall(r"^\d+\. (\w+)[:：]", text, flags=re.M)


def play(v: Blind, log: dict) -> dict:
    """One P3 session through ``v``: what to report and the evidence for it."""
    ids = claim_ids(v.task())
    c = Corral(v, log)
    c.poke()
    c.find_and_fix()
    m = c.measure()
    log.update({k: m[k] for k in ("e0_mev", "drift_comp_nm_per_min", "radius_nm")})
    log["peaks_found"] = len(m["peaks_v"])
    claims = [{"id": i, "kind": "peak_in_list"} for i in ids if i.startswith("peak")]
    got = {cid: (val * 1e3, "meV", m["spectra"]) for cid, val in _peak_values(claims, m["labels"], m["peaks_v"]).items()}
    for cid in ids:
        if cid.startswith("radius"):
            got[cid] = (m["radius_nm"], "nm", [m["radius_frame"]])
    return got


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P3 episode by the baseline; what it reported and how it scored, and (from the truth,
    after the episode only) how far off it was."""
    inst = Instrument(scenario, seed, out_dir)
    log: dict = {}
    got, error = {}, None
    try:
        got = play(Blind(inst), log)
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
    for cid, (val, unit, ev) in got.items():
        inst.report(cid, val, unit=unit, evidence=ev)
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    values = {cid: float(val) for cid, (val, _, _) in got.items()}
    want, err = truth_errors(inst, values, [c for c in values if c.startswith("radius")]) if values else ({}, {})
    return {"paper": "P3", "seed": int(seed), "success": bool(out["success"]), "partial": float(out.get("partial", 0.0)),
            "error": error, "values": values, "truth": want, "abs_err": err,
            "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
            "instrument_s": float(inst.used_s),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
