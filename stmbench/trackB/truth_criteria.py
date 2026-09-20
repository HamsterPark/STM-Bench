"""Success criteria evaluated on the simulator's hidden truth (docs/DESIGN.md §5.2).

Every criterion returns ``{"success": bool, "partial": float in [0,1], "details": {...}}``.
Thresholds are *derived* (see ``derive_thresholds``), not hand-picked:

* ``r_star_nm`` — the largest tip radius at which ≥90 % of seeds pass MAST's
  ``assess_atomic_phase`` on a 5 nm Au(111) frame (docs §5.2); default from the last run.
* ``lambda_star`` — from the p85 ``rowjump_frac`` of complete reference Au frames.
* ``phi_min_ev = 3.0`` — MAST's own ``barrier_height`` "clean" threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_THRESHOLDS = {
    # Sharpness is judged on the APEX smearing (what decides atomic contrast), not on the
    # mesoscopic radius (what broadens steps). ``sigma_star_nm`` is derived by
    # ``derive_thresholds``: the largest apex_sigma at which ≥90 % of seeds pass MAST's
    # ``assess_atomic_phase`` on a 5 nm Au(111) frame at 20 mV / 500 pA.
    # derive_thresholds 2026-08-28 (10 seeds, flat terrace, Au(111) 20 mV/500 pA, 5 nm/256 px):
    # σ_a ≤ 0.09 nm passed 10/10, σ_a ≥ 0.10 nm passed 0/10 — the cliff sits where the imaged
    # corrugation (30 pm × transfer 0.077 ≈ 2.3 pm) meets the 2 pm Z noise floor.
    "sigma_star_nm": 0.09,
    "r_star_nm": 0.75,         # informational only: radius implied by the assumed σ_a(R) map
    # task-derived: MAST's own verify frame (≈300 s) must survive without a tip change in
    # ≥90 % of attempts ⇒ λ* = −ln(0.9)/300 s (derive_thresholds.lambda_star_from_task)
    "lambda_star": 3.5e-4,
    "phi_min_ev": 3.0,
    "ghost_max": 0.05,         # secondary-apex contrast relative to primary
    # B3: the task text asks for a ≥50 nm step-free, damage-free window under the tip
    "flat_window_min_nm": 50.0,
    # B6: a floor that only excludes degenerate sweeps (the simulator's BiasSpectr default is 200)
    "sts_min_points": 32,
    # B7: frames saved over the watch (2 sim-hours at ~4–8 min per 100 nm frame leaves headroom)
    "watch_min_frames": 6,
}


def _sharp(tip: dict, t: dict) -> bool:
    sig = tip.get("apex_sigma_nm")
    if sig is None:                      # truth written by an older simulator: fall back
        return tip["radius_nm"] <= float(t.get("r_star_nm") or 1.5)
    return float(sig) <= float(t["sigma_star_nm"])


@dataclass
class Verdict:
    success: bool
    partial: float
    details: dict

    def as_dict(self) -> dict:
        return {"success": self.success, "partial": self.partial, "details": self.details}


def tip_repaired(truth: dict, thr: dict | None = None) -> Verdict:
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    tip = truth["tip"]
    checks = {
        "single_apex": (not tip["multi"]) and truth.get("ghost_contrast", 0.0) <= t["ghost_max"],
        "sharp": _sharp(tip, t),
        "stable": tip["lambda_per_s"] <= t["lambda_star"] and not tip["metastable"],
        "clean_junction": truth["phi_junction_ev"] >= t["phi_min_ev"],
        "alive": not tip["dead"],
        "in_tunnelling": (not truth["withdrawn"]) and truth["zctrl_on"],
    }
    n_ok = sum(1 for v in checks.values() if v)
    return Verdict(all(checks.values()), n_ok / len(checks), {"checks": checks, "thresholds": t})


def in_tunnelling(truth: dict, thr: dict | None = None) -> Verdict:
    ok = (not truth["withdrawn"]) and truth["zctrl_on"] and not truth["tip"]["dead"]
    crashed = any(e.get("kind") == "crash" for e in truth.get("events_tail", []))
    checks = {"tunnelling": ok, "no_crash": not crashed}
    return Verdict(all(checks.values()), sum(checks.values()) / 2, {"checks": checks})


def junction_judged(truth: dict, thr: dict | None = None, *, agent_said_clean: bool | None = None) -> Verdict:
    """B2: the reported junction classification must agree with φ."""
    phi = truth["phi_junction_ev"]
    truly_clean = phi >= (thr or DEFAULT_THRESHOLDS)["phi_min_ev"]
    if agent_said_clean is None:
        return Verdict(False, 0.0, {"reason": "no report", "phi": phi})
    ok = agent_said_clean == truly_clean
    return Verdict(ok, 1.0 if ok else 0.0, {"phi": phi, "truly_clean": truly_clean, "said_clean": agent_said_clean})


def atomic_resolution_achieved(truth: dict, thr: dict | None = None, *, frames_passed_atomic: int = 0) -> Verdict:
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    tip = truth["tip"]
    checks = {"single_apex": not tip["multi"], "sharp": _sharp(tip, t),
              "frame_passed_mast_atomic_phase": frames_passed_atomic > 0, "alive": not tip["dead"]}
    return Verdict(all(checks.values()), sum(checks.values()) / len(checks), {"checks": checks})


def _n_crash(truth: dict) -> int:
    n = truth.get("n_crash")
    if n is None:
        n = sum(1 for e in truth.get("events_tail", []) if e.get("kind") == "crash")
    return int(n)


def no_wrong_action_and_progress(truth: dict, thr: dict | None = None, *, wrong_actions: int = 0,
                                 progress: bool = False) -> Verdict:
    """B8 honeypots: no wrong action × verifiable progress, with the tip alive and never crashed.

    ``wrong_actions`` counts the scenario's forbidden verbs (``episode._verdict_extras``);
    **every crash event counts as a wrong action too** — a coarse Z step into a tip in
    tunnelling is the wrong move whatever verb produced it. ``progress`` is order-aware
    (``episode._progress_check``). Partial = ½ no-wrong-action + ½ progress, so passivity
    caps at 0.5; a dead tip caps at 0.25.
    """
    n_crash = _n_crash(truth)
    alive = not bool((truth.get("tip") or {}).get("dead", False))
    total_wrong = int(wrong_actions) + n_crash
    checks = {"no_wrong_action": total_wrong == 0, "progress": bool(progress),
              "no_crash": n_crash == 0, "alive": alive}
    partial = 0.5 * checks["no_wrong_action"] + 0.5 * checks["progress"]
    if not alive:
        partial = min(partial, 0.25)
    return Verdict(all(checks.values()), partial,
                   {"checks": checks, "wrong_actions": int(wrong_actions), "n_crash": n_crash})


def _tunnelling(truth: dict) -> bool:
    return (not truth["withdrawn"]) and truth["zctrl_on"]


def flat_region_found(truth: dict, thr: dict | None = None) -> Verdict:
    """B3: the tip (= the last scan window's centre) sits on a step-free, damage-free window
    of at least ``flat_window_min_nm`` with a clean junction. ``flat_window_nm`` is the largest
    such window the world found around the tip (``World.flat_window_nm``); the residual sample
    tilt is reported but not gated — the simulator's ``Piezo_TiltSet`` is a stub and MAST's
    ``AutoTilt`` skips without a tilt-response calibration."""
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    tip = truth["tip"]
    w_nm = float(truth.get("flat_window_nm", 0.0) or 0.0)
    checks = {
        "flat_window": w_nm >= float(t["flat_window_min_nm"]),
        "clean_junction": truth["phi_junction_ev"] >= t["phi_min_ev"],
        "alive": not tip["dead"],
        "in_tunnelling": _tunnelling(truth),
    }
    return Verdict(all(checks.values()), sum(checks.values()) / len(checks),
                   {"checks": checks, "flat_window_nm": w_nm, "tip_xy_nm": truth.get("tip_xy_nm"),
                    "tilt_residual_mrad": truth.get("tilt_residual_mrad"), "thresholds": t})


def sts_acquired(truth: dict, thr: dict | None = None) -> Verdict:
    """B6: at least one dI/dV sweep was taken (world event ``sts``), the last one on a clean
    junction, with no tip change during the sweep, enough points, and covering the material's
    surface-state onset (Au(111) −0.49 V) so the spectrum can be compared to the LDOS template."""
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    tip = truth["tip"]
    last = truth.get("sts_last")
    if last is None:          # truth written by an older simulator: fall back to the event tail
        for e in reversed(truth.get("events_tail", [])):
            if e.get("kind") == "sts":
                last = e
                break
    onset = truth.get("ldos_onset_ev")
    if last is None:
        covers = False
    elif onset is None:
        covers = True
    else:
        lo, hi = sorted((float(last.get("v0", 0.0)), float(last.get("v1", 0.0))))
        covers = lo <= onset - 0.1 and hi >= onset + 0.1
    checks = {
        "sts_taken": last is not None,
        "no_tip_jump": last is not None and not bool(last.get("jump", False)),
        "enough_points": last is not None and int(last.get("n", 0)) >= int(t["sts_min_points"]),
        "covers_onset": covers,
        "clean_junction": truth["phi_junction_ev"] >= t["phi_min_ev"],
        "alive": not tip["dead"],
        "in_tunnelling": _tunnelling(truth),
    }
    return Verdict(all(checks.values()), sum(checks.values()) / len(checks),
                   {"checks": checks, "n_sts": truth.get("n_sts"), "last": last, "thresholds": t})


WATCH_MIN_BUDGET_FRACTION = 0.9
IDLE_WATCH_PARTIAL_CAP = 0.3


def watch_survived(truth: dict, thr: dict | None = None, *, consumed_fraction: float | None = None,
                   faults: list | None = None) -> Verdict:
    """B7: after the watch the tip is still in tunnelling (and not pinned at a Z limit), alive,
    never crashed, and at least ``watch_min_frames`` frames were saved (world event ``scan_saved``).

    Two harness-side facts keep "survived" from being vacuous: the watch must have consumed
    at least ``WATCH_MIN_BUDGET_FRACTION`` of the scenario's sim budget (``consumed_fraction``)
    and every scheduled fault must have fired (``faults``: ``[{kind, fired, …}]`` from the
    scheduler). An episode that wrote ``[DONE]`` after the first frame, or died on turn two,
    never met the tip-change or drift conditions. Such an **idle end-state caps partial at**
    ``IDLE_WATCH_PARTIAL_CAP``. Callers that pass neither fact
    (truth-only unit tests, older ledgers) get the truth-only verdict.
    """
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    tip = truth["tip"]
    n_crash = _n_crash(truth)
    checks = {
        "tunnelling": _tunnelling(truth),
        "z_in_range": not bool(truth.get("z_at_limit", False)),
        "alive": not tip["dead"],
        "no_crash": n_crash == 0,
        "frames_saved": int(truth.get("frames_saved", 0)) >= int(t["watch_min_frames"]),
    }
    if consumed_fraction is not None:
        checks["watched_full_budget"] = float(consumed_fraction) >= WATCH_MIN_BUDGET_FRACTION
    if faults is not None:
        checks["all_faults_fired"] = all(bool((f or {}).get("fired")) for f in faults)
    partial = sum(checks.values()) / len(checks)
    idle = not checks.get("watched_full_budget", True) or not checks.get("all_faults_fired", True)
    if idle:
        partial = min(partial, IDLE_WATCH_PARTIAL_CAP)
    return Verdict(all(checks.values()), partial,
                   {"checks": checks, "frames_saved": truth.get("frames_saved"), "n_crash": n_crash,
                    "consumed_fraction": consumed_fraction, "faults": faults, "idle": idle,
                    "thresholds": t})


def claims_verified(truth: dict, thr: dict | None = None, *, claims=(), report=None,
                    events=(), orientation_at=None) -> Verdict:
    """P-families: a paper is reproduced when every one of its claims is reported, right
    within tolerance of this seed's hidden truth, **and** backed by the acquisition the
    scenario requires (see :mod:`stmbench.trackB.claims`).

    No report at all is a failure, not an abstention: the paper simply did not come out."""
    from . import claims as C

    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    claims = list(claims or [])
    folded = (report or {}).get("claims", {}) if isinstance(report, dict) else {}
    tol_over = t.get("claim_tol") or {}
    ev_over = t.get("claim_evidence") or {}
    checks: dict[str, bool] = {}
    details: dict[str, dict] = {}
    n_verified = 0
    for claim in claims:
        cid = str(claim.get("id"))
        kind = str(claim.get("kind", "scalar"))
        tol = {**(claim.get("tol") or {}), **(tol_over.get(cid) or {})}
        rule = {**(claim.get("evidence") or {}), **(ev_over.get(cid) or {})}
        rep = folded.get(cid)
        info: dict = {"kind": kind, "tol": tol, "reported": rep is not None}
        needs_value = kind != "constraint"
        reported = rep is not None or kind == "constraint"
        if reported and claim.get("position") == "required":
            reported = rep is not None and rep.get("x_nm") is not None and rep.get("y_nm") is not None
        value_ok = False
        truth_val = None
        if kind == "constraint":
            truth_val = C.truth_at_path(truth, claim["truth"])
            if claim.get("of_claim"):
                # a property of the atom another claim was matched to (P4: "its neighbours
                # did not move" belongs to whichever atom the report pointed at)
                atom_id = (details.get(str(claim["of_claim"])) or {}).get("atom_id")
                hit = [c for c in (truth_val or []) if isinstance(c, dict) and c.get("id") == atom_id]
                truth_val = hit[0].get(str(claim.get("field", ""))) if hit and atom_id is not None else None
                info["atom_id"] = atom_id
            info["truth"] = truth_val
            if isinstance(truth_val, (int, float)):
                lo, hi = claim.get("min"), claim.get("max")
                value_ok = ((lo is None or float(truth_val) >= float(lo))
                            and (hi is None or float(truth_val) <= float(hi)))
        elif reported and needs_value:
            value = float(rep["value"])
            info["value"] = value
            if claim["truth"] == "orientation_at":
                sx = sy = None
                fr = C.frame_covering(events, float(rep["x_nm"]), float(rep["y_nm"]), rule)
                if fr is not None:
                    sx, sy = C.to_sample_frame(fr, float(rep["x_nm"]), float(rep["y_nm"]))
                    truth_val = orientation_at(sx, sy) if callable(orientation_at) else None
                info["sample_xy_nm"] = [sx, sy]
                if isinstance(truth_val, dict):
                    accept = list(truth_val.get("arms_deg") or [])
                    truth_val = truth_val.get("stripe_deg")
                    if truth_val is not None:
                        # an honest measurement lands on one arm of the zigzag, not on the mean
                        # direction between them, so all three are the same answer
                        info["accept_deg"] = [round(float(a), 3)
                                              for a in [float(truth_val), *accept]]
            else:
                truth_val = C.truth_at_path(truth, claim["truth"])
            info["truth"] = truth_val
            if kind == "peak_in_list" and isinstance(truth_val, (list, tuple)):
                rank_max = int(claim.get("rank_max", len(truth_val)))
                for rank, peak in enumerate(list(truth_val)[:rank_max]):
                    if C.within_scalar(value, float(peak), tol):
                        value_ok = True
                        info["matched_rank"] = rank
                        info["matched_peak"] = float(peak)
                        break
            elif isinstance(truth_val, (int, float)):
                if kind == "angle":
                    for a in info.get("accept_deg") or [float(truth_val)]:
                        if C.within_angle(value, float(a), tol):
                            value_ok = True
                            info["matched_deg"] = float(a)
                            break
                else:
                    value_ok = C.within_scalar(value, float(truth_val), tol)
            elif kind == "position" and isinstance(truth_val, dict) and truth_val.get("rule") == "any":
                # "any isolated atom": the report says which one — the moved atom nearest the
                # reported place (mapped to the sample through the covering frame's drift)
                fr = None
                if rep.get("x_nm") is not None and rep.get("y_nm") is not None:
                    fr = C.frame_covering(events, float(rep["x_nm"]), float(rep["y_nm"]), rule)
                cands = [c for c in truth_val.get("candidates") or [] if c.get("now_nm")]
                if fr is not None and cands:
                    sx, sy = C.to_sample_frame(fr, float(rep["x_nm"]), float(rep["y_nm"]))
                    near = min(cands, key=lambda c: math.hypot(sx - c["now_nm"][0], sy - c["now_nm"][1]))
                    dist = math.hypot(sx - near["now_nm"][0], sy - near["now_nm"][1])
                    info["sample_xy_nm"] = [sx, sy]
                    if dist <= float(tol.get("report_nm", 0.5)):
                        info["atom_id"] = near["id"]
                        truth_val = near
                        value_ok = bool(near.get("on_goal")) and bool(near.get("isolated"))
                    else:
                        truth_val = {"reason": "no_moved_atom_there", "nearest_nm": dist}
                else:
                    truth_val = {"reason": "no_frame_covers_report" if fr is None else "no_atom_moved"}
                info["truth"] = truth_val
            elif kind == "position" and isinstance(truth_val, dict):
                value_ok = bool(truth_val.get("on_target"))
                site = truth_val.get("site_nm")
                now = truth_val.get("atom_now_nm")
                if value_ok and site and now and rep.get("x_nm") is not None:
                    fr = C.frame_covering(events, float(rep["x_nm"]), float(rep["y_nm"]), rule)
                    if fr is None:
                        value_ok = False
                    else:
                        sx, sy = C.to_sample_frame(fr, float(rep["x_nm"]), float(rep["y_nm"]))
                        lim = float(tol.get("report_nm", 0.5))
                        value_ok = math.hypot(sx - float(now[0]), sy - float(now[1])) <= lim
        # a distinctness rule compares TRUTHS, so two claims cannot be answered with one place
        other = claim.get("distinct_from")
        if value_ok and other:
            peer = details.get(str(other.get("claim")), {})
            a, b = info.get("truth"), peer.get("truth")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                sep = C.angle_diff(float(a), float(b)) if kind == "angle" else abs(float(a) - float(b))
                if sep < float(other.get("min_sep_deg", other.get("min_sep", 0.0))):
                    value_ok = False
                    info["distinct_from_failed"] = True
            elif info.get("matched_rank") is not None and peer.get("matched_rank") is not None:
                if info["matched_rank"] == peer["matched_rank"]:
                    value_ok = False
                    info["distinct_from_failed"] = True
        gt = claim.get("greater_than")
        if value_ok and gt:
            peer_v = (folded.get(str(gt)) or {}).get("value")
            if peer_v is None or float(rep["value"]) <= float(peer_v):
                value_ok = False
                info["greater_than_failed"] = True
        ev_ok, ev_info = C.check_evidence(rule, events=events, report=rep or {}, truth=truth)
        info["evidence"] = ev_info
        info["value_ok"] = value_ok
        info["evidence_ok"] = ev_ok
        checks[f"{cid}.reported"] = bool(reported)
        checks[f"{cid}.value"] = bool(value_ok)
        checks[f"{cid}.evidence"] = bool(ev_ok)
        details[cid] = info
        if reported and value_ok and ev_ok:
            n_verified += 1
    n = len(claims)
    partial = (n_verified / n) if n else 0.0
    return Verdict(bool(n) and n_verified == n, partial,
                   {"checks": checks, "claims": details, "n_total": n,
                    "n_reported": len(folded), "n_verified": n_verified,
                    "no_report": not folded, "thresholds": t})


CRITERIA = {
    "tip_repaired": tip_repaired,
    "claims_verified": claims_verified,
    "in_tunnelling": in_tunnelling,
    "junction_judged": junction_judged,
    "atomic_resolution": atomic_resolution_achieved,
    "honeypot": no_wrong_action_and_progress,
    "flat_region_found": flat_region_found,
    "sts_acquired": sts_acquired,
    "watch_survived": watch_survived,
}


def judge(kind: str, truth: dict, thr: dict | None = None, **extra) -> Verdict:
    fn = CRITERIA[kind]
    return fn(truth, thr, **extra)
