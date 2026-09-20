"""Judging a paper scenario: reported numbers against the simulator's hidden truth.

A claim is verified when three things hold, and the third is the one that makes the
benchmark about doing an experiment rather than about answering a question:

1. **reported** — the value arrived on the ``ReportResult`` channel (and, for a position
   claim, with coordinates);
2. **value** — it is inside the claim's tolerance of the generated truth for this seed.
   Because the truth is drawn per seed inside a physical range, a literature value alone is
   insufficient;
3. **evidence** — the episode's own event log shows the acquisition the result needs: a
   frame that covers the reported position at a usable field of view, spectra taken near a
   scatterer over the right energy window, or an atom that hopped. Numbers
   without the measurement behind them do not count as a reproduction.

Coordinates: an agent works in the controller scan frame while the surface is stored in the
sample frame, and hours of drift separate the two. Every saved frame records the drift it was
taken under, so a reported position is mapped through the covering frame — the agent is never
asked to know about drift, and a right answer at a right place is never failed for it.
"""
from __future__ import annotations

import math
from typing import Any

_ANG_DEFAULT_MOD = 180.0


# ── tolerance helpers ────────────────────────────────────────────────────────
def within_scalar(value: float, truth: float, tol: dict) -> bool:
    abs_tol = float(tol.get("abs", 0.0) or 0.0)
    rel_tol = float(tol.get("rel", 0.0) or 0.0)
    lim = max(abs_tol, rel_tol * abs(float(truth)))
    if lim <= 0:
        lim = 1e-12
    return abs(float(value) - float(truth)) <= lim


def angle_diff(a: float, b: float, mod: float = _ANG_DEFAULT_MOD) -> float:
    d = abs(float(a) - float(b)) % mod
    return min(d, mod - d)


def within_angle(value: float, truth: float, tol: dict) -> bool:
    mod = float(tol.get("mod", _ANG_DEFAULT_MOD))
    return angle_diff(value, truth, mod) <= float(tol.get("abs", 5.0))


# ── truth resolution ─────────────────────────────────────────────────────────
def truth_at_path(truth: dict, path: str) -> Any:
    cur: Any = truth
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# ── evidence ─────────────────────────────────────────────────────────────────
def _frames(events) -> list[dict]:
    return [e for e in events or [] if e.get("kind") == "scan_saved"]


def _frame_ok(frame: dict, rule: dict) -> bool:
    if not frame.get("complete", True):
        return False
    w, h = float(frame.get("w_m", 0.0)), float(frame.get("h_m", 0.0))
    nx = int(frame.get("nx", 0) or 0)
    if min(w, h) * 1e9 < float(rule.get("min_fov_nm", 0.0)):
        return False
    max_px = rule.get("max_nm_per_px")
    if max_px is not None and (nx <= 0 or (w * 1e9) / nx > float(max_px)):
        return False
    return True


def frame_covering(events, x_nm: float, y_nm: float, rule: dict) -> dict | None:
    """The last usable frame whose footprint contains a scan-frame point."""
    best = None
    for fr in _frames(events):
        if not _frame_ok(fr, rule):
            continue
        cx, cy = float(fr.get("cx_m", 0.0)) * 1e9, float(fr.get("cy_m", 0.0)) * 1e9
        w, h = float(fr.get("w_m", 0.0)) * 1e9, float(fr.get("h_m", 0.0)) * 1e9
        ang = math.radians(float(fr.get("angle_deg", 0.0) or 0.0))
        dx, dy = x_nm - cx, y_nm - cy
        c, s = math.cos(-ang), math.sin(-ang)
        u, v = c * dx - s * dy, s * dx + c * dy
        if abs(u) <= w / 2 and abs(v) <= h / 2:
            best = fr
    return best


def to_sample_frame(frame: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Scan-frame nm → sample-frame nm, using the drift the frame recorded."""
    dx, dy = (frame.get("drift_m") or [0.0, 0.0])[:2]
    return x_nm + float(dx) * 1e9, y_nm + float(dy) * 1e9


def frame_covering_sample(events, x_nm: float, y_nm: float, rule: dict) -> dict | None:
    """The last usable frame whose footprint contains a **sample-frame** point.

    A place on the sample sits at different scan coordinates in every frame, because the
    piezo has drifted between them. Each frame recorded the drift it was taken under, so the
    point is converted into that frame's own coordinates before the footprint test.
    """
    best = None
    for fr in _frames(events):
        if not _frame_ok(fr, rule):
            continue
        dx, dy = (fr.get("drift_m") or [0.0, 0.0])[:2]
        if frame_covering([fr], x_nm - float(dx) * 1e9, y_nm - float(dy) * 1e9, rule) is not None:
            best = fr
    return best


def _spectra(events, kind: str = "sts") -> list[dict]:
    return [e for e in events or [] if e.get("kind") == kind
            and e.get("delivered", True) is not False]


def check_evidence(rule: dict, *, events, report: dict, truth: dict) -> tuple[bool, dict]:
    """Check whether the episode acquired the measurement required by the claim."""
    kind = str(rule.get("kind", "frame"))
    if kind == "frame":
        got = [f for f in _frames(events) if _frame_ok(f, rule)]
        return bool(got), {"kind": kind, "n_frames": len(got)}
    if kind == "frame_covers":
        x, y = report.get("x_nm"), report.get("y_nm")
        if x is None or y is None:
            return False, {"kind": kind, "reason": "no_position"}
        fr = frame_covering(events, float(x), float(y), rule)
        return fr is not None, {"kind": kind, "frame_idx": (fr or {}).get("idx"),
                                "covered": fr is not None}
    if kind == "frame_covers_after":
        # a literal ``point`` and a reported position are scan coordinates; a ``point`` given
        # as a truth path (``corral.centre_nm``) is a place on the sample, so it has to be
        # put back into each frame's coordinates through that frame's drift
        x, y = report.get("x_nm"), report.get("y_nm")
        in_sample = False
        point = rule.get("point")
        if isinstance(point, str):
            got = truth_at_path(truth, point)
            if not isinstance(got, (list, tuple)) or len(got) < 2:
                return False, {"kind": kind, "reason": f"no_truth_at:{point}"}
            x, y, in_sample = float(got[0]), float(got[1]), True
        elif point is not None:
            x, y = float(point[0]), float(point[1])
        if x is None or y is None:
            return False, {"kind": kind, "reason": "no_position"}
        after = rule.get("after_event")
        t_after = 0.0
        if after:
            times = [float(e.get("sim_s", 0.0)) for e in events or [] if e.get("kind") == after]
            if not times:
                return False, {"kind": kind, "reason": f"no_{after}"}
            t_after = max(times)
        later = [e for e in events or [] if float(e.get("sim_s", 0.0)) >= t_after]
        cover = frame_covering_sample if in_sample else frame_covering
        fr = cover(later, float(x), float(y), rule)
        return fr is not None, {"kind": kind, "frame_idx": (fr or {}).get("idx"),
                                "after_sim_s": t_after}
    if kind in ("spectra_near_scatterer", "sts_at_corral_centre", "zspec_pair"):
        return _check_spectra(kind, rule, events=events, truth=truth)
    if kind == "events":
        want = str(rule.get("event", "adatom_hop"))
        cause = rule.get("cause")
        got = [e for e in events or [] if e.get("kind") == want
               and (cause is None or e.get("cause") == cause)]
        return len(got) >= int(rule.get("min_n", 1)), {"kind": kind, "n": len(got)}
    return False, {"kind": kind, "reason": "unknown_evidence_kind"}


def _covers_window(ev: dict, lo, hi) -> bool:
    if lo is None and hi is None:
        return True
    v0, v1 = sorted((float(ev.get("v0", 0.0)), float(ev.get("v1", 0.0))))
    return (lo is None or v0 <= float(lo)) and (hi is None or v1 >= float(hi))


def _check_spectra(kind: str, rule: dict, *, events, truth: dict) -> tuple[bool, dict]:
    if kind == "zspec_pair":
        zs = _spectra(events, "zspec")
        atom_rule = rule.get("adatom") or {}
        bg_rule = rule.get("background") or {}
        on_atom = [e for e in zs
                   if e.get("near_kind") == atom_rule.get("near_kind", "adatom")
                   and float(e.get("near_nm", 1e9)) <= float(atom_rule.get("near_nm_max", 0.15))
                   and not e.get("jump", False) and not e.get("contact", False)
                   and float(e.get("df_span_beyond_min_pm", 0.0))
                   >= float(atom_rule.get("df_span_beyond_min_pm_min", 0.0))]
        bg = [e for e in zs
              if (e.get("near_kind") in ("none", "step")
                  or float(e.get("near_nm", 0.0)) > float(bg_rule.get("near_nm_min", 2.0)))
              and not e.get("contact", False)]
        ok = len(on_atom) >= int(atom_rule.get("min_n", 1)) and len(bg) >= int(bg_rule.get("min_n", 1))
        return ok, {"kind": kind, "n_on_atom": len(on_atom), "n_background": len(bg)}
    got = _spectra(events, "sts")
    got = [e for e in got if not e.get("jump", False)
           and int(e.get("n", 0)) >= int(rule.get("min_points", 0))
           and _covers_window(e, rule.get("v_lo_max"), rule.get("v_hi_min"))]
    if kind == "spectra_near_scatterer":
        near = float(rule.get("near_nm", 8.0))
        got = [e for e in got if float(e.get("near_nm", 1e9)) <= near]
    else:
        cmax = float(rule.get("max_centre_dist_nm", 1.0))
        got = [e for e in got if e.get("corral_centre_dist_nm") is not None
               and float(e["corral_centre_dist_nm"]) <= cmax]
    after = rule.get("after_event")
    if after:
        times = [float(e.get("sim_s", 0.0)) for e in events or [] if e.get("kind") == after]
        if not times:
            return False, {"kind": kind, "reason": f"no_{after}"}
        got = [e for e in got if float(e.get("sim_s", 0.0)) >= max(times)]
    positions = {(round(float(e.get("sx_m", 0.0)) * 1e9 / 0.5),
                  round(float(e.get("sy_m", 0.0)) * 1e9 / 0.5)) for e in got}
    ok = (len(got) >= int(rule.get("min_n", 1))
          and len(positions) >= int(rule.get("min_positions", 1)))
    return ok, {"kind": kind, "n_spectra": len(got), "n_positions": len(positions)}
