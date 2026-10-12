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
sample frame, and hours of drift separate the two. Every saved frame records the drift each of
its rows was imaged under, so a reported position is mapped through a frame — the agent is
never asked to know about drift, and a right answer at a right place is never failed for it.

Version-1 rules (the rule card an agent is shown, :mod:`stmbench.trackB.rules`): a report
names its data. A position is read in the frame the report names, and the evidence rule is
checked within the frames and spectra the report cites, instead of anywhere in the episode.
That removes the guess of *which* covering frame a position belongs to, and ties the evidence
to the result it supports.
"""
from __future__ import annotations

import math
import re
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


def _row_offset_nm(frame: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Drift offset (nm) of the acquisition row in which a scan-frame point was imaged.

    The renderer adds one drift/creep offset per acquisition row (``row_offsets_nm``), so a
    long frame maps different rows to different places on the sample: across a 150 s frame
    the drift moves 1-3 nm. Frames saved before per-row records existed carry only the offset
    at their recorded start (``drift_m``), which is used for them."""
    rows = frame.get("row_offsets_nm")
    if not rows:
        dx, dy = (frame.get("drift_m") or [0.0, 0.0])[:2]
        return float(dx) * 1e9, float(dy) * 1e9
    cx, cy = float(frame.get("cx_m", 0.0)) * 1e9, float(frame.get("cy_m", 0.0)) * 1e9
    h = float(frame.get("h_m", 0.0)) * 1e9
    th = math.radians(float(frame.get("angle_deg", 0.0) or 0.0))
    ny = len(rows)
    v = -math.sin(th) * (x_nm - cx) + math.cos(th) * (y_nm - cy)
    j = (h / 2 - v) / h * ny - 0.5 if h > 0 else 0.0         # geometric row, 0 = top (high v)
    k = (ny - 1 - j) if frame.get("scan_dir") == "up" else j   # acquisition row
    k = min(max(k, 0.0), ny - 1.0)
    k0 = int(math.floor(k))
    k1 = min(k0 + 1, ny - 1)
    f = k - k0
    return (float(rows[k0][0]) * (1 - f) + float(rows[k1][0]) * f,
            float(rows[k0][1]) * (1 - f) + float(rows[k1][1]) * f)


def to_sample_frame(frame: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Scan-frame nm -> sample-frame nm, through the drift of the row the point was imaged in."""
    ox, oy = _row_offset_nm(frame, x_nm, y_nm)
    return x_nm + ox, y_nm + oy


def to_scan_frame(frame: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Sample-frame nm -> the scan-frame point of ``frame`` that imaged it (the inverse of
    :func:`to_sample_frame`; the offset varies slowly with the row, so a few fixed-point
    steps converge)."""
    px, py = x_nm, y_nm
    for _ in range(5):
        ox, oy = _row_offset_nm(frame, px, py)
        px, py = x_nm - ox, y_nm - oy
    return px, py


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
        px, py = to_scan_frame(fr, x_nm, y_nm)
        if frame_covering([fr], px, py, rule) is not None:
            best = fr
    return best


# -- citations: the data a report names ---------------------------------------
def _basename(path) -> str:
    return re.split(r"[\\/]", str(path or ""))[-1].strip().lower()


def resolve_citation(events, ref) -> dict | None:
    """The event a cited data id names, or ``None``.

    A saved frame is named by its file name (``sim001.sxm``) or ``frame:N``; a spectrum by
    its ``.dat`` file name or ``sts:N`` / ``zspec:N`` (N = the controller's running index)."""
    events = list(events or [])
    ref = str(ref or "").strip()
    low = ref.lower()
    for prefix, kind in (("frame:", "scan_saved"), ("sts:", "sts"), ("zspec:", "zspec")):
        if low.startswith(prefix):
            try:
                n = int(low[len(prefix):])
            except ValueError:
                return None
            hits = [e for e in events if e.get("kind") == kind and int(e.get("idx", -1)) == n]
            return hits[-1] if hits else None
    name = _basename(ref)
    if not name:
        return None
    for e in reversed(events):
        if _basename(e.get("path")) != name:
            continue
        if e.get("kind") == "scan_saved":
            return e
        if e.get("kind") == "dat_saved":
            for key, kind in (("sts_idx", "sts"), ("zspec_idx", "zspec")):
                if e.get(key) is not None:
                    hits = [x for x in events if x.get("kind") == kind and x.get("idx") == e[key]]
                    return hits[-1] if hits else None
    return None


def _after_last(events, frame: dict, event_kind: str) -> bool:
    """Was ``frame`` saved at or after the last ``event_kind`` event? (no such event: no)"""
    times = [float(e.get("sim_s", 0.0)) for e in events or [] if e.get("kind") == event_kind]
    return bool(times) and float(frame.get("sim_s", 0.0)) >= max(times)


def placing_frame(events, report: dict, rule: dict) -> tuple[dict | None, str | None]:
    """The frame a reported position is read in under the version-1 rules: the one the report
    names in ``frame``. It must be a saved frame that satisfies the claim's evidence rule
    (size, pixel, completeness, and the ``after_event`` order) and contain the position."""
    ref = report.get("frame")
    if not ref:
        return None, "no_reference_frame"
    fr = resolve_citation(events, ref)
    if fr is None or fr.get("kind") != "scan_saved":
        return None, "reference_frame_not_found"
    x, y = report.get("x_nm"), report.get("y_nm")
    if x is None or y is None:
        return None, "no_position"
    if not _frame_ok(fr, rule) or frame_covering([fr], float(x), float(y), rule) is None:
        return None, "reference_frame_does_not_cover_or_fit"
    after = rule.get("after_event")
    if after and not _after_last(events, fr, str(after)):
        return None, f"reference_frame_not_after_{after}"
    return fr, None


def _spectra(events, kind: str = "sts") -> list[dict]:
    return [e for e in events or [] if e.get("kind") == kind
            and e.get("delivered", True) is not False]


def check_evidence(rule: dict, *, events, report: dict, truth: dict,
                   cited: list[dict] | None = None) -> tuple[bool, dict]:
    """Check whether the episode acquired the measurement required by the claim.

    ``cited`` (version-1 rules): the events of the data the report names. The rule is then
    checked within that data only, and a reported position must sit in the frame the report
    names. ``None`` keeps the earlier rule, which looks through the whole episode."""
    kind = str(rule.get("kind", "frame"))
    pool = list(events or []) if cited is None else list(cited)
    tag = {} if cited is None else {"cited": len(cited)}
    if kind == "frame":
        got = [f for f in _frames(pool) if _frame_ok(f, rule)]
        return bool(got), {"kind": kind, "n_frames": len(got), **tag}
    if kind == "frame_covers":
        x, y = report.get("x_nm"), report.get("y_nm")
        if x is None or y is None:
            return False, {"kind": kind, "reason": "no_position", **tag}
        if cited is not None:
            fr, why = placing_frame(events, report, rule)
            return fr is not None, {"kind": kind, "frame_idx": (fr or {}).get("idx"),
                                    "covered": fr is not None, **({"reason": why} if why else {}), **tag}
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
            return False, {"kind": kind, "reason": "no_position", **tag}
        if cited is not None and point is None:
            fr, why = placing_frame(events, report, rule)
            return fr is not None, {"kind": kind, "frame_idx": (fr or {}).get("idx"),
                                    **({"reason": why} if why else {}), **tag}
        after = rule.get("after_event")
        t_after = 0.0
        if after:
            times = [float(e.get("sim_s", 0.0)) for e in events or [] if e.get("kind") == after]
            if not times:
                return False, {"kind": kind, "reason": f"no_{after}"}
            t_after = max(times)
        later = [e for e in pool if float(e.get("sim_s", 0.0)) >= t_after]
        cover = frame_covering_sample if in_sample else frame_covering
        fr = cover(later, float(x), float(y), rule)
        return fr is not None, {"kind": kind, "frame_idx": (fr or {}).get("idx"),
                                "after_sim_s": t_after, **tag}
    if kind in ("spectra_near_scatterer", "sts_at_corral_centre", "zspec_pair"):
        ok, info = _check_spectra(kind, rule, events=events, truth=truth, pool=pool)
        return ok, {**info, **tag}
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


def _check_spectra(kind: str, rule: dict, *, events, truth: dict,
                   pool: list[dict] | None = None) -> tuple[bool, dict]:
    """``pool``: the spectra to look through (the cited ones, or the whole episode); the
    ``after_event`` order is always read from the whole episode."""
    pool = list(events or []) if pool is None else pool
    if kind == "zspec_pair":
        zs = _spectra(pool, "zspec")
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
    got = _spectra(pool, "sts")
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
