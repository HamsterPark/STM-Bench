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
    # a constant-current topograph: the tip tunnelling, the Z loop on. Withdrawn, or with the
    # loop off, the frame is a flat sheet of noise that passes every test of its geometry
    # (ledgers from before the state was recorded carry neither key)
    if frame.get("withdrawn") or frame.get("feedback") is False:
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
        c, s = math.cos(ang), math.sin(ang)
        u, v = c * dx - s * dy, s * dx + c * dy
        if abs(u) <= w / 2 and abs(v) <= h / 2:
            best = fr
    return best


def _apex_nm(frame: dict, k: float) -> tuple[float, float]:
    """The lead apex's lateral offset (nm) at acquisition row ``k`` (``apex_runs_nm``)."""
    dx = dy = 0.0
    for row, ax, ay in frame.get("apex_runs_nm") or ():
        if row > k + 0.5:
            break
        dx, dy = float(ax), float(ay)
    return dx, dy


def _row_offset_nm(frame: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Offset (nm) from a scan-frame point of a frame to the place on the sample it imaged: the
    drift of the acquisition row it was imaged in, plus the offset of the apex that imaged it.

    The renderer adds one drift/creep offset per acquisition row (``row_offsets_nm``), so a
    long frame maps different rows to different places on the sample: across a 150 s frame
    the drift moves 1-3 nm. The tip images through the apex nearest the surface, so every
    feature shows displaced by that apex's lateral offset — and the drags are, too, so an
    operator who moves an atom by what the images show moves it right. A tip change can hand
    the imaging to another apex nanometres off (P4 seed 15 of the baseline: 2.5 nm, after a
    new apex took over from a receded one), which no measurement on the sample can see;
    ``apex_runs_nm`` records it. Frames saved before per-row records existed carry only the
    drift at their recorded start (``drift_m``), which is used for them."""
    rows = frame.get("row_offsets_nm")
    if not rows:
        dx, dy = (frame.get("drift_m") or [0.0, 0.0])[:2]
        return float(dx) * 1e9, float(dy) * 1e9
    cx, cy = float(frame.get("cx_m", 0.0)) * 1e9, float(frame.get("cy_m", 0.0)) * 1e9
    h = float(frame.get("h_m", 0.0)) * 1e9
    th = math.radians(float(frame.get("angle_deg", 0.0) or 0.0))
    ny = len(rows)
    v = math.sin(th) * (x_nm - cx) + math.cos(th) * (y_nm - cy)
    j = (h / 2 - v) / h * ny - 0.5 if h > 0 else 0.0         # geometric row, 0 = top (high v)
    k = (ny - 1 - j) if frame.get("scan_dir") == "up" else j   # acquisition row
    k = min(max(k, 0.0), ny - 1.0)
    k0 = int(math.floor(k))
    k1 = min(k0 + 1, ny - 1)
    f = k - k0
    ax, ay = _apex_nm(frame, k)
    return (float(rows[k0][0]) * (1 - f) + float(rows[k1][0]) * f + ax,
            float(rows[k0][1]) * (1 - f) + float(rows[k1][1]) * f + ay)


def to_sample_frame(frame: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Scan-frame nm -> sample-frame nm, through the drift of the row the point was imaged in
    and the apex that imaged it."""
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
    its ``.dat`` file name or ``sts:N`` / ``zspec:N`` (N = the controller's running index); a
    move trace by the ``.dat`` file name ``move_tip`` returned or ``move:N`` (N = the trace's
    index, ``dat_saved.move_idx``), and it resolves to its ``dat_saved`` event."""
    events = list(events or [])
    ref = str(ref or "").strip()
    low = ref.lower()
    if low.startswith("move:"):
        try:
            n = int(low[len("move:"):])
        except ValueError:
            return None
        hits = [e for e in events if e.get("kind") == "dat_saved" and e.get("move_idx") is not None
                and int(e["move_idx"]) == n]
        return hits[-1] if hits else None
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
            if e.get("move_idx") is not None:
                return e
            for key, kind in (("sts_idx", "sts"), ("zspec_idx", "zspec")):
                if e.get(key) is not None:
                    hits = [x for x in events if x.get("kind") == kind and x.get("idx") == e[key]]
                    return hits[-1] if hits else None
    return None


def _acquired_s(frame: dict) -> float:
    """When a frame's acquisition began. A frame shows the surface as it was then (it is
    rendered at its start), and it can be saved, or saved again, any time later; ledgers from
    before the start was recorded fall back to the save time."""
    t = frame.get("t_start_sim")
    return float(frame.get("sim_s", 0.0) if t is None else t)


def _event_s(e: dict) -> float:
    """When an event happened. An atom moved by imaging carries the time of the row the tip
    was on (``row_sim_s``): the frame's events are stamped with its start."""
    t = e.get("row_sim_s")
    return float(e.get("sim_s", 0.0) if t is None else t)


def _last_event_s(events, event_kind: str) -> float | None:
    times = [_event_s(e) for e in events or [] if e.get("kind") == event_kind]
    return max(times) if times else None


def _after_last(events, frame: dict, event_kind: str) -> bool:
    """Was ``frame`` acquired after the last ``event_kind`` event: did its scan start later?
    (No such event: no.) A frame taken before an atom moved and saved after it shows where the
    atom was, as does one during which it moved."""
    t = _last_event_s(events, event_kind)
    return t is not None and _acquired_s(frame) > t


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


#: two resistances of a drag series are one value when the larger is within this factor of the
#: smaller (a series of drags at "the same" resistance is not a series over resistances)
DISTINCT_RESISTANCE_FACTOR = 1.02


def select_drags(rule: dict, truth: dict, cited: list[dict] | None) -> dict:
    """Which of the cited move traces count for a ``drag_series`` claim, and the series they form.

    One function for the evidence rule and for the truth (:func:`drag_series_threshold`), so the
    two can never disagree about which drags count. A cited drag **qualifies** when its record
    (``truth["drags"]["records"]``, the one whose ``index`` is the trace's ``move_idx``) shows the
    loop on, the tip in tunnelling and in range, no atom already on it, a finite resistance, an atom
    held by an apex that started within ``start_within_nm`` of it, and length and speed within
    ``protocol_rel`` of the claim's ``protocol`` (``rule["protocol"]``, copied from the claim).
    Citations that are not move traces are ignored. ``series`` is the qualifying drags of the atom
    with the most of them (ties: the lowest atom id); the evidence fails when there are two atoms."""
    drags = (truth or {}).get("drags") or {}
    records = {int(r["index"]): r for r in drags.get("records") or [] if r.get("index") is not None}
    proto = rule.get("protocol") or {}
    rel = float(rule.get("protocol_rel", 0.0) or 0.0)
    within = float(rule.get("start_within_nm", 0.0) or 0.0)
    eps = 1e-9
    qualifying: list[dict] = []
    rejected: list[dict] = []
    seen: set[int] = set()
    for ev in cited or []:
        if ev.get("kind") != "dat_saved" or ev.get("move_idx") is None:
            continue
        idx = int(ev["move_idx"])
        if idx in seen:
            continue
        seen.add(idx)
        rec = records.get(idx)
        why = None
        if rec is None:
            why = "no_record"
        elif not rec.get("feedback"):
            why = "feedback_off"
        elif rec.get("withdrawn"):
            why = "withdrawn"
        elif rec.get("over_range"):
            why = "over_range"
        elif rec.get("carrying"):
            why = "carrying"
        elif rec.get("r_kohm") is None or not math.isfinite(float(rec["r_kohm"])):
            why = "no_resistance"
        elif rec.get("atom_id") is None:
            why = "no_atom"
        elif rec.get("start_dist_nm") is None or float(rec["start_dist_nm"]) > within + eps:
            why = "start_too_far"
        else:
            for key, name in (("length_nm", "length_off"), ("speed_nm_s", "speed_off")):
                want = proto.get(key)
                if want is not None and abs(float(rec.get(key, 0.0)) - float(want)) > rel * abs(float(want)) + eps:
                    why = name
                    break
        if why:
            rejected.append({"index": idx, "reason": why})
        else:
            qualifying.append(rec)
    by_atom: dict[int, list[dict]] = {}
    for rec in qualifying:
        by_atom.setdefault(int(rec["atom_id"]), []).append(rec)
    atom_id = min(by_atom, key=lambda a: (-len(by_atom[a]), a)) if by_atom else None
    return {"qualifying": qualifying, "rejected": rejected, "atom_id": atom_id,
            "series": by_atom.get(atom_id, []), "atoms": sorted(by_atom)}


def distinct_resistances(values) -> int:
    """How many distinct resistances: sorted, a new value starts where one exceeds the first of the
    current group by more than 2 %."""
    n, first = 0, None
    for v in sorted(float(v) for v in values):
        if first is None or v > first * DISTINCT_RESISTANCE_FACTOR:
            n, first = n + 1, v
    return n


def drag_series_threshold(rule: dict, truth: dict, cited: list[dict] | None) -> tuple[float | None, dict]:
    """The truth of a ``drag_threshold`` claim, kΩ: the geometric mean over the series' drags of the
    R½ of the atom each held under that drag's conditions (``record_threshold``). Drags whose
    threshold does not exist are left out; with none left the truth is ``None`` and says why."""
    from stmsim.physics.drag_threshold import record_threshold

    sel = select_drags(rule, truth, cited)
    info = {"atom_id": sel["atom_id"], "n_qualifying": len(sel["series"])}
    proto = rule.get("protocol") or {}
    lattice = ((truth or {}).get("drags") or {}).get("lattice")
    logs = []
    for rec in sel["series"]:
        r = record_threshold(lattice, rec, length_m=float(proto.get("length_nm", 1.0)) * 1e-9,
                             speed_m_s=float(proto.get("speed_nm_s", 0.1)) * 1e-9)
        if r is not None and math.isfinite(r) and r > 0:
            logs.append(math.log(r / 1e3))
    info["n_thresholds"] = len(logs)
    if not logs:
        info["reason"] = "no_qualifying_drag" if not sel["series"] else "no_threshold_for_the_drags"
        return None, info
    return math.exp(sum(logs) / len(logs)), info


def _check_drag_series(rule: dict, *, report: dict, truth: dict, cited: list[dict] | None,
                       peer_atom: int | None, kind: str) -> tuple[bool, dict]:
    sel = select_drags(rule, truth, cited)
    series = sel["series"]
    rs = [float(r["r_kohm"]) for r in series]
    info: dict = {"kind": kind, "atom_id": sel["atom_id"], "n_qualifying": len(sel["qualifying"]),
                  "rejected": sel["rejected"],
                  "resistances_kohm": [round(r, 3) for r in sorted(rs)]}
    why = []
    if len(sel["atoms"]) > 1:
        why.append("more_than_one_atom")
    if len(series) < int(rule.get("min_n", 1)):
        why.append("too_few_drags")
    n_res = distinct_resistances(rs)
    info["n_resistances"] = n_res
    if n_res < int(rule.get("min_resistances", 1)):
        why.append("too_few_resistances")
    value = report.get("value")
    carried = failed = False
    if value is not None:
        v = float(value)
        carried = any(r.get("followed") and float(r["r_kohm"]) <= v for r in series)
        failed = any((not r.get("followed")) and float(r["r_kohm"]) >= v for r in series)
        info["bracket"] = {"carried_at_or_below": carried, "not_carried_at_or_above": failed}
    if not carried:
        why.append("no_carry_below_value")
    if not failed:
        why.append("no_failure_above_value")
    peer_id = rule.get("distinct_atom_from")
    if peer_id and peer_atom is not None and sel["atom_id"] is not None and sel["atom_id"] == peer_atom:
        why.append("same_atom_as")
        info["same_atom_as"] = str(peer_id)
    if why:
        info["reason"] = why[0]
        info["reasons"] = why
    return not why, info


def check_evidence(rule: dict, *, events, report: dict, truth: dict,
                   cited: list[dict] | None = None, peer_atom: int | None = None) -> tuple[bool, dict]:
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
        later = _frames(pool)
        t_after = 0.0
        if after:
            t_last = _last_event_s(events, str(after))
            if t_last is None:
                return False, {"kind": kind, "reason": f"no_{after}"}
            t_after = t_last
            later = [e for e in later if _acquired_s(e) > t_after]
        cover = frame_covering_sample if in_sample else frame_covering
        fr = cover(later, float(x), float(y), rule)
        return fr is not None, {"kind": kind, "frame_idx": (fr or {}).get("idx"),
                                "after_sim_s": t_after, **tag}
    if kind in ("spectra_near_scatterer", "sts_at_corral_centre", "zspec_pair"):
        ok, info = _check_spectra(kind, rule, events=events, truth=truth, pool=pool)
        return ok, {**info, **tag}
    if kind == "didv_maps":
        maps = [f for f in _frames(pool) if f.get("didv_map") and _frame_ok(f, rule)
                and f.get("near_nm") is not None and float(f["near_nm"]) <= float(rule.get("near_nm", 10.0))]
        energies = sorted({round(float(f.get("bias_v", 0.0)), 2) for f in maps})
        lo, hi = rule.get("v_lo_max"), rule.get("v_hi_min")
        span = bool(energies) and (lo is None or energies[0] <= float(lo)) and (hi is None or energies[-1] >= float(hi))
        ok = len(maps) >= int(rule.get("min_n", 1)) and len(energies) >= int(rule.get("min_energies", 1)) and span
        return ok, {"kind": kind, "n_maps": len(maps), "energies_v": energies, **tag}
    if kind == "drag_series":
        # within the cited move traces only: a series is the drags the report names; with no
        # citations at all (the pre-card rules) there is nothing to check it against
        return _check_drag_series(rule, report=report, truth=truth, cited=pool if cited is not None else [],
                                  peer_atom=peer_atom, kind=kind)
    if kind == "any_of":
        parts = [check_evidence(r, events=events, report=report, truth=truth, cited=cited,
                                peer_atom=peer_atom)
                 for r in rule.get("rules") or []]
        return any(ok for ok, _ in parts), {"kind": kind, "parts": [info for _, info in parts], **tag}
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
        t_last = _last_event_s(events, str(after))
        if t_last is None:
            return False, {"kind": kind, "reason": f"no_{after}"}
        got = [e for e in got if float(e.get("sim_s", 0.0)) >= t_last]
    positions = {(round(float(e.get("sx_m", 0.0)) * 1e9 / 0.5),
                  round(float(e.get("sy_m", 0.0)) * 1e9 / 0.5)) for e in got}
    ok = (len(got) >= int(rule.get("min_n", 1))
          and len(positions) >= int(rule.get("min_positions", 1)))
    return ok, {"kind": kind, "n_spectra": len(got), "n_positions": len(positions)}
