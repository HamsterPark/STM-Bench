"""An episode's ledger as one timeline on the instrument clock (sim seconds).

Three streams meet here:

* the model's steps (``driver.json``): each tool call with the instrument time it started
  and ended — stamped by the harness since ``sim_stamped``; for older ledgers rebuilt from
  wall times (:func:`calibrate_clock`), because the instrument clock stands still while the
  model thinks and only runs inside tool calls;
* the world (``events.json``): frames, atoms moving, crashes, pulses — already in sim time;
* the tip (``tip_timeline.json``): its state at the start and after every change. Older
  ledgers have only the start and the end; the changes in between are then inferred from
  the world's pulse / poke / crash events and flagged ``reconstructed``.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import lay

#: two seconds of wall clock either side, when fitting a world event into a tool call
CAL_TOL_WALL_S = 2.0
#: tip fields the page needs (the rest of the snapshot stays in the ledger)
TIP_FIELDS = ("n_apex", "multi", "radius_nm", "apex_sigma_nm", "flicker_dz_pm", "metastable",
              "carried", "dead", "phi_ev", "ldos")
#: consecutive hops of one atom closer than this (sim s) are one drag
DRAG_GAP_S = 30.0


@dataclass
class Ledger:
    run_dir: Path
    episode: dict
    driver: dict | None
    events: list[dict]
    tip_timeline: dict | None

    @classmethod
    def load(cls, run_dir: str | Path) -> "Ledger":
        d = Path(run_dir)
        if not (d / "episode.json").is_file():
            raise FileNotFoundError(f"no episode.json in {d}")

        def _read(name: str):
            p = d / name
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

        return cls(run_dir=d, episode=_read("episode.json"), driver=_read("driver.json"),
                   events=list(_read("events.json") or []), tip_timeline=_read("tip_timeline.json"))

    @property
    def session_dir(self) -> Path:
        return self.run_dir / "session"


# ── the model's clock ───────────────────────────────────────────────────────
def _tool_intervals(driver_events: list[dict]) -> list[dict]:
    """tool_start / tool_end pairs, in start order: ``{start, end}`` (the records)."""
    open_: dict[str, dict] = {}
    out: list[dict] = []
    for e in driver_events:
        k = e.get("kind")
        if k == "tool_start":
            rec = {"start": e, "end": None}
            open_[str(e.get("tool_call_id") or len(out))] = rec
            out.append(rec)
        elif k == "tool_end":
            rec = open_.pop(str(e.get("tool_call_id") or ""), None)
            if rec is not None:
                rec["end"] = e
    for rec in out:
        if rec["end"] is None:                       # the episode ended inside the call
            rec["end"] = dict(rec["start"])
    return out


def calibrate_clock(intervals: list[dict], world_events: list[dict], sim_start: float,
                    time_scale: float) -> tuple[Any, dict]:
    """``sim_at(epoch)`` for a ledger whose driver events carry wall times only.

    The instrument clock runs only while a tool runs, so instrument time at a wall instant
    is ``sim_start + time_scale × (tool time so far) + shift``. The shift (start-up, per-step
    overheads) is fitted so the world's own events — which carry exact instrument times —
    fall inside the tool calls: the Δ region inside the most events, and its midpoint."""
    ts = float(time_scale or 1.0)
    starts = [float(r["start"].get("t", 0.0)) for r in intervals]
    ends = [max(float(r["end"].get("t", 0.0)), s) for r, s in zip(intervals, starts)]
    cum = []
    c = 0.0
    for s, e in zip(starts, ends):
        cum.append(c)
        c += e - s

    def running(t: float) -> float:
        k = bisect.bisect_right(starts, t) - 1
        if k < 0:
            return 0.0
        return cum[k] + min(max(t - starts[k], 0.0), ends[k] - starts[k])

    p0 = [sim_start + ts * cum[k] for k in range(len(starts))]
    p1 = [sim_start + ts * (cum[k] + ends[k] - starts[k]) for k in range(len(starts))]
    tol = CAL_TOL_WALL_S * ts
    sims = [float(e["sim_s"]) for e in world_events if isinstance(e.get("sim_s"), (int, float))]
    shift, covered = 0.0, 0
    if sims and p0:
        # per event, the shifts that put it inside some tool call (merged, so an event is
        # counted once); then the shift region inside the most events, and its widest piece
        edges: list[tuple[float, int]] = []
        for s in sims:
            spans = sorted((s - b - tol, s - a + tol) for a, b in zip(p0, p1))
            merged: list[list[float]] = []
            for lo, hi in spans:
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in merged:
                edges += [(lo, +1), (hi, -1)]
        edges.sort(key=lambda x: (x[0], -x[1]))
        best, depth, seg = 0, 0, None
        for (x, d), nxt in zip(edges, edges[1:] + [(math.inf, 0)]):
            depth += d
            if nxt[0] <= x or not math.isfinite(nxt[0]):
                continue
            if depth > best or (depth == best and seg is not None and nxt[0] - x > seg[1] - seg[0]):
                best, seg = depth, (x, nxt[0])
        if seg is not None:
            shift = 0.5 * (seg[0] + seg[1])
        covered = sum(1 for s in sims if any(a + shift - tol <= s <= b + shift + tol for a, b in zip(p0, p1)))

    def sim_at(t: float) -> float:
        return sim_start + shift + ts * running(float(t))

    return sim_at, {"source": "calibrated", "shift_s": shift, "events": len(sims), "covered": covered}


def _stamped(rec: dict) -> float | None:
    v = rec.get("sim_s")
    return float(v) if isinstance(v, (int, float)) else None


# ── steps ───────────────────────────────────────────────────────────────────
def build_steps(driver: dict | None, world_events: list[dict], sim_start: float, sim_end: float,
                time_scale: float) -> tuple[list[dict], list[dict], dict]:
    """(steps, messages, clock info). A step is one tool call, with its caption."""
    evs = list((driver or {}).get("events") or [])
    intervals = _tool_intervals(evs)
    if not intervals and not evs:
        return [], [], {"source": "none"}
    stamped = bool(intervals) and all(_stamped(r["start"]) is not None and _stamped(r["end"]) is not None
                                      for r in intervals)
    if stamped:
        clock = {"source": "stamped"}
        sim_at = None
    else:
        sim_at, clock = calibrate_clock(intervals, world_events, sim_start, time_scale)

    def at(rec: dict) -> float:
        v = _stamped(rec) if stamped else None
        if v is None:
            v = sim_at(float(rec.get("t", 0.0))) if sim_at is not None else sim_start
        return min(max(v, sim_start), sim_end)

    steps: list[dict] = []
    prev_end_t = None
    last = sim_start
    first_t = min((float(e.get("t")) for e in evs if isinstance(e.get("t"), (int, float))), default=None)
    for k, r in enumerate(intervals):
        s, e = r["start"], r["end"]
        name = str(s.get("name") or "")
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        t0 = float(s.get("t", 0.0))
        t1 = max(float(e.get("t", t0)), t0)
        sim0 = max(at(s), last)
        sim1 = max(at(e), sim0)
        last = sim1
        think = t0 - (prev_end_t if prev_end_t is not None else (first_t if first_t is not None else t0))
        prev_end_t = t1
        stub = lay.is_stub_reply(e.get("preview"))
        steps.append({"i": k, "name": name, "caption": lay.tool_caption(name, args),
                      "args": _short_args(args), "result": "" if stub else lay.result_text(e.get("preview")),
                      "ok": bool(e.get("ok", True)), "stub": stub, "sim0": sim0, "sim1": sim1,
                      "think_s": max(0.0, think), "wall_s": t1 - t0})
    messages = []
    for e in evs:
        if e.get("kind") == "message" and str(e.get("role", "assistant")) == "assistant" and str(e.get("text") or "").strip():
            messages.append({"sim": at(e), "text": _clean_text(str(e["text"]))})
    return steps, messages, clock


def _short_args(args: dict) -> dict:
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and (len(v) > 60 or lay._PATH_RE.search(v)):
            continue                                # paths and blobs mean nothing on the page
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
    return out


def _clean_text(s: str) -> str:
    for mark in ("[DONE]", "[ABORT]"):
        s = s.replace(mark, "")
    return s.strip()


# ── frames ──────────────────────────────────────────────────────────────────
def build_frames(world_events: list[dict]) -> list[dict]:
    """Every scan the instrument ran, with the file it was saved to (if any), the rows it
    got and how long a row takes — enough to redraw it row by row."""
    frames: list[dict] = []
    for e in world_events:
        k = e.get("kind")
        if k == "scan_start":
            line = float(e.get("line_s") or 0.0)
            per_row = float(e.get("per_row_s") or 2.0 * line)
            frames.append({"sim0": float(e["sim_s"]), "per_row_s": per_row, "w_nm": e.get("w_nm"),
                           "px": e.get("px"), "scan_dir": e.get("scan_dir"),
                           "tip_events": e.get("tip_events"), "file": None, "rows": None})
        elif k == "scan_stop" and frames and frames[-1]["file"] is None and frames[-1]["rows"] is None:
            frames[-1]["rows"] = int(e.get("rows") or 0)             # stopped: this many rows exist
        elif k == "scan_saved":
            t0 = float(e.get("t_start_sim", e.get("sim_s", 0.0)))
            cand = [f for f in frames if f["file"] is None and abs(f["sim0"] - t0) < 1.0]
            fr = min(cand, key=lambda f: abs(f["sim0"] - t0)) if cand else None
            if fr is None:                            # saved without a start we saw
                fr = {"sim0": t0, "per_row_s": None, "w_nm": (e.get("w_m") or 0) * 1e9, "px": e.get("nx"),
                      "scan_dir": None, "tip_events": None}
                frames.append(fr)
            drift = e.get("drift_m") or [0.0, 0.0]
            fr.update({"file": Path(str(e.get("path") or "")).name or None, "idx": e.get("idx"),
                       "rows": int(e.get("rows") or 0), "complete": bool(e.get("complete")),
                       "sim_saved": float(e.get("sim_s", t0)),
                       "geom": {"cx_nm": float(e.get("cx_m") or 0) * 1e9, "cy_nm": float(e.get("cy_m") or 0) * 1e9,
                                "w_nm": float(e.get("w_m") or 0) * 1e9, "h_nm": float(e.get("h_m") or 0) * 1e9,
                                "angle_deg": float(e.get("angle_deg") or 0.0),
                                "nx": int(e.get("nx") or 0), "ny": int(e.get("ny") or 0)},
                       "drift_nm": [float(drift[0]) * 1e9, float(drift[1]) * 1e9],
                       "bias_v": e.get("bias_v"), "setpoint_a": e.get("setpoint_a")})
            if not fr.get("per_row_s") and fr["rows"]:
                fr["per_row_s"] = (fr["sim_saved"] - fr["sim0"]) / fr["rows"]
    for i, fr in enumerate(frames):
        nxt = frames[i + 1]["sim0"] if i + 1 < len(frames) else math.inf
        rows = fr.get("rows")
        if rows is None:                              # never saved: it ran until the next start
            ny = int(fr.get("px") or 0)
            rows = ny if not math.isfinite(nxt) else int(max(0.0, nxt - fr["sim0"]) // max(fr["per_row_s"] or 1, 1e-9))
            rows = min(rows, ny) if ny else rows
        fr["rows_done"] = int(rows)
        fr["sim1"] = min(fr["sim0"] + fr["rows_done"] * float(fr.get("per_row_s") or 0.0), nxt)
    return frames


# ── the tip ─────────────────────────────────────────────────────────────────
def _reduce(state: dict | None) -> dict:
    s = state or {}
    return {k: s.get(k) for k in TIP_FIELDS if k in s}


def _synth_tip(prev: dict, ev: dict) -> dict:
    """An old ledger's tip after a pulse / poke / crash, from the event's outcome."""
    s = dict(prev)
    kind, out = ev.get("kind"), str(ev.get("outcome") or "")
    if kind == "crash":
        n = int(ev.get("n_apex") or 2)
        s.update(n_apex=n, multi=n > 1, flicker_dz_pm=float(ev.get("flicker_dz_pm") or 100.0), metastable=True,
                 radius_nm=float(s.get("radius_nm") or 1.0) * 3.0 * float(ev.get("severity") or 1.0))
    elif kind == "pulse":
        if out == "reshaped_better":
            s.update(n_apex=1, multi=False, flicker_dz_pm=0.0, metastable=True,
                     radius_nm=max(0.3, float(s.get("radius_nm") or 1.0) * 0.55))
        elif out == "reshaped_multi":
            s.update(n_apex=2, multi=True, flicker_dz_pm=0.0, metastable=True)
        elif out == "reshaped_blunt":
            s.update(n_apex=1, multi=False, flicker_dz_pm=0.0, metastable=True,
                     radius_nm=float(s.get("radius_nm") or 1.0) * 2.2)
        elif out == "destroyed":
            s.update(flicker_dz_pm=150.0, metastable=True, radius_nm=float(s.get("radius_nm") or 1.0) * 3.5)
    elif kind == "poke":
        if out == "ring_up":
            s.update(n_apex=2, multi=True, radius_nm=float(s.get("radius_nm") or 1.0) * 2.2)
        elif out in ("cluster", "pit") and float(ev.get("depth_pm") or 0.0) <= 700.0:
            s.update(flicker_dz_pm=0.0, metastable=False)
    return s


def build_tip(ledger: Ledger, frames: list[dict], sim_start: float, sim_end: float) -> tuple[list[dict], str]:
    """The tip over time: ``[{sim, state, look, cause}]`` and where it came from
    (``timeline`` exact, ``reconstructed`` from world events, ``none``)."""
    ep = ledger.episode
    tt = ledger.tip_timeline
    first = (tt or {}).get("initial") or (ep.get("truth_before") or {}).get("tip")
    out: list[dict] = []

    def add(sim: float, state: dict | None, cause: str | None, detail: dict | None = None) -> None:
        red = _reduce(state)
        out.append({"sim": float(sim), "state": red, "look": lay.tip_look(red), "cause": cause,
                    "detail": {k: v for k, v in (detail or {}).items() if isinstance(v, (int, float, str, bool))}})

    add(sim_start, first, None)
    if tt is not None:
        # a change drawn by a frame's render lands at the row it hit — unless the frame
        # stopped first; then it shows from where the frame stopped (the next frame has it)
        owner: dict[int, dict] = {}
        for fr in frames:
            rng = fr.get("tip_events")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                for i in range(int(rng[0]), int(rng[1])):
                    owner[i] = fr
        for e in tt.get("events") or []:
            t = float(e.get("sim_s", sim_start))
            fr = owner.get(int(e.get("i", -1)))
            if fr is not None:
                t = min(t, float(fr.get("sim1", t)))
            add(min(max(t, sim_start), sim_end), e.get("state"), e.get("kind"), e.get("detail"))
        out.sort(key=lambda r: r["sim"])
        return out, "timeline"
    state = dict(first or {})
    n = 0
    for e in ledger.events:
        if e.get("kind") in ("crash", "pulse", "poke"):
            state = _synth_tip(state, e)
            add(float(e["sim_s"]), state, e.get("kind"), {"outcome": e.get("outcome")})
            n += 1
    final = (ep.get("truth_after") or {}).get("tip")
    if final:
        if n:
            out[-1]["state"] = _reduce(final)          # the real end state beats the inference
            out[-1]["look"] = lay.tip_look(out[-1]["state"])
        elif lay.tip_look(_reduce(final))["kind"] != out[-1]["look"]["kind"]:
            add(sim_end, final, None)
    return out, "reconstructed" if first else "none"


# ── atoms ───────────────────────────────────────────────────────────────────
def _lattice_xy(lat: dict, i: int, j: int) -> list[float]:
    a = float(lat["a_nm"])
    th = math.radians(float(lat["angle_deg"])) + math.pi / 6
    ox, oy = lat.get("origin_nm") or [0.0, 0.0]
    return [i * a * math.cos(th) + j * a * math.cos(th + math.pi / 3) + ox,
            i * a * math.sin(th) + j * a * math.sin(th + math.pi / 3) + oy]


def build_atoms(ledger: Ledger) -> dict | None:
    """Adatoms in the sample frame (nm): where they started, every move, and the designed
    target / ring when the scenario has one."""
    snap = (ledger.episode.get("truth_before") or {}).get("adatoms")
    if not isinstance(snap, dict) or not snap.get("sites"):
        return None
    initial = [{"id": s["id"], "role": s.get("role") or "", "x": s["x_nm"], "y": s["y_nm"],
                "status": s.get("status", "on_surface")} for s in snap["sites"]]
    moves: list[dict] = []
    for e in ledger.events:
        k = e.get("kind")
        t = float(e.get("row_sim_s", e.get("sim_s", 0.0)))
        if k == "adatom_hop" and e.get("to_xy_nm"):
            moves.append({"sim": t, "id": e.get("atom_id"), "x": e["to_xy_nm"][0], "y": e["to_xy_nm"][1],
                          "status": "on_surface", "cause": e.get("cause"), "n": int(e.get("n_hops") or 1)})
        elif k == "adatom_dropped" and e.get("xy_nm"):
            moves.append({"sim": t, "id": e.get("atom_id"), "x": e["xy_nm"][0], "y": e["xy_nm"][1],
                          "status": "on_surface", "cause": "drop"})
        elif k == "adatom_picked":
            moves.append({"sim": t, "id": e.get("atom_id"), "status": "on_tip", "cause": e.get("cause")})
        elif k == "adatom_lost":
            for aid in e.get("atom_ids") or []:
                moves.append({"sim": t, "id": aid, "status": "lost", "cause": e.get("cause")})
    moves.sort(key=lambda m: m["sim"])
    out: dict = {"initial": initial, "moves": moves, "a_nm": (snap.get("lattice") or {}).get("a_nm"),
                 "sigma_nm": (snap.get("params") or {}).get("sigma_nm")}
    tgt = snap.get("target")
    if isinstance(tgt, dict) and tgt.get("rule") == "any":
        # no atom is named: mark the one that was moved (on its goal if any is), else the
        # example the scenario guarantees
        after = (((ledger.episode.get("truth_after") or {}).get("adatoms") or {}).get("target") or {})
        cands = [c for c in after.get("candidates") or [] if c.get("goal_nm")]
        best = max(cands, key=lambda c: (bool(c.get("on_goal")), int(c.get("n_hops") or 0)), default=None)
        if best is not None:
            out["target"] = {"atom_id": best["id"], "site": best["goal_nm"], "start": best.get("start_nm")}
        elif isinstance(tgt.get("example"), dict):
            ex = tgt["example"]
            out["target"] = {"atom_id": ex.get("atom_id"), "site": ex.get("goal_nm"), "start": ex.get("start_nm")}
    elif isinstance(tgt, dict) and tgt.get("site_nm"):
        out["target"] = {"atom_id": tgt.get("atom_id"), "site": tgt["site_nm"], "start": tgt.get("start_nm")}
    ring = snap.get("ring")
    lat = snap.get("lattice")
    if isinstance(ring, dict) and lat:
        out["ring"] = {"centre": ring.get("centre_nm"), "radius_nm": ring.get("radius_nm"),
                       "sites": [_lattice_xy(lat, *s) for s in ring.get("sites") or []],
                       "gaps": [_lattice_xy(lat, *s) for s in ring.get("gap_sites") or []]}
    return out


# ── moments ─────────────────────────────────────────────────────────────────
def build_moments(ledger: Ledger, steps: list[dict], tip: list[dict], atoms: dict | None,
                  frames: list[dict], sim_end: float) -> list[dict]:
    """The instants the audience should not miss, each ``{sim, level, kind, title, text}``."""
    out: list[dict] = []

    def add(sim, level, kind, title, text=""):
        out.append({"sim": float(sim), "level": level, "kind": kind, "title": title, "text": text})

    saved = [f for f in frames if f.get("file")]
    if saved:
        add(saved[0]["sim0"], "info", "scan", "开始扫第一幅图")
    rank = {"good": 0, "warn": 1, "bad": 2}
    for a, b in zip(tip, tip[1:]):
        la, lb = a["look"], b["look"]
        if (la["kind"], la["level"]) == (lb["kind"], lb["level"]):
            continue
        cause = lay.TIP_CAUSE.get(str(b.get("cause") or ""), "")
        text = lb["text"] + (f"（起因：针尖{cause}。）" if cause else "")
        if lb["level"] == "good":
            add(b["sim"], "good", "tip", f"针尖恢复了，{lb['title']}", text)
        elif rank[lb["level"]] < rank[la["level"]]:
            add(b["sim"], "good", "tip", f"针尖好一些了：{lb['title']}", text)
        else:
            add(b["sim"], lb["level"], "tip", f"针尖{lb['title']}", text)
    for e in ledger.events:
        k = e.get("kind")
        if k == "crash":
            add(e["sim_s"], "bad", "crash", "针尖撞上了表面！", "针尖扎进了样品，在表面留下一个坑。")
        elif k == "adatom_lost":
            n = len(e.get("atom_ids") or [])
            add(e["sim_s"], "bad", "atom_lost", f"{n} 个原子被撞没了" if n > 1 else "一个原子被撞没了")
        elif k == "pulse":
            add(e["sim_s"], "info", "pulse", "给针尖打了一个电压脉冲", "想把针尖“打尖”：用几伏的电压让尖端的原子重新排一排。")
        elif k == "poke":
            add(e["sim_s"], "info", "poke", "让针尖轻戳了一下表面", "想让针尖粘上一点金属、重新长出一个尖。")
        elif k == "adatom_picked":
            add(float(e.get("row_sim_s", e["sim_s"])), "warn", "pickup", "原子被针尖捡了起来")
        elif k == "adatom_dropped":
            add(e["sim_s"], "info", "drop", "针尖把原子放回了表面")
    if atoms:
        # one drag = consecutive hops of one atom
        pos = {a["id"]: (a["x"], a["y"]) for a in atoms["initial"]}
        drag = None
        target = (atoms.get("target") or {}).get("atom_id")

        def close(d):
            if d is None:
                return
            dist = math.hypot(d["x1"] - d["x0"], d["y1"] - d["y0"])
            by_scan = d["cause"] == "scan"
            who = "目标原子" if d["id"] == target else "一个原子"
            title = (f"扫描时把{who}带动了 {lay.num(dist, 2)} nm" if by_scan
                     else f"{who}被针尖拖动了 {lay.num(dist, 2)} nm")
            add(d["sim0"], "warn" if by_scan or d["id"] != target else "good", "drag", title,
                f"跳了 {d['hops']} 个格点。")

        for m in atoms["moves"]:
            if m.get("status") != "on_surface" or "x" not in m or m.get("cause") == "drop":
                pos[m["id"]] = (m.get("x"), m.get("y")) if "x" in m else pos.get(m["id"])
                continue
            x0, y0 = pos.get(m["id"], (m["x"], m["y"]))
            if drag and (drag["id"] != m["id"] or m["sim"] - drag["sim1"] > DRAG_GAP_S or drag["cause"] != m.get("cause")):
                close(drag)
                drag = None
            if drag is None:
                drag = {"id": m["id"], "sim0": m["sim"], "sim1": m["sim"], "x0": x0, "y0": y0, "hops": 0,
                        "cause": m.get("cause")}
            drag.update(sim1=m["sim"], x1=m["x"], y1=m["y"], hops=drag["hops"] + int(m.get("n") or 1))
            pos[m["id"]] = (m["x"], m["y"])
        close(drag)
    drift_on = False
    for s in steps:
        if s.get("stub"):
            continue
        if s["name"] == "ReportResult":
            add(s["sim1"], "info", "report", s["caption"].replace("提交答案：", "AI 提交答案："))
        elif s["name"] == "ReportTipState":
            add(s["sim1"], "info", "diagnosis", "AI " + s["caption"])
        elif s["name"] == "SetDriftCompensation" and "开启" in s["caption"]:
            if drift_on:
                add(s["sim1"], "info", "drift", "AI 重新调了漂移补偿", "又量了一次漂移，把补偿调得更准。")
            else:
                add(s["sim1"], "info", "drift", "AI 开启了漂移补偿", "样品在慢慢漂移，AI 让扫描跟着它走。")
            drift_on = True
        elif s["name"] == "SetDriftCompensation":
            drift_on = False
    ep = ledger.episode
    ok = bool(ep.get("success"))
    add(sim_end, "good" if ok else "bad", "end", "结束：复现成功" if ok else "结束：没能复现")
    # at one instant the cause reads before its consequences: crash, then the tip, then atoms
    order = {"crash": 0, "pulse": 0, "poke": 0, "tip": 1, "atom_lost": 2, "pickup": 2, "drop": 2}
    out.sort(key=lambda m: (m["sim"], order.get(m["kind"], 3)))
    return out


# ── the whole thing ─────────────────────────────────────────────────────────
def build_timeline(ledger: Ledger, *, claim_specs: dict[str, dict] | None = None) -> dict:
    ep = ledger.episode
    tb, ta = ep.get("truth_before") or {}, ep.get("truth_after") or {}
    sim_start = float(tb.get("sim_s", 0.0))
    sim_end = max(float(ta.get("sim_s", sim_start)), sim_start)
    ts = float(ep.get("time_scale") or 1.0)
    steps, messages, clock = build_steps(ledger.driver, ledger.events, sim_start, sim_end, ts)
    frames = build_frames(ledger.events)
    tip, tip_source = build_tip(ledger, frames, sim_start, sim_end)
    atoms = build_atoms(ledger)
    moments = build_moments(ledger, steps, tip, atoms, frames, sim_end)
    reports: dict[str, dict] = {}
    for s in steps:
        if s["name"] == "ReportResult" and s["args"].get("claim_id"):
            reports[str(s["args"]["claim_id"])] = s["args"]
    claims = [lay.claim_verdict(cid, rec, (claim_specs or {}).get(cid), reports.get(cid))
              for cid, rec in ((ep.get("verdict") or {}).get("details", {}).get("claims") or {}).items()]
    caps = ep.get("caps") or {}
    return {"sim_start": sim_start, "sim_end": sim_end, "budget_s": caps.get("max_sim_s"),
            "time_scale": ts, "clock": clock, "steps": steps, "messages": messages, "frames": frames,
            "tip": tip, "tip_source": tip_source, "atoms": atoms, "moments": moments, "claims": claims,
            "final_text": _clean_text(str(((ledger.driver or {}).get("final_text")) or ""))}
