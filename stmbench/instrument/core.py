"""The benchmark instrument: primitive actions on stmsim, on a virtual clock, without MAST.

Why it exists. A benchmark's agent-facing interface must not belong to one agent framework,
and an episode must be re-checkable: the same seed and the same actions have to give the same
measurements and the same verdict on any host. :class:`Instrument` drives the simulator's own
verbs - the ones the controller modules call when a client sends ``Scan.Action``,
``BiasSpectr.Start``, ``FolMe.XYPosSet`` ... - directly, on a
:class:`~stmsim.physics.clock.VirtualClock` that each action advances by its physical
duration, and keeps a log from which :func:`replay` rebuilds the episode.

Three kinds of call:

* **actions** (:data:`ACTIONS`, :meth:`Instrument.act`) - the only calls that change the
  instrument or spend instrument time;
* **observations** (``status``, ``info``, ``list_data``, ``frame``, ``spectrum``) - read only:
  no instrument time and no random draw, so how often an agent looks cannot change what it
  later measures;
* **reports** (``report``, ``report_tip_state``, ``finish``) - the result channel of
  :mod:`stmbench.harness.results`, judged by the version-1 rules shown on the rule card.

What is deliberately absent: MAST's composite skills and its safety gate. An action does what
the instrument would do, hard limits only (bias, piezo range, preamplifier range); a poke too
deep, a pulse too high, a coarse step in tunnelling have their physical consequences.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stmsim.physics.scanner import SIG_CURRENT, SIG_Z
from stmsim.scenario import Scenario

from ..harness.results import ResultSink, fold_results
from ..trackB.rules import OVERRUN_TOLERANCE, RULES_VERSION, task_text

#: version of the action set, the observations and the log format
API_VERSION = 1
SCENARIO_DIR = Path(__file__).resolve().parents[1] / "trackB" / "scenarios"
MAX_ACTIONS = 3000
_GAIN_FULL_SCALE_NA = {10000.0: 0, 1000.0: 1, 100.0: 2, 10.0: 3, 1.0: 4, 0.1: 5}


class ActionError(ValueError):
    """A refused call: nothing happened on the instrument and no time passed."""


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                       # number | integer | boolean | string
    doc: str
    required: bool = False
    default: Any = None
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


@dataclass(frozen=True)
class ActionSpec:
    name: str
    doc: str
    params: tuple[Param, ...] = ()

    def schema(self) -> dict:
        """JSON schema of the parameters (MCP ``inputSchema``)."""
        props: dict[str, dict] = {}
        for p in self.params:
            t = {"number": "number", "integer": "integer", "boolean": "boolean", "string": "string"}[p.kind]
            d: dict[str, Any] = {"type": t, "description": p.doc}
            if p.lo is not None:
                d["minimum"] = p.lo
            if p.hi is not None:
                d["maximum"] = p.hi
            if p.choices is not None:
                d["enum"] = list(p.choices)
            if p.default is not None:
                d["default"] = p.default
            props[p.name] = d
        return {"type": "object", "properties": props,
                "required": [p.name for p in self.params if p.required], "additionalProperties": False}


def _P(*a, **k) -> Param:
    return Param(*a, **k)


ACTIONS: dict[str, ActionSpec] = {s.name: s for s in (
    ActionSpec("scan", "Image a square frame in constant-current mode and save it. Returns the frame's "
                       "file name, which reports cite. Takes pixels × 2 × line_time_s of instrument time.",
               (_P("center_x_nm", "number", "frame centre x (scan coordinates, nm)", True),
                _P("center_y_nm", "number", "frame centre y (scan coordinates, nm)", True),
                _P("size_nm", "number", "side of the square frame (nm)", True, lo=0.5, hi=2000.0),
                _P("pixels", "integer", "pixels per line = lines", default=256, lo=16, hi=1024),
                _P("angle_deg", "number", "scan rotation, anticlockwise (deg)", default=0.0, lo=-180.0, hi=180.0),
                _P("line_time_s", "number", "time per line, each way (s)", default=0.586, lo=0.02, hi=60.0),
                _P("direction", "string", "slow-axis direction", default="down", choices=("down", "up")))),
    ActionSpec("set_bias", "Set the sample bias.", (_P("bias_v", "number", "bias (V)", True, lo=-10.0, hi=10.0),)),
    ActionSpec("set_setpoint", "Set the tunnelling-current setpoint of the feedback loop; the loop settles.",
               (_P("current_pa", "number", "setpoint current (pA)", True, lo=0.1, hi=1.0e5),)),
    ActionSpec("feedback", "Switch the Z feedback loop on or off (off holds Z where it is).",
               (_P("on", "boolean", "true = on", True),)),
    ActionSpec("set_current_range", "Select the preamplifier range (full-scale current).",
               (_P("full_scale_na", "number", "full scale (nA)", True, choices=tuple(_GAIN_FULL_SCALE_NA)),)),
    ActionSpec("move_tip", "Move the tip laterally in a straight line at a set speed, in feedback. At a low "
                           "junction resistance (|bias| / setpoint) this drags whatever the tip grips.",
               (_P("x_nm", "number", "target x (scan coordinates, nm)", True),
                _P("y_nm", "number", "target y (scan coordinates, nm)", True),
                _P("speed_nm_s", "number", "lateral speed (nm/s)", default=293.0, lo=0.01, hi=10000.0))),
    ActionSpec("pulse", "Apply a bias pulse at the tip's position (tip conditioning).",
               (_P("bias_v", "number", "pulse voltage (V, absolute)", True, lo=-10.0, hi=10.0),
                _P("width_ms", "number", "pulse width (ms)", default=50.0, lo=0.1, hi=5000.0),
                _P("z_hold", "boolean", "hold Z during the pulse", default=True))),
    ActionSpec("poke", "Indent the tip into the surface and retract (tip conditioning by controlled contact).",
               (_P("depth_nm", "number", "indentation depth below the tunnelling height (nm)", default=0.5, lo=0.0, hi=20.0),
                _P("bias_v", "number", "bias during the indentation (V); omit to keep the current bias", lo=-10.0, hi=10.0),
                _P("dwell_s", "number", "time held at depth (s)", default=0.5, lo=0.0, hi=10.0))),
    ActionSpec("sts", "Take a bias-spectroscopy sweep (I(V) and dI/dV) at the tip's position and save it. "
                      "Returns the spectrum's file name, which reports cite.",
               (_P("v_start", "number", "sweep start (V)", True, lo=-10.0, hi=10.0),
                _P("v_end", "number", "sweep end (V)", True, lo=-10.0, hi=10.0),
                _P("points", "integer", "points", default=200, lo=8, hi=4096),
                _P("settle_ms", "number", "settling time per point (ms)", default=5.0, lo=0.0, hi=1000.0),
                _P("integration_ms", "number", "integration time per point (ms)", default=5.0, lo=0.0, hi=1000.0),
                _P("z_offset_nm", "number", "tip offset from the setpoint height, positive = away (nm)", default=0.0, lo=-2.0, hi=5.0),
                _P("backward", "boolean", "also sweep back", default=True))),
    ActionSpec("zspec", "Take a Z sweep toward the surface at the tip's position (I(z); Δf(z) with the qPlus "
                        "sensor oscillating) and save it.",
               (_P("sweep_nm", "number", "distance swept toward the surface (nm)", default=0.2, lo=0.01, hi=5.0),
                _P("points", "integer", "points", default=100, lo=8, hi=4096),
                _P("z_offset_nm", "number", "start offset from the setpoint height, positive = away (nm)", default=0.0, lo=-2.0, hi=5.0),
                _P("sweeps", "integer", "sweeps averaged", default=1, lo=1, hi=100),
                _P("settle_ms", "number", "settling time per point (ms)", default=5.0, lo=0.0, hi=1000.0),
                _P("integration_ms", "number", "integration time per point (ms)", default=5.0, lo=0.0, hi=1000.0),
                _P("backward", "boolean", "also sweep back", default=True))),
    ActionSpec("pll", "Switch the qPlus oscillation (PLL output with amplitude control) on or off; on, the "
                      "oscillation rings up and the frequency shift is re-zeroed where the tip is.",
               (_P("on", "boolean", "true = oscillating", True),
                _P("amplitude_pm", "number", "oscillation amplitude setpoint (pm)", lo=1.0, hi=2000.0))),
    ActionSpec("drift_comp", "Program the drift compensation: the piezo follows features moving at this velocity "
                             "in the scan frame (what a two-frame drift measurement reports).",
               (_P("on", "boolean", "true = compensating", True),
                _P("vx_nm_per_min", "number", "x velocity of the features (nm/min)", default=0.0, lo=-100.0, hi=100.0),
                _P("vy_nm_per_min", "number", "y velocity of the features (nm/min)", default=0.0, lo=-100.0, hi=100.0),
                _P("vz_pm_per_min", "number", "z velocity (pm/min)", default=0.0, lo=-1000.0, hi=1000.0))),
    ActionSpec("withdraw", "Retract the tip fully and switch the feedback off."),
    ActionSpec("approach", "Coarse auto-approach until tunnelling is found (or the timeout).",
               (_P("timeout_s", "number", "give up after (s)", default=600.0, lo=1.0, hi=7200.0),)),
    ActionSpec("coarse_move", "Step the coarse positioner. X/Y steps of 3 or more move to a new area of the "
                              "sample; stepping toward the sample in tunnelling crashes the tip.",
               (_P("direction", "string", "axis and sense", True, choices=("X+", "X-", "Y+", "Y-", "Z+", "Z-")),
                _P("steps", "integer", "steps", True, lo=1, hi=10000))),
    ActionSpec("wait", "Let instrument time pass (drift and creep go on).",
               (_P("seconds", "number", "how long (s)", True, lo=0.0, hi=7200.0),)),
)}


def validate(spec: ActionSpec, args: dict) -> dict:
    """Parameters with defaults filled in, checked against the spec; :class:`ActionError` if not."""
    args = dict(args or {})
    unknown = sorted(set(args) - {p.name for p in spec.params})
    if unknown:
        raise ActionError(f"{spec.name}: unknown parameter(s) {unknown}")
    out: dict[str, Any] = {}
    for p in spec.params:
        if p.name not in args or args[p.name] is None:
            if p.required:
                raise ActionError(f"{spec.name}: {p.name} is required")
            if p.default is not None:
                out[p.name] = p.default
            continue
        v = args[p.name]
        if p.kind == "boolean":
            if not isinstance(v, bool):
                raise ActionError(f"{spec.name}: {p.name} must be true or false")
        elif p.kind in ("number", "integer"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ActionError(f"{spec.name}: {p.name} must be a number")
            if p.kind == "integer":
                if float(v) != int(v):
                    raise ActionError(f"{spec.name}: {p.name} must be an integer")
                v = int(v)
            else:
                v = float(v)
            if p.lo is not None and v < p.lo or p.hi is not None and v > p.hi:
                raise ActionError(f"{spec.name}: {p.name} = {v} is outside [{p.lo}, {p.hi}]")
        elif p.kind == "string":
            if not isinstance(v, str):
                raise ActionError(f"{spec.name}: {p.name} must be a string")
        if p.choices is not None and v not in p.choices:
            raise ActionError(f"{spec.name}: {p.name} must be one of {list(p.choices)}")
        out[p.name] = v
    return out


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def array_digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


# ── the instrument ──────────────────────────────────────────────────────────
@dataclass
class FrameRecord:
    name: str
    geometry: dict
    z_fwd: np.ndarray            # image orientation: row 0 = top (high y), column 0 = left
    z_bwd: np.ndarray
    i_fwd: np.ndarray
    i_bwd: np.ndarray
    t_start_s: float
    meta: dict = field(default_factory=dict)


class Instrument:
    """One benchmark episode on the simulated instrument (see the module docstring)."""

    def __init__(self, scenario: Scenario | str | Path, seed: int, out_dir: str | Path | None = None, *,
                 max_actions: int = MAX_ACTIONS):
        if not isinstance(scenario, Scenario):
            p = Path(scenario)
            scenario = Scenario.load(p if p.exists() else SCENARIO_DIR / f"{scenario}.yaml")
        self.scenario = scenario
        self.seed = int(seed)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            session = self.out_dir / "session"
        else:
            self._tmp = tempfile.TemporaryDirectory(prefix="stmbench_instrument_")
            session = Path(self._tmp.name) / "session"
        self.world = scenario.build_world(self.seed, session_dir=session, virtual_clock=True)
        self.t0 = self.world.clock.sim()
        b = scenario.budget or {}
        self.max_sim_s = float(b["sim_hours"]) * 3600.0 if b.get("sim_hours") else None
        self.max_actions = int(max_actions)
        self.log: list[dict] = []
        self.sink = ResultSink()
        self.diagnosis: list[dict] = []
        self.frames: dict[str, FrameRecord] = {}
        self.spectra: dict[str, dict] = {}
        self.n_actions = 0
        self.finished = False
        self.verdict: Any = None

    # ── what the agent is told ──
    def task(self) -> str:
        """The task with its rule card (stmbench.trackB.rules)."""
        return task_text(self.scenario)

    @staticmethod
    def manual() -> str:
        """The action set, one line per parameter: what every agent is told the instrument does."""
        lines = [f"Instrument API v{API_VERSION}: units are nm, V, pA, s; positions are scan coordinates "
                 "(x right, y up); actions spend instrument time, observations do not."]
        for s in ACTIONS.values():
            lines.append(f"- {s.name}: {s.doc}")
            for p in s.params:
                extra = []
                if p.required:
                    extra.append("required")
                if p.default is not None:
                    extra.append(f"default {p.default}")
                if p.choices is not None:
                    extra.append("one of " + ", ".join(map(str, p.choices)))
                lines.append(f"    {p.name} ({p.kind}): {p.doc}" + (f" [{'; '.join(extra)}]" if extra else ""))
        return "\n".join(lines)

    # ── time ──
    @property
    def used_s(self) -> float:
        """Instrument time spent so far (s)."""
        return float(self.world.clock.sim() - self.t0)

    def _advance(self, seconds: float) -> None:
        self.world.clock.advance(max(0.0, float(seconds)))

    def _transient_left_s(self) -> float:
        w = self.world
        now = w.clock.wall()
        left = [tr.t0_wall + float(tr.t[-1]) - now for tr in w.transients]
        return max([0.0, *left])

    def _settle(self, at_least: float = 0.0) -> None:
        self._advance(max(float(at_least), self._transient_left_s()))

    # ── the one door that changes the instrument ──
    def act(self, name: str, **args) -> dict:
        """Run one action; returns what the instrument reports. :class:`ActionError` if refused."""
        try:
            if self.finished:
                raise ActionError("the episode is over")
            spec = ACTIONS.get(name)
            if spec is None:
                raise ActionError(f"unknown action {name!r} (known: {', '.join(ACTIONS)})")
            params = validate(spec, args)
            if self.max_sim_s is not None and self.used_s > self.max_sim_s * OVERRUN_TOLERANCE:
                raise ActionError("the instrument-time budget is spent")
            if self.n_actions >= self.max_actions:
                raise ActionError(f"the action limit ({self.max_actions}) is reached")
        except ActionError as exc:
            self.log.append({"i": len(self.log), "kind": "refused", "name": name, "args": _jsonable(args),
                             "error": str(exc), "t_s": _r(self.used_s, 6)})
            raise
        t0 = self.used_s
        out = getattr(self, f"_do_{name}")(**params)
        self.n_actions += 1
        self.log.append({"i": len(self.log), "kind": "act", "name": name, "params": params,
                         "t0_s": _r(t0, 6), "t1_s": _r(self.used_s, 6), "result": _jsonable(out)})
        return {**out, "instrument_time_s": _r(self.used_s, 3), "budget_s": self.max_sim_s}

    # ── actions ──
    def _do_scan(self, center_x_nm, center_y_nm, size_nm, pixels, angle_deg, line_time_s, direction) -> dict:
        w = self.world
        cx, cy, size = center_x_nm * 1e-9, center_y_nm * 1e-9, size_nm * 1e-9
        rx, ry = w.rig.xy_range_m
        reach = abs(math.cos(math.radians(angle_deg))) + abs(math.sin(math.radians(angle_deg)))
        if abs(cx) + size * reach / 2 > rx / 2 * 1.02 or abs(cy) + size * reach / 2 > ry / 2 * 1.02:
            raise ActionError("scan: the frame exceeds the piezo range")
        st = w.scan
        st.cx, st.cy, st.w, st.h, st.angle_deg = cx, cy, size, size, float(angle_deg)
        st.nx = st.ny = int(pixels)
        st.line_time_fwd_s = st.line_time_bwd_s = float(line_time_s)
        st.speed_fwd_m_per_s = st.speed_bwd_m_per_s = size / float(line_time_s)
        st.autosave = st.continuous = st.bouncy = False
        w.move_xy(cx, cy)
        w.scan_start(direction_up=(direction == "up"))
        fr = w.frame
        t_start = self.used_s
        self._advance(st.frame_time_s)
        w.scan_running()                                  # reveals the last rows, ends the frame
        path = w.save_frame()
        name = Path(path).name
        order = np.arange(st.ny) if direction == "down" else np.arange(st.ny)[::-1]
        z_f, z_b = fr.data[SIG_Z]
        i_f, i_b = fr.data[SIG_CURRENT]
        geom = {"center_nm": [_r(center_x_nm), _r(center_y_nm)], "size_nm": _r(size_nm),
                "pixels": int(pixels), "angle_deg": _r(angle_deg), "direction": direction,
                "nm_per_px": _r(size_nm / pixels, 6), "line_time_s": _r(line_time_s, 6)}
        self.frames[name] = FrameRecord(
            name=name, geometry=geom,
            # backward lines are acquired right to left: flip them into image columns
            z_fwd=np.asarray(z_f)[order], z_bwd=np.asarray(z_b)[order][:, ::-1],
            i_fwd=np.asarray(i_f)[order], i_bwd=np.asarray(i_b)[order][:, ::-1],
            t_start_s=t_start, meta={"bias_v": w.bias_v, "setpoint_pa": w.zctrl.setpoint_a * 1e12})
        return {"frame": name, **geom, "duration_s": _r(st.frame_time_s, 3)}

    def _do_set_bias(self, bias_v) -> dict:
        self.world.set_bias(float(bias_v))
        self._settle(0.05)
        return {"bias_v": _r(bias_v, 6)}

    def _do_set_setpoint(self, current_pa) -> dict:
        self.world.set_setpoint(float(current_pa) * 1e-12)
        self._settle(0.05)
        return {"setpoint_pa": _r(current_pa, 6)}

    def _do_feedback(self, on) -> dict:
        self.world.zctrl_set(bool(on))
        self._settle(0.05)
        return {"feedback": bool(on)}

    def _do_set_current_range(self, full_scale_na) -> dict:
        w = self.world
        idx = _GAIN_FULL_SCALE_NA[float(full_scale_na)]
        w.preamp.gain_index = idx
        w.preamp.full_scale_a = 10.0 / (10.0 ** (6 + idx))
        self._settle(0.05)
        return {"full_scale_na": float(full_scale_na)}

    def _do_move_tip(self, x_nm, y_nm, speed_nm_s) -> dict:
        w = self.world
        x, y = x_nm * 1e-9, y_nm * 1e-9
        rx, ry = w.rig.xy_range_m
        if abs(x) > rx / 2 or abs(y) > ry / 2:
            raise ActionError("move_tip: outside the piezo range")
        dist = math.hypot(x - w.tip_x, y - w.tip_y)
        n_before = len(w.events)
        w.move_xy_path(x, y, speed_m_s=float(speed_nm_s) * 1e-9)
        self._settle(dist / (float(speed_nm_s) * 1e-9) + 0.05)
        # what an operator sees in the current trace: hops show as spikes, not as atom ids
        hops = sum(int(e.get("n_hops", 1)) for e in w.events[n_before:] if e.get("kind") == "adatom_hop")
        return {"tip_nm": [_r(x_nm), _r(y_nm)], "distance_nm": _r(dist * 1e9),
                "current_spikes": int(hops)}

    def _do_pulse(self, bias_v, width_ms, z_hold) -> dict:
        w = self.world
        if w.withdrawn:
            raise ActionError("pulse: the tip is withdrawn")
        w.bias_pulse(float(width_ms) * 1e-3, float(bias_v), bool(z_hold), 1)
        self._settle(float(width_ms) * 1e-3 + 0.1)
        return {"bias_v": _r(bias_v, 6), "width_ms": _r(width_ms, 6)}

    def _do_poke(self, depth_nm, dwell_s, bias_v=None) -> dict:
        w = self.world
        if w.withdrawn:
            raise ActionError("poke: the tip is withdrawn")
        props = {"switch_off_delay": 0.05, "change_bias": bias_v is not None,
                 "bias_v": float(bias_v) if bias_v is not None else w.bias_v,
                 "tip_lift_m": -float(depth_nm) * 1e-9, "lift_time_1_s": 0.1, "bias_lift_v": 0.0,
                 "bias_settling_s": float(dwell_s), "lift_height_m": float(depth_nm) * 1e-9,
                 "lift_time_2_s": 0.1, "end_wait_s": 0.5, "restore_feedback": True}
        w.tip_shaper_start(props)
        total = 0.05 + 0.1 + float(dwell_s) + 0.1 + 0.5 + 0.3
        self._settle(total)
        return {"depth_nm": _r(depth_nm, 6), "dwell_s": _r(dwell_s, 6)}

    def _do_sts(self, v_start, v_end, points, settle_ms, integration_ms, z_offset_nm, backward) -> dict:
        from stmsim.io.dat_writer import save_bias_spectrum

        w = self.world
        if w.withdrawn:
            raise ActionError("sts: no tunnelling junction (the tip is withdrawn)")
        settle, integ = float(settle_ms) * 1e-3, float(integration_ms) * 1e-3
        cur = w.sts_curve(float(v_start), float(v_end), int(points), z_offset_m=float(z_offset_nm) * 1e-9,
                          settle_s=settle, integ_s=integ)
        duration = 0.3 + int(points) * (settle + integ) * (2 if backward else 1) + 0.1
        path = save_bias_spectrum(w, "", v=cur["v"], i=cur["i"], didv=cur["didv"])
        w.dats_saved.append(path)
        if w.sts_records:
            w.sts_records[-1]["path"] = path
        w._event("dat_saved", path=path, experiment="bias spectroscopy", idx=len(w.dats_saved),
                 sts_idx=w.sts_counter)
        name = Path(path).name
        self.spectra[name] = {"name": name, "kind": "sts", "index": int(w.sts_counter),
                              "tip_nm": [_r(w.tip_x * 1e9), _r(w.tip_y * 1e9)],
                              "v": np.asarray(cur["v"], float), "i": np.asarray(cur["i"], float),
                              "didv": np.asarray(cur["didv"], float), "t_s": self.used_s}
        self._advance(duration)
        return {"spectrum": name, "points": int(points), "duration_s": _r(duration, 3)}

    def _do_zspec(self, sweep_nm, points, z_offset_nm, sweeps, settle_ms, integration_ms, backward) -> dict:
        from stmsim.io.dat_writer import save_z_spectrum

        w = self.world
        if w.withdrawn:
            raise ActionError("zspec: no tunnelling junction (the tip is withdrawn)")
        settle, integ = float(settle_ms) * 1e-3, float(integration_ms) * 1e-3
        cur = w.zspec_curve(int(points), float(sweep_nm) * 1e-9, z_offset_m=float(z_offset_nm) * 1e-9,
                            bwd=bool(backward), sweeps=int(sweeps), settle_s=settle, integ_s=integ,
                            retract=(0, 0.0, 0, 0))
        duration = 0.3 + int(points) * (settle + integ) * (2 if backward else 1) * int(sweeps) + 0.1
        path = save_z_spectrum(w, "", z_rel=cur["z_rel"], i=cur["i"], df=cur["df"], amp=cur["amp"],
                               z_abs=np.asarray(cur["z_n"], float))
        w.dats_saved.append(path)
        if w.zspec_records:
            w.zspec_records[-1]["path"] = path
        w._event("dat_saved", path=path, experiment="Z spectroscopy", idx=len(w.dats_saved),
                 zspec_idx=cur["idx"])
        name = Path(path).name
        self.spectra[name] = {"name": name, "kind": "zspec", "index": int(cur["idx"]),
                              "tip_nm": [_r(w.tip_x * 1e9), _r(w.tip_y * 1e9)],
                              "z_rel": np.asarray(cur["z_rel"], float), "i": np.asarray(cur["i"], float),
                              "df": np.asarray(cur["df"], float), "amp": np.asarray(cur["amp"], float),
                              "t_s": self.used_s}
        self._settle(duration)
        return {"spectrum": name, "points": int(points), "duration_s": _r(duration, 3)}

    def _do_pll(self, on, amplitude_pm=None) -> dict:
        w = self.world
        p = getattr(w, "pll", None)
        if p is None:
            raise ActionError("pll: this instrument has no qPlus sensor")
        if amplitude_pm is not None:
            p.amp_setpoint_m = float(amplitude_pm) * 1e-12
        p.amp_ctrl_on = p.phase_ctrl_on = bool(on)
        if bool(on) and p.output_on:
            p.excitation_v = p.excitation_for(p.amp_setpoint_m)
        p.set_output(bool(on), w.clock.wall())
        self._advance(5.0 * p.tau_s if on else 0.05)
        # the frequency shift is re-zeroed where the tip is (PLL.FreqShiftAutoCenter)
        if on and not w.withdrawn:
            amp = p.amplitude_now(w.clock.wall())
            gap = w.current_z_tip() - w.surface_height_here()
            p.center_freq_hz = float(w.tip.qplus_f0_hz + w.df_physical(gap, amp))
        return {"oscillating": bool(on), "amplitude_setpoint_pm": _r(p.amp_setpoint_m * 1e12)}

    def _do_drift_comp(self, on, vx_nm_per_min, vy_nm_per_min, vz_pm_per_min) -> dict:
        self.world.set_drift_comp(bool(on), vx_nm_per_min * 1e-9 / 60.0, vy_nm_per_min * 1e-9 / 60.0,
                                  vz_pm_per_min * 1e-12 / 60.0)
        self._settle(0.05)
        return {"on": bool(on), "v_nm_per_min": [_r(vx_nm_per_min), _r(vy_nm_per_min)],
                "vz_pm_per_min": _r(vz_pm_per_min)}

    def _do_withdraw(self) -> dict:
        self.world.withdraw()
        self._settle(0.5)
        return {"withdrawn": True}

    def _do_approach(self, timeout_s) -> dict:
        w = self.world
        w.auto_approach_start()
        waited = 0.0
        while waited < timeout_s:
            self._advance(0.5)
            waited += 0.5
            if not w.auto_approach_poll():
                break
        if w.approach.running:
            w.auto_approach_stop()
        self._settle(0.0)
        return {"landed": bool(w.approach.landed), "duration_s": _r(waited, 3)}

    def _do_coarse_move(self, direction, steps) -> dict:
        w = self.world
        w.motor_move(str(direction), int(steps))
        self._settle(int(steps) / max(w.coarse.freq_hz, 1.0) + 0.1)
        return {"direction": direction, "steps": int(steps)}

    def _do_wait(self, seconds) -> dict:
        self._advance(float(seconds))
        return {"waited_s": _r(seconds, 3)}

    # ── observations: no instrument time, no random draw ──
    def _observe(self, what: str, **args) -> None:
        self.log.append({"i": len(self.log), "kind": "observe", "what": what, "args": _jsonable(args),
                         "t_s": _r(self.used_s, 6)})

    def _z_and_current(self) -> tuple[float, float]:
        """The loop's Z and the DC current, without readout noise (a reading must not draw)."""
        w = self.world
        tr = w._active_transient()
        if tr is not None:
            z_tip, i = tr.sample(w.clock.wall())
            return float(w.z_ctrl_from_tip(z_tip)), float(w.preamp.clamp(i))
        if w.zctrl.on and not w.withdrawn:
            w.achievable_z_tip()
        return float(w.zctrl.z_n), float(w.preamp.clamp(w._dc_current()))

    def status(self) -> dict:
        w = self.world
        self._observe("status")
        z, i = self._z_and_current()
        p = getattr(w, "pll", None)
        return {"instrument_time_s": _r(self.used_s, 3), "budget_s": self.max_sim_s,
                "bias_v": _r(w.bias_v, 6), "setpoint_pa": _r(w.zctrl.setpoint_a * 1e12, 4),
                "feedback": bool(w.zctrl.on), "withdrawn": bool(w.withdrawn),
                "z_nm": _r(z * 1e9, 5), "current_pa": _r(i * 1e12, 4),
                "tip_nm": [_r(w.tip_x * 1e9), _r(w.tip_y * 1e9)],
                "current_full_scale_na": _r(w.preamp.full_scale_a * 1e9, 4),
                "oscillating": bool(p.output_on) if p is not None else None,
                "actions": self.n_actions}

    def info(self) -> dict:
        """Fixed facts about the instrument, as an operator would find them registered."""
        w = self.world
        self._observe("info")
        rx, ry = w.rig.xy_range_m
        out = {"api_version": API_VERSION, "rules_version": RULES_VERSION,
               "piezo_xy_range_nm": [_r(rx * 1e9, 2), _r(ry * 1e9, 2)],
               "piezo_z_range_nm": _r(2 * w.rig.z_range_m * 1e9, 2), "bias_range_v": [-10.0, 10.0],
               "current_ranges_na": sorted(_GAIN_FULL_SCALE_NA, reverse=True),
               "budget_s": self.max_sim_s, "temperature_k": float(w.temperature_k),
               "material": self.scenario.material}
        if getattr(w, "pll", None) is not None:
            t = w.tip
            out["qplus"] = {"f0_hz": _r(t.qplus_f0_hz, 3), "k_n_per_m": _r(t.qplus_k_n_per_m, 3),
                            "q": _r(t.qplus_q, 1)}
        return out

    def list_data(self) -> dict:
        self._observe("list_data")
        return {"frames": [{"frame": f.name, **f.geometry, "t_start_s": _r(f.t_start_s, 3)}
                           for f in self.frames.values()],
                "spectra": [{"spectrum": s["name"], "kind": s["kind"], "tip_nm": s["tip_nm"],
                             "points": int(len(s.get("v", s.get("z_rel", [])))), "t_s": _r(s["t_s"], 3)}
                            for s in self.spectra.values()]}

    def frame(self, name: str, channel: str = "z", direction: str = "forward") -> dict:
        """A saved frame's image (row 0 = top) and geometry. ``channel``: z (m) or current (A)."""
        self._observe("frame", name=name, channel=channel, direction=direction)
        f = self.frames.get(Path(str(name)).name)
        if f is None:
            raise ActionError(f"no frame {name!r}")
        arrs = {("z", "forward"): f.z_fwd, ("z", "backward"): f.z_bwd,
                ("current", "forward"): f.i_fwd, ("current", "backward"): f.i_bwd}
        if (channel, direction) not in arrs:
            raise ActionError("channel is z or current, direction is forward or backward")
        unit = "m" if channel == "z" else "A"
        return {"frame": f.name, **f.geometry, "channel": channel, "direction": direction, "unit": unit,
                "array": np.array(arrs[(channel, direction)], copy=True), **f.meta}

    def spectrum(self, name: str) -> dict:
        self._observe("spectrum", name=name)
        s = self.spectra.get(Path(str(name)).name)
        if s is None:
            raise ActionError(f"no spectrum {name!r}")
        return {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in s.items()}

    # ── reports ──
    def report(self, claim_id: str, value: float, *, x_nm: float | None = None, y_nm: float | None = None,
               frame: str | None = None, evidence: list[str] | None = None, unit: str = "",
               sigma: float | None = None, note: str = "") -> str:
        if self.finished:
            raise ActionError("the episode is over")
        ids = self.scenario.claim_ids
        if claim_id not in ids:
            raise ActionError(f"unknown claim {claim_id!r} (this task's claims: {', '.join(ids)})")
        self.sink.report(claim_id, float(value), unit, x_nm=x_nm, y_nm=y_nm, sigma=sigma, note=note,
                         frame=frame, evidence=list(evidence) if evidence is not None else None)
        args = {"claim_id": claim_id, "value": float(value), "x_nm": x_nm, "y_nm": y_nm, "frame": frame,
                "evidence": list(evidence) if evidence is not None else None, "unit": unit, "sigma": sigma,
                "note": note}
        self.log.append({"i": len(self.log), "kind": "report", "args": args, "t_s": _r(self.used_s, 6)})
        to_report = [str(c["id"]) for c in self.scenario.claims if c.get("kind") != "constraint"]
        missing = [c for c in to_report if c not in self.sink.folded()["claims"]]
        return f"recorded {claim_id} = {value}; still to report: {', '.join(missing) or 'none'}"

    def report_tip_state(self, tip_state: str, junction: str | None = None, note: str = "") -> str:
        rec = {"tip_state": str(tip_state), "junction": junction, "note": note}
        self.diagnosis.append(rec)
        self.log.append({"i": len(self.log), "kind": "diagnosis", "args": rec, "t_s": _r(self.used_s, 6)})
        return "recorded"

    # ── the end ──
    def finish(self) -> dict:
        """Judge the episode (version-1 rules, budget overrun rule) and write the ledger."""
        if self.finished:
            return self._summary()
        from ..harness.episode import apply_budget_overrun
        from ..trackB.truth_criteria import judge

        w = self.world
        self.finished = True
        self.log.append({"i": len(self.log), "kind": "finish", "t_s": _r(self.used_s, 6)})
        truth = w.truth()
        surf = w.surface
        v = judge(self.scenario.success.get("kind", "claims_verified"), truth,
                  self.scenario.success.get("thresholds"), claims=self.scenario.claims,
                  report=self.sink.folded(), events=list(w.events), rules_version=RULES_VERSION,
                  orientation_at=(lambda x, y: surf.orientation_at(x * 1e-9, y * 1e-9)))
        self.verdict = apply_budget_overrun(v, self.used_s, self.max_sim_s)
        self.truth_after = truth
        if self.out_dir is not None:
            self._write_ledger()
        return self._summary()

    def _summary(self) -> dict:
        v = self.verdict
        det = (v.details or {}) if v is not None else {}
        return {"scenario_id": self.scenario.id, "seed": self.seed, "success": bool(v.success) if v else None,
                "partial": float(v.partial) if v else None, "claims_total": det.get("n_total"),
                "claims_verified": det.get("n_verified"), "instrument_time_s": _r(self.used_s, 3),
                "actions": self.n_actions, "budget_overrun": det.get("budget_overrun")}

    def digest(self) -> dict:
        """Hashes of everything measured, for re-checking an episode by replay."""
        return {"frames": {n: {k: array_digest(getattr(f, k)) for k in ("z_fwd", "z_bwd", "i_fwd", "i_bwd")}
                           for n, f in self.frames.items()},
                "spectra": {n: {k: array_digest(v) for k, v in s.items() if isinstance(v, np.ndarray)}
                            for n, s in self.spectra.items()},
                "events": hashlib.sha256(json.dumps(_portable_events(self.world.events), sort_keys=True,
                                                    default=str).encode()).hexdigest()}

    def _write_ledger(self) -> None:
        out = self.out_dir
        v = self.verdict
        det = v.details or {}
        ep = {"api_version": API_VERSION, "rules_version": RULES_VERSION, "scenario_id": self.scenario.id,
              "paper_id": self.scenario.paper_id, "seed": self.seed, "success": bool(v.success),
              "partial": float(v.partial), "reproduced": bool(v.success),
              "claims_total": det.get("n_total"), "claims_reported": det.get("n_reported"),
              "claims_verified": det.get("n_verified"), "instrument_time_s": self.used_s,
              "max_sim_s": self.max_sim_s, "actions": self.n_actions, "verdict": v.as_dict(),
              "digest": self.digest(), "diagnosis": self.diagnosis, "truth_after": self.truth_after}
        (out / "episode.json").write_text(json.dumps(ep, default=_json_default, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        with (out / "actions.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self.log:
                fh.write(json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n")
        (out / "events.json").write_text(json.dumps(self.world.events, default=_json_default,
                                                    ensure_ascii=False, indent=0), encoding="utf-8")
        (out / "report.json").write_text(json.dumps(self.sink.folded(), default=_json_default,
                                                    ensure_ascii=False, indent=1), encoding="utf-8")


# ── re-checking an episode ──────────────────────────────────────────────────
def replay(run_dir: str | Path) -> Instrument:
    """Rebuild an episode from its ledger: same scenario and seed, the same calls in order."""
    run_dir = Path(run_dir)
    ep = json.loads((run_dir / "episode.json").read_text(encoding="utf-8"))
    inst = Instrument(ep["scenario_id"], int(ep["seed"]))
    for line in (run_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        kind = rec.get("kind")
        if kind == "act":
            inst.act(rec["name"], **rec["params"])
        elif kind == "report":
            a = dict(rec["args"])
            inst.report(a.pop("claim_id"), a.pop("value"), **a)
        elif kind == "diagnosis":
            inst.report_tip_state(**rec["args"])
        elif kind == "finish":
            break
    inst.finish()
    return inst


def verify(run_dir: str | Path) -> dict:
    """Replay an episode and compare what was measured and how it was judged."""
    run_dir = Path(run_dir)
    ep = json.loads((run_dir / "episode.json").read_text(encoding="utf-8"))
    inst = replay(run_dir)
    again = inst.digest()
    want = ep.get("digest") or {}
    diffs = []
    for part in ("frames", "spectra"):
        if again.get(part) != want.get(part):
            diffs.append(part)
    if again.get("events") != want.get("events"):
        diffs.append("events")
    s = inst._summary()
    same_verdict = (bool(s["success"]) == bool(ep.get("success"))
                    and abs(float(s["partial"]) - float(ep.get("partial", -1))) < 1e-12)
    if not same_verdict:
        diffs.append("verdict")
    return {"ok": not diffs, "differences": diffs, "success": s["success"], "partial": s["partial"],
            "recorded": {"success": ep.get("success"), "partial": ep.get("partial")}}


# ── helpers ─────────────────────────────────────────────────────────────────
def _portable_events(events: list[dict]) -> list[dict]:
    """Events without what depends on where the run directory is (file paths → file names)."""
    out = []
    for e in events:
        d = dict(e)
        if d.get("path"):
            d["path"] = Path(str(d["path"])).name
        out.append(d)
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=_json_default))


__all__ = ["ACTIONS", "API_VERSION", "ActionError", "ActionSpec", "Instrument", "Param", "replay", "validate",
           "verify", "fold_results"]
