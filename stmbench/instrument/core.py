"""The benchmark instrument: primitive actions on stmsim, on a virtual clock.

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
* **reports** (``report``, ``finish``) - the result channel of :mod:`stmbench.trackB.results`,
  judged by the rules the rule card shows (:data:`~stmbench.trackB.rules.RULES_VERSION`).

What is deliberately absent: composite skills, a safety gate, a default strength for tip
conditioning (the width, depth and dwell of a pulse or poke are the operator's to choose), and
any call that judges nothing, such as a statement about the tip: no scenario scores it, and
offering it would only tell an agent to worry about its tip. An action does what the instrument
would do, hard limits only (bias, piezo range, preamplifier range); a poke too deep, a pulse too
high, a coarse step in tunnelling have their physical consequences.
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

from stmsim.physics.scanner import SIG_CURRENT, SIG_LI_X, SIG_Z
from stmsim.scenario import Scenario

from ..trackB.results import ResultSink, fold_results
from ..trackB.rules import OVERRUN_TOLERANCE, RULES_VERSION, task_text

#: version of the action set, the observations and the log format. 2: no tip-diagnosis call (so no
#: ``diagnosis`` record or ledger key either), and ``pulse.width_ms``, ``poke.depth_nm`` and
#: ``poke.dwell_s`` are required. A ledger or an export of another version is refused by
#: :func:`replay`, :func:`verify` and the grader (:func:`require_api_version`): its calls and
#: records are not this version's, so only the code of its own version can re-check it.
API_VERSION = 2
SCENARIO_DIR = Path(__file__).resolve().parents[1] / "trackB" / "scenarios"
MAX_ACTIONS = 3000
#: success kinds the instrument can judge from what it records: the paper scenarios'. The
#: skill scenarios (B*) judge on evidence this instrument does not collect, so they are
#: refused at the start rather than failing at the end of an episode.
JUDGED_KINDS = ("claims_verified",)
_GAIN_FULL_SCALE_NA = {10000.0: 0, 1000.0: 1, 100.0: 2, 10.0: 3, 1.0: 4, 0.1: 5}


class ActionError(ValueError):
    """A refused call: nothing happened on the instrument and no time passed."""


class VersionError(ValueError):
    """A ledger or an export written under another API version than this code's: it needs the code
    of its own version (see :func:`require_api_version`)."""


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


# pulse.width_ms, poke.depth_nm and poke.dwell_s are required, with no default: any default would be
# the answer to how hard to condition the tip (in a screen-only trial one default poke mended a
# double tip at the first try)
ACTIONS: dict[str, ActionSpec] = {s.name: s for s in (
    ActionSpec("scan", "Image a square frame in constant-current mode and save it. Returns the frame's "
                       "file name, which reports cite. Takes pixels × 2 × line_time_s of instrument time.",
               (_P("center_x_nm", "number", "frame centre x (scan coordinates, nm)", True),
                _P("center_y_nm", "number", "frame centre y (scan coordinates, nm)", True),
                _P("size_nm", "number", "side of the square frame (nm)", True, lo=0.5, hi=2000.0),
                _P("pixels", "integer", "pixels per line = lines", default=256, lo=16, hi=1024),
                _P("angle_deg", "number", "scan rotation, clockwise positive (deg)", default=0.0, lo=-180.0, hi=180.0),
                _P("line_time_s", "number", "time per line, each way (s)", default=0.586, lo=0.02, hi=60.0),
                _P("direction", "string", "slow-axis direction", default="down", choices=("down", "up")))),
    ActionSpec("set_bias", "Set the sample bias.", (_P("bias_v", "number", "bias (V)", True, lo=-10.0, hi=10.0),)),
    ActionSpec("set_setpoint", "Set the tunnelling-current setpoint of the feedback loop; the loop settles.",
               (_P("current_pa", "number", "setpoint current (pA)", True, lo=0.1, hi=1.0e5),)),
    ActionSpec("feedback", "Switch the Z feedback loop on or off (off holds Z where it is).",
               (_P("on", "boolean", "true = on", True),)),
    ActionSpec("set_current_range", "Select the preamplifier range (full-scale current).",
               (_P("full_scale_na", "number", "full scale (nA)", True, choices=tuple(_GAIN_FULL_SCALE_NA)),)),
    ActionSpec("move_tip", "Move the tip laterally in a straight line at a set speed, recording Z and the "
                           "current along the path. Returns the file name of the saved trace.",
               (_P("x_nm", "number", "target x (scan coordinates, nm)", True),
                _P("y_nm", "number", "target y (scan coordinates, nm)", True),
                _P("speed_nm_s", "number", "lateral speed (nm/s)", default=293.0, lo=0.01, hi=10000.0))),
    ActionSpec("pulse", "Apply a bias pulse at the tip's position (tip conditioning).",
               (_P("bias_v", "number", "pulse voltage (V, absolute)", True, lo=-10.0, hi=10.0),
                _P("width_ms", "number", "pulse width (ms)", True, lo=0.1, hi=5000.0),
                _P("z_hold", "boolean", "hold Z during the pulse", default=True))),
    ActionSpec("poke", "Indent the tip into the surface and retract (tip conditioning by controlled contact).",
               (_P("depth_nm", "number", "indentation depth below the tunnelling height (nm)", True, lo=0.0, hi=20.0),
                _P("bias_v", "number", "bias during the indentation (V); omit to keep the current bias", lo=-10.0, hi=10.0),
                _P("dwell_s", "number", "time held at depth (s)", True, lo=0.0, hi=10.0))),
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
    ActionSpec("lockin", "Switch the bias modulation and lock-in on or off. On, every scan also records a "
                         "dI/dV map at the scan bias (channel 'didv') - the sample's local density of states at "
                         "that energy; bias sweeps give dI/dV either way.",
               (_P("on", "boolean", "true = modulating", True),
                _P("amplitude_mv", "number", "modulation amplitude, zero to peak (mV)", default=10.0, lo=0.1, hi=200.0))),
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


def _peak(a) -> float:
    """The largest |value| of a curve, NaN-safe (0 for a curve with no finite point)."""
    a = np.asarray(a, float)
    return float(np.nanmax(np.abs(a))) if np.any(np.isfinite(a)) else 0.0


def array_digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


#: Relative tolerance of a fingerprint comparison (scale: the larger of |min|, |max| of the recorded
#: finite values). Chosen from two measurements (docs/DESIGN.md, "Export version 2"):
#:  - rounding: trial 6 and a P2 and a P5 episode recorded on Windows (i5-14400F) and replayed on
#:    Linux (EPYC 9654), numpy 2.5.4 / OpenBLAS 0.3.34, 2026-11-16 (also under OPENBLAS_CORETYPE=Haswell
#:    and without AVX-512): the largest relative fingerprint difference of any array is 4.2e-12;
#:  - real divergence (same Windows host, 2026-11-16): trial-6 records with a drag end moved one lattice
#:    site, a scan centre moved 0.1 nm (two scans), seed 10 instead of 9, and a 1 s pause inserted: in
#:    each, the largest difference over the episode's arrays is >= 2.6e-3 (the first changed file, >= 1.5e-3).
#: An episode is inconsistent when ANY array exceeds the tolerance, so the first must stay a factor
#: >= 1e3 below RTOL (4.2e-12 -> 2400x) and the second a factor >= 1e3 above it (>= 1e5x). A single
#: array that hardly responds to a change (the current at constant current) may be below RTOL.
RTOL = 1e-8
#: an event's float below this (in its own unit) is not told apart from 0: far below anything physical
EVENT_ATOL = 1e-15

_BLOCKS_2D = 8          # block means of a frame: 8 x 8 blocks, whatever the shape
_SEGMENTS_1D = 32       # block means of a curve: min(32, n) segments


def _mean_or_none(x: np.ndarray) -> float | None:
    f = x[np.isfinite(x)]
    return float(f.mean()) if f.size else None


def array_fingerprint(a: np.ndarray) -> dict:
    """A coarse, comparable summary of one array, for telling a last-bits rounding difference
    (another libm or BLAS kernel) from a different measurement: the shape, the count and hash of the
    non-finite mask (exact), and over the finite values their min, max, mean, std and block means on
    a fixed grid (2-D: 8 x 8 blocks, anything else flattened into min(32, n) segments; a block with
    no finite value is None). Numbers are Python floats, so they survive JSON exactly."""
    x = np.asarray(a, dtype=np.float64)
    mask = np.isfinite(x)
    fin = x[mask]
    if x.ndim == 2:
        blocks = [_mean_or_none(b) for rows in np.array_split(x, _BLOCKS_2D, axis=0)
                  for b in np.array_split(rows, _BLOCKS_2D, axis=1)]
    else:
        flat = x.ravel()
        blocks = [_mean_or_none(b) for b in np.array_split(flat, min(_SEGMENTS_1D, flat.size))] if flat.size else []
    return {"shape": list(x.shape), "nonfinite": int(mask.size - fin.size),
            "nonfinite_mask": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
            "min": float(fin.min()) if fin.size else None, "max": float(fin.max()) if fin.size else None,
            "mean": float(fin.mean()) if fin.size else None, "std": float(fin.std()) if fin.size else None,
            "blocks": blocks}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_fingerprints(want, got) -> tuple[str, float | None]:
    """The recorded fingerprint against the replay's: ``("rounding", r)`` when the structure is equal
    and every number is within :data:`RTOL` of the recorded scale (``r`` the largest relative
    difference), ``("beyond", r)`` when a number is further, ``("structure", None)`` when shape,
    non-finite mask, which blocks are empty, or the form of the record differ. A recorded scale of 0
    allows no difference at all."""
    try:
        if not isinstance(want, dict) or set(want) != set(got):
            return "structure", None
        for k in ("shape", "nonfinite", "nonfinite_mask"):
            if want[k] != got[k]:
                return "structure", None
        wb, gb = want["blocks"], got["blocks"]
        if not isinstance(wb, list) or len(wb) != len(gb):
            return "structure", None
        pairs = [(want[k], got[k]) for k in ("min", "max", "mean", "std")] + list(zip(wb, gb))
        for w, g in pairs:
            if (w is None) != (g is None) or not (w is None or _num(w)):
                return "structure", None
        lo, hi = want["min"], want["max"]
        scale = max(abs(lo), abs(hi)) if lo is not None else 0.0
        worst = 0.0
        for w, g in pairs:
            if w is None:
                continue
            d = abs(float(w) - float(g))
            rel = d / scale if scale > 0 else (0.0 if d == 0 else float("inf"))
            worst = max(worst, rel)
        return ("rounding" if worst <= RTOL else "beyond"), worst
    except (KeyError, TypeError, ValueError):
        return "structure", None


def events_differ(want, got) -> str | None:
    """Why two event lists (JSON-shaped, paths already reduced) differ beyond rounding, or None.
    Same length, keys, strings, booleans and integers exactly; floats within :data:`RTOL` relative
    (to the larger magnitude of the two) or :data:`EVENT_ATOL` absolute."""
    if not isinstance(want, list) or not isinstance(got, list) or len(want) != len(got):
        return "the event count differs"
    for k, (a, b) in enumerate(zip(want, got)):
        why = _value_differs(a, b)
        if why:
            return f"event {k}: {why}"
    return None


def _value_differs(a, b) -> str | None:
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return "the keys differ"
        for key in a:
            why = _value_differs(a[key], b[key])
            if why:
                return f"{key}: {why}"
        return None
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return "a list differs in length"
        for x, y in zip(a, b):
            why = _value_differs(x, y)
            if why:
                return why
        return None
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (a != a and b != b):
            return None
        d = abs(a - b)
        return None if d <= max(EVENT_ATOL, RTOL * max(abs(a), abs(b))) else "a float differs beyond rounding"
    return None if (a == b and type(a) is type(b)) else "a value differs"


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
    didv_fwd: np.ndarray | None = None       # lock-in X (A) when the frame was a dI/dV map
    didv_bwd: np.ndarray | None = None


class Instrument:
    """One benchmark episode on the simulated instrument (see the module docstring)."""

    def __init__(self, scenario: Scenario | str | Path, seed: int, out_dir: str | Path | None = None, *,
                 max_actions: int = MAX_ACTIONS):
        if not isinstance(scenario, Scenario):
            p = Path(scenario)
            scenario = Scenario.load(p if p.exists() else SCENARIO_DIR / f"{scenario}.yaml")
        kind = (scenario.success or {}).get("kind", "claims_verified")
        if kind not in JUDGED_KINDS:
            raise ValueError(f"{scenario.id}: the instrument judges {', '.join(JUDGED_KINDS)} scenarios; "
                             f"this one's success kind is {kind!r}")
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
        # the world as the episode found it, for the ledger only (a replay rebuilds and checks
        # the sample against it, and starts the tip's story from it); reading it draws nothing
        self.truth_before = self.world.truth()
        b = scenario.budget or {}
        self.max_sim_s = float(b["sim_hours"]) * 3600.0 if b.get("sim_hours") else None
        self.max_actions = int(max_actions)
        self.log: list[dict] = []
        self.sink = ResultSink()
        self.frames: dict[str, FrameRecord] = {}
        self.spectra: dict[str, dict] = {}
        self.n_actions = 0
        self.finished = False
        self.verdict: Any = None

    # ── what the agent is told ──
    def task(self, lang: str = "zh") -> str:
        """The task with its rule card (stmbench.trackB.rules), in ``lang``: "zh" or "en"."""
        return task_text(self.scenario, lang)

    @staticmethod
    def manual() -> str:
        """The action set, one line per parameter with its default, its choices and its range (min, max): what every
        agent is told the instrument does."""
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
                # the range an MCP agent's input schema carries as minimum and maximum (the screen agent of the
                # P4 trials learnt the 100000 pA setpoint limit only from a refusal)
                if p.lo is not None:
                    extra.append(f"min {p.lo}")
                if p.hi is not None:
                    extra.append(f"max {p.hi}")
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

    def _settle_tunnelling(self, at_least: float = 0.0, *, current_a: float | None = None,
                           bias_v: float | None = None) -> None:
        """:meth:`_settle` for an action that leaves the tip tunnelling: the time it takes also
        counts for the tip's hazards outside scans (:meth:`World.tunnel_interval`; ``current_a``
        and ``bias_v`` are the dose and field of a sweep, else the setpoint and the bias)."""
        dt = max(float(at_least), self._transient_left_s())
        self.world.tunnel_interval(dt, current_a=current_a, bias_v=bias_v)
        self._advance(dt)

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
        try:
            out = getattr(self, f"_do_{name}")(**params)
        except ActionError as exc:
            # the instrument's own checks (piezo range, a withdrawn tip, no qPlus sensor) refuse
            # before anything moves: a refusal like the ones above, and on the record as one
            self.log.append({"i": len(self.log), "kind": "refused", "name": name, "args": _jsonable(args),
                             "error": str(exc), "t_s": _r(self.used_s, 6)})
            raise
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
        # a frame counts its rows as floor(elapsed / row time): one microsecond past the nominal
        # duration keeps floating-point rounding from leaving the last row unacquired (NaN)
        self._advance(st.frame_time_s + 1e-6)
        w.scan_running()                                  # reveals the last rows, ends the frame
        if not fr.finished:
            raise RuntimeError(f"scan: frame ended with {fr.rows_done}/{fr.rows_total} rows")
        path = w.save_frame()
        name = Path(path).name
        order = np.arange(st.ny) if direction == "down" else np.arange(st.ny)[::-1]
        z_f, z_b = fr.data[SIG_Z]
        i_f, i_b = fr.data[SIG_CURRENT]
        d_f, d_b = fr.data.get(SIG_LI_X, (None, None))
        geom = {"center_nm": [_r(center_x_nm), _r(center_y_nm)], "size_nm": _r(size_nm),
                "pixels": int(pixels), "angle_deg": _r(angle_deg), "direction": direction,
                "nm_per_px": _r(size_nm / pixels, 6), "line_time_s": _r(line_time_s, 6)}
        self.frames[name] = FrameRecord(
            name=name, geometry=geom,
            # backward lines are acquired right to left: flip them into image columns
            z_fwd=np.asarray(z_f)[order], z_bwd=np.asarray(z_b)[order][:, ::-1],
            i_fwd=np.asarray(i_f)[order], i_bwd=np.asarray(i_b)[order][:, ::-1],
            t_start_s=t_start, meta={"bias_v": w.bias_v, "setpoint_pa": w.zctrl.setpoint_a * 1e12},
            didv_fwd=None if d_f is None else np.asarray(d_f)[order],
            didv_bwd=None if d_b is None else np.asarray(d_b)[order][:, ::-1])
        channels = ["z", "current"] + (["didv"] if d_f is not None else [])
        return {"frame": name, **geom, "channels": channels, "duration_s": _r(st.frame_time_s, 3)}

    def _do_set_bias(self, bias_v) -> dict:
        self.world.set_bias(float(bias_v))
        self._settle_tunnelling(0.05)
        return {"bias_v": _r(bias_v, 6)}

    def _do_set_setpoint(self, current_pa) -> dict:
        self.world.set_setpoint(float(current_pa) * 1e-12)
        self._settle_tunnelling(0.05)
        return {"setpoint_pa": _r(current_pa, 6)}

    def _do_feedback(self, on) -> dict:
        self.world.zctrl_set(bool(on))
        self._settle_tunnelling(0.05)
        return {"feedback": bool(on)}

    def _do_set_current_range(self, full_scale_na) -> dict:
        w = self.world
        w.set_current_range(_GAIN_FULL_SCALE_NA[float(full_scale_na)])      # the world checks the setpoint against it
        self._settle_tunnelling(0.05)
        return {"full_scale_na": float(full_scale_na)}

    def _do_move_tip(self, x_nm, y_nm, speed_nm_s) -> dict:
        from stmsim.io.dat_writer import save_move_trace

        w = self.world
        x, y = x_nm * 1e-9, y_nm * 1e-9
        rx, ry = w.rig.xy_range_m
        if abs(x) > rx / 2 or abs(y) > ry / 2:
            raise ActionError("move_tip: outside the piezo range")
        dist = math.hypot(x - w.tip_x, y - w.tip_y)
        t_start = self.used_s
        w.move_xy_path(x, y, speed_m_s=float(speed_nm_s) * 1e-9, trace=True)
        # what the controller recorded along the path, kept like a spectrum: an operator reads
        # whether atoms hopped from it, and the instrument does not say
        tr = w.last_move_trace
        path = save_move_trace(w, "", tr)
        w.dats_saved.append(path)
        w._event("dat_saved", path=path, experiment="move trace", idx=len(w.dats_saved), move_idx=tr["index"])
        name = Path(path).name
        self.spectra[name] = {"name": name, "kind": "move", "index": int(tr["index"]),
                              "tip_nm": [_r(x_nm), _r(y_nm)], "from_nm": [_r(v) for v in tr["from_nm"]],
                              "to_nm": [_r(x_nm), _r(y_nm)], "speed_nm_s": _r(speed_nm_s, 6),
                              "bias_v": _r(tr["bias_v"], 6), "setpoint_pa": _r(tr["setpoint_pa"], 6),
                              "feedback": bool(tr["feedback"]), "t_s": t_start,
                              "t": np.asarray(tr["t"], float), "dist": np.asarray(tr["dist"], float),
                              "z_abs": np.asarray(tr["z_abs"], float), "i": np.asarray(tr["i"], float)}
        self._settle(dist / (float(speed_nm_s) * 1e-9) + 0.05)
        return {"tip_nm": [_r(x_nm), _r(y_nm)], "distance_nm": _r(dist * 1e9), "trace": name,
                "duration_s": _r(float(tr["t"][-1]), 3)}

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
        # the sweep's hazards: its largest field and its largest current dose (the measurement
        # above is the tip it started with; a change inside the sweep shows from the next one on)
        w.tunnel_interval(duration, current_a=_peak(cur["i"]), bias_v=max(abs(float(v_start)), abs(float(v_end))))
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
                              # the loop's Z at every point (m), as the .dat file has it: what lines up
                              # a curve over an atom with one over the bare surface
                              "z_abs": np.asarray(cur["z_n"], float), "t_s": self.used_s}
        self._settle_tunnelling(duration, current_a=_peak(cur["i"]))
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
        ring = 5.0 * p.tau_s if on else 0.05
        w.tunnel_interval(ring)                          # the tip tunnels while the oscillation rings up
        self._advance(ring)
        # the frequency shift is re-zeroed where the tip is (PLL.FreqShiftAutoCenter)
        if on and not w.withdrawn:
            amp = p.amplitude_now(w.clock.wall())
            gap = w.current_z_tip() - w.surface_height_here()
            p.center_freq_hz = float(w.tip.qplus_f0_hz + w.df_physical(gap, amp))
        return {"oscillating": bool(on), "amplitude_setpoint_pm": _r(p.amp_setpoint_m * 1e12)}

    def _do_lockin(self, on, amplitude_mv) -> dict:
        w = self.world
        w.lockin_on = bool(on)
        w.lockin_amp_v = float(amplitude_mv) * 1e-3
        chans = [c for c in w.scan.channels if c != SIG_LI_X]
        w.scan.channels = chans + ([SIG_LI_X] if on else [])
        self._settle_tunnelling(0.05)
        return {"lockin": bool(on), "amplitude_mv": _r(amplitude_mv, 6)}

    def _do_drift_comp(self, on, vx_nm_per_min, vy_nm_per_min, vz_pm_per_min) -> dict:
        self.world.set_drift_comp(bool(on), vx_nm_per_min * 1e-9 / 60.0, vy_nm_per_min * 1e-9 / 60.0,
                                  vz_pm_per_min * 1e-12 / 60.0)
        self._settle_tunnelling(0.05)
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
        self.world.tunnel_interval(float(seconds))
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
        dv = w.drift_comp_v                      # m/s as set_drift_comp stored them: what was programmed, not the drift
        return {"instrument_time_s": _r(self.used_s, 3), "budget_s": self.max_sim_s,
                "bias_v": _r(w.bias_v, 6), "setpoint_pa": _r(w.zctrl.setpoint_a * 1e12, 4),
                "feedback": bool(w.zctrl.on), "withdrawn": bool(w.withdrawn),
                "z_nm": _r(z * 1e9, 5), "current_pa": _r(i * 1e12, 4),
                "tip_nm": [_r(w.tip_x * 1e9), _r(w.tip_y * 1e9)],
                "current_full_scale_na": _r(w.preamp.full_scale_a * 1e9, 4),
                "oscillating": bool(p.output_on) if p is not None else None,
                "lockin": bool(getattr(w, "lockin_on", False)),
                # shaped like the drift_comp action's reply, so a reading can be compared with what was set
                "drift_comp": {"on": bool(w.drift_comp_on), "v_nm_per_min": [_r(dv[0] * 6e10), _r(dv[1] * 6e10)],
                               "vz_pm_per_min": _r(dv[2] * 6e13)},
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
                             "points": int(len(s.get("v", s.get("z_rel", s.get("t", []))))), "t_s": _r(s["t_s"], 3)}
                            for s in self.spectra.values()]}

    def frame(self, name: str, channel: str = "z", direction: str = "forward") -> dict:
        """A saved frame's image (row 0 = top) and geometry. ``channel``: z (m), current (A), or
        didv (lock-in X, A) when the frame was taken with the lock-in on."""
        self._observe("frame", name=name, channel=channel, direction=direction)
        f = self.frames.get(Path(str(name)).name)
        if f is None:
            raise ActionError(f"no frame {name!r}")
        arrs = {("z", "forward"): f.z_fwd, ("z", "backward"): f.z_bwd,
                ("current", "forward"): f.i_fwd, ("current", "backward"): f.i_bwd}
        if f.didv_fwd is not None:
            arrs.update({("didv", "forward"): f.didv_fwd, ("didv", "backward"): f.didv_bwd})
        if (channel, direction) not in arrs:
            raise ActionError(f"{f.name} has channels {sorted({c for c, _ in arrs})}; "
                              "direction is forward or backward")
        unit = "m" if channel == "z" else "A"
        return {"frame": f.name, **f.geometry, "channel": channel, "direction": direction, "unit": unit,
                "t_start_s": f.t_start_s,          # instrument time the frame began: drift = shift / Δt
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

    # ── the end ──
    def finish(self) -> dict:
        """Judge the episode (the rules of ``RULES_VERSION``, the budget overrun rule) and write the ledger."""
        if self.finished:
            return self._summary()
        from ..trackB.rules import apply_budget_overrun
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

    def frame_digest(self, name: str) -> dict:
        """Hashes of one saved frame's arrays (the frame's entry in :meth:`digest`)."""
        f = self.frames[name]
        return {k: array_digest(getattr(f, k)) for k in ("z_fwd", "z_bwd", "i_fwd", "i_bwd", "didv_fwd", "didv_bwd")
                if getattr(f, k) is not None}

    def spectrum_digest(self, name: str) -> dict:
        """Hashes of one saved spectrum's arrays (the spectrum's entry in :meth:`digest`)."""
        return {k: array_digest(v) for k, v in self.spectra[name].items() if isinstance(v, np.ndarray)}

    def frame_fingerprint(self, name: str) -> dict:
        """Fingerprints (:func:`array_fingerprint`) of one saved frame's arrays."""
        f = self.frames[name]
        return {k: array_fingerprint(getattr(f, k)) for k in ("z_fwd", "z_bwd", "i_fwd", "i_bwd", "didv_fwd", "didv_bwd")
                if getattr(f, k) is not None}

    def spectrum_fingerprint(self, name: str) -> dict:
        """Fingerprints of one saved spectrum's (or move trace's) arrays."""
        return {k: array_fingerprint(v) for k, v in self.spectra[name].items() if isinstance(v, np.ndarray)}

    def fingerprints(self) -> dict:
        """Fingerprints of everything measured: the ledger's rounding-tolerant companion of :meth:`digest`."""
        return {"frames": {n: self.frame_fingerprint(n) for n in self.frames},
                "spectra": {n: self.spectrum_fingerprint(n) for n in self.spectra}}

    def digest(self) -> dict:
        """Hashes of everything measured, for re-checking an episode by replay."""
        return {"frames": {n: self.frame_digest(n) for n in self.frames},
                "spectra": {n: self.spectrum_digest(n) for n in self.spectra},
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
              "digest": self.digest(), "fingerprints": self.fingerprints(),
              "sim_t0_s": self.t0,
              "truth_before": self.truth_before, "truth_after": self.truth_after}
        (out / "episode.json").write_text(json.dumps(ep, default=_json_default, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        from ..replay.record import tip_timeline

        (out / "tip_timeline.json").write_text(
            json.dumps(tip_timeline(self.world, initial=(self.truth_before or {}).get("tip")),
                       default=_json_default, ensure_ascii=False, indent=1), encoding="utf-8")
        with (out / "actions.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self.log:
                fh.write(json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n")
        (out / "events.json").write_text(json.dumps(self.world.events, default=_json_default,
                                                    ensure_ascii=False, indent=0), encoding="utf-8")
        (out / "report.json").write_text(json.dumps(self.sink.folded(), default=_json_default,
                                                    ensure_ascii=False, indent=1), encoding="utf-8")


# ── re-checking an episode ──────────────────────────────────────────────────
_ABSENT = object()


def _shown(value: Any, limit: int = 40) -> str:
    """A value read out of a ledger or an export as a message quotes it: one line, printable, cut short."""
    s = "".join(c if c.isprintable() else "?" for c in json.dumps(value, ensure_ascii=False, default=str))
    return s if len(s) <= limit else s[:limit - 3] + "..."


def require_api_version(found: Any, what: str) -> None:
    """Pass when ``found``, the ``api_version`` a ledger or an export carries, is this code's
    :data:`API_VERSION`; otherwise :class:`VersionError` naming ``what`` is refused.

    The calls an agent can make, their parameters and the kinds of record in a log change with the
    version, so a record of another one would replay into a confusing mismatch, or grade as a zero,
    instead of into its own episode. It needs the code of its own version, and the message says so."""
    if isinstance(found, int) and not isinstance(found, bool) and found == API_VERSION:
        return
    state = "carries no api_version" if found is None else f"is of api_version {_shown(found)}"
    raise VersionError(f"{what} {state}; this code is api_version {API_VERSION}. A record replays only under "
                       "the code of its own API version, so this one needs the code that wrote it "
                       "(it cannot be replayed, verified or graded here)")


def replay_records(scenario: Scenario | str | Path, seed: int, records: list[dict], *,
                   out_dir: str | Path | None = None,
                   raise_refused: bool = False) -> tuple[Instrument, list[dict]]:
    """Rebuild an episode from its log records: the same scenario and seed, the same calls in order.

    ``act`` records are run again (their ``params``), ``report`` records sent again (their
    ``args``); ``refused`` and ``observe`` records changed nothing and are skipped. Any other kind
    (a ``diagnosis`` record, say, which no instrument of this API version writes) is a mismatch.
    The records end at ``finish``, and the instrument is then finished: judged as it stands,
    which is also how an episode that never reached ``finish`` is judged. What the records say
    was measured or judged is never used, only what was done.

    Each act's reply, and the instrument times around it, are compared with the record's, since
    the same seed and calls only measure the same thing in the same numerical environment. What
    differs comes back with the instrument, as ``{"i", "name", "field"}`` for a reply or a time
    that differs and ``{"i", "name", "error"}`` for a call the replay refused or a record it could
    not read: a divergence is reported, not raised. ``raise_refused`` raises a refused call's
    :class:`ActionError` instead (what :func:`replay` does)."""
    inst = Instrument(scenario, int(seed), out_dir)
    mismatches: list[dict] = []
    over = False
    for rec in records:
        if not isinstance(rec, dict):
            mismatches.append({"i": None, "name": None, "error": "a record is not an object"})
            continue
        kind, i = rec.get("kind"), rec.get("i")
        if kind == "finish":
            over = True
        elif over:
            # after the end the instrument refuses actions and reports, and logs a refusal, not an
            # act or a report
            if kind in ("act", "report"):
                mismatches.append({"i": i, "name": rec.get("name") if kind == "act" else kind,
                                   "error": "a call after finish"})
        elif kind == "act":
            _replay_act(inst, rec, mismatches, raise_refused)
        elif kind == "report":
            _replay_report(inst, rec, mismatches, raise_refused)
        elif kind not in ("refused", "observe"):
            mismatches.append({"i": i, "name": str(kind), "error": "unknown record kind"})
    inst.finish()
    return inst, mismatches


def _replay_act(inst: Instrument, rec: dict, mismatches: list[dict], raise_refused: bool) -> None:
    i, name, params = rec.get("i"), rec.get("name"), rec.get("params")
    if not isinstance(name, str) or not isinstance(params, dict) or "name" in params:
        mismatches.append({"i": i, "name": name if isinstance(name, str) else None,
                           "error": "malformed act record"})
        return
    try:
        inst.act(name, **params)
    except ActionError as exc:
        if raise_refused:
            raise
        mismatches.append({"i": i, "name": name, "error": str(exc)})
        return
    done = inst.log[-1]            # what the act just logged: its result as the log stores it, and its times
    want = rec.get("result")
    if isinstance(want, dict):
        fields = [k for k in sorted(set(done["result"]) | set(want))
                  if done["result"].get(k, _ABSENT) != want.get(k, _ABSENT)]
    else:
        fields = ["result"]
    fields += [k for k in ("t0_s", "t1_s") if done[k] != rec.get(k)]
    mismatches.extend({"i": i, "name": name, "field": k} for k in fields)


def _replay_report(inst: Instrument, rec: dict, mismatches: list[dict], raise_refused: bool) -> None:
    args = rec.get("args")
    try:
        if not isinstance(args, dict) or "claim_id" not in args or "value" not in args:
            raise ValueError("malformed report record")
        a = dict(args)
        inst.report(a.pop("claim_id"), a.pop("value"), **a)
    except ActionError as exc:
        if raise_refused:
            raise
        mismatches.append({"i": rec.get("i"), "name": "report", "error": str(exc)})
    except (TypeError, ValueError) as exc:      # arguments the instrument cannot take at all
        mismatches.append({"i": rec.get("i"), "name": "report", "error": str(exc)})


def replay(run_dir: str | Path) -> Instrument:
    """Rebuild an episode from its ledger: same scenario and seed, the same calls in order.

    :class:`VersionError` for a ledger of another API version: only the code of its own version can
    replay it."""
    run_dir = Path(run_dir)
    ep = json.loads((run_dir / "episode.json").read_text(encoding="utf-8"))
    require_api_version(ep.get("api_version"), f"the ledger in {run_dir}")
    records = [json.loads(line) for line in (run_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    return replay_records(ep["scenario_id"], int(ep["seed"]), records, raise_refused=True)[0]


def verify(run_dir: str | Path) -> dict:
    """Replay an episode and compare what was measured and how it was judged. A ledger of another
    API version is refused, as :func:`replay` refuses it (:class:`VersionError`).

    An array whose hash differs is compared by its fingerprint when the ledger has one (the grader's
    rule: :func:`compare_fingerprints`), and events are compared by value (:func:`events_differ`),
    so a last-bits difference between machines does not fail the check. A ledger without
    fingerprints is checked exactly. ``rounding`` lists the arrays that differed by rounding only,
    with their largest relative difference."""
    run_dir = Path(run_dir)
    ep = json.loads((run_dir / "episode.json").read_text(encoding="utf-8"))
    inst = replay(run_dir)
    again = inst.digest()
    want = ep.get("digest") or {}
    fps_want = ep.get("fingerprints")
    fps_got = inst.fingerprints() if isinstance(fps_want, dict) else None
    diffs: list[str] = []
    rounding: list[dict] = []
    for part in ("frames", "spectra"):
        bad = False
        for name in sorted(set(again.get(part) or {}) | set(want.get(part) or {})):
            w, g = (want.get(part) or {}).get(name), (again.get(part) or {}).get(name)
            if w == g:
                continue
            if not isinstance(w, dict) or not isinstance(g, dict) or fps_got is None:
                bad = True
                continue
            for arr in sorted(set(w) | set(g)):
                if w.get(arr) == g.get(arr):
                    continue
                fw = ((fps_want.get(part) or {}).get(name) or {}).get(arr)
                fg = (fps_got[part].get(name) or {}).get(arr)
                kind, rel = compare_fingerprints(fw, fg) if fw is not None and fg is not None else ("structure", None)
                if kind == "rounding":
                    rounding.append({part[:-1]: name, "array": arr, "max_rel_diff": rel})
                else:
                    bad = True
        if bad:
            diffs.append(part)
    if again.get("events") != want.get("events"):
        events_path = run_dir / "events.json"
        if fps_got is None or not events_path.exists():
            diffs.append("events")
        else:
            recorded = _jsonable(_portable_events(json.loads(events_path.read_text(encoding="utf-8"))))
            if events_differ(recorded, _jsonable(_portable_events(inst.world.events))) is not None:
                diffs.append("events")
    s = inst._summary()
    same_verdict = (bool(s["success"]) == bool(ep.get("success"))
                    and abs(float(s["partial"]) - float(ep.get("partial", -1))) < 1e-12)
    if not same_verdict:
        diffs.append("verdict")
    return {"ok": not diffs, "differences": diffs, "rounding": rounding, "success": s["success"],
            "partial": s["partial"], "recorded": {"success": ep.get("success"), "partial": ep.get("partial")}}


# ── helpers ─────────────────────────────────────────────────────────────────
def _portable_events(events: list[dict]) -> list[dict]:
    """Events without what depends on where the run directory is (file paths → file names)."""
    out = []
    for e in events:
        d = dict(e)
        if d.get("path"):
            # either separator: a ledger written on Windows is read on Linux, where Path() would keep
            # "D:\run\x.sxm" whole (the digest of its events was computed from the bare file name)
            d["path"] = str(d["path"]).replace("\\", "/").rsplit("/", 1)[-1]
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


__all__ = ["ACTIONS", "API_VERSION", "RTOL", "array_fingerprint", "compare_fingerprints", "events_differ", "ActionError", "ActionSpec", "Instrument", "Param", "VersionError", "replay",
           "replay_records", "require_api_version", "validate", "verify", "fold_results"]
