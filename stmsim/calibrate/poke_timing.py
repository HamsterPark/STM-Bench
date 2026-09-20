"""Poke (TipShapeWithReadback) timing and Δz from the rig's readback traces.

DESIGN.md §4.5 row "扎针 ①–③ 段时序/Δz": the simulator's ``world.tip_shaper_start`` plays a
four-segment Z/I trajectory whose lags today are literals (``hw_lag = 0.22 + U(0, 0.11)``).
This module measures them from 168 ``mast.readback_trace/1`` JSON traces recorded during a
calibration acquisition (under the configured trace directory) and writes
``$STM_BENCH_DATA/calib/poke_timing.json`` with medians / quantiles and ``n`` per quantity.

The four segments (the current channel pins them; see MAST ``io/z_trace.feedback_restored_t``)::

    ①  baseline        Z = z1,   I = setpoint          feedback ON
    ②  pressed −d      Z = z1−d, I saturated           feedback OFF, tip in the surface
    ③  back at z1      Z = z1,   I still saturated     Z ramped back, feedback NOT yet on
    ④  new equilibrium Z = z1+Δ, I back to setpoint    feedback restored

What is measured per trace (all times in trace seconds, Δz in pm relative to z1):

* ``plunge_lag_s``       — actual start of the ② ramp minus the *declared*
                           ``z_ramp_1_plunge.t_start`` (the hardware lag the world needs);
* ``plunge_ramp_s``      — actual ramp duration (declared ``lift_time_1_s``);
* ``press_hold_s``       — how long Z actually sits at −d (declared ``bias_settling_s``);
* ``retract_lag_s``      — actual start of the ③ ramp minus declared ``z_ramp_2_retract.t_start``;
* ``retract_end_lag_s``  — actual end of the ③ ramp minus declared ``z_ramp_2_retract.t_end``;
* ``depth_reached_frac`` — achieved excursion / commanded ``-tip_lift_m``;
* ``dz3_pm``             — Δz in segment ③ (by construction ≈ 0: it is the firmware putting Z
                           back where it was commanded; NOT a verdict);
* ``dz4_pm``             — Δz once the current has returned to the setpoint (segment ④); only
                           when the capture reached ④, else ``None`` with ``why`` = ``no_return``
                           (capture ended first) or ``no_press`` (current never rose);
* ``current_rise_lag_s`` — first current sample ≥ 3× baseline minus the actual plunge start
                           (readback lag; the current channel is a polled TCP value that holds
                           between updates, so this is an upper bound on the physical lag).

Ramp edges are located by 20 % / 80 % threshold crossings of the achieved excursion
(sustained for ``sustain_s``) and extrapolated linearly, so a 2 kHz trace gives ~1 ms edges.

Run (STM-Bench root)::

    python -m stmsim.calibrate.poke_timing [--traces DIR] [--out PATH]

``--traces`` defaults to ``$STM_BENCH_DATA/calibration_inputs/poke_traces``; pass it explicitly
to use another trace directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from stmsim.paths import calib_dir, data_path

SCHEMA = "mast.readback_trace/1"
CURRENT_RISE_FACTOR = 3.0        # |I| ≥ 3 × baseline ⇒ the tip is pressing
CURRENT_RETURN_FACTOR = 2.0      # |I| ≤ 2 × baseline after the rise ⇒ feedback back (④)
EDGE_LO, EDGE_HI = 0.2, 0.8      # threshold fractions used to locate ramp edges

NUMERIC_KEYS = ("plunge_lag_s", "plunge_ramp_s", "press_hold_s", "retract_lag_s", "retract_end_lag_s",
                "depth_reached_frac", "dz3_pm", "dz4_pm", "seg3_s", "seg4_s", "current_rise_lag_s",
                "capture_end_after_declared_s", "i_sat_over_i0", "z_noise_mad_pm")


def default_traces_dir() -> Path:
    return data_path("calibration_inputs", "poke_traces")


def load_trace(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        tr = json.load(f)
    if tr.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {tr.get('schema')!r}, expected {SCHEMA!r}")
    return tr


def declared_stages(trace: dict) -> dict[str, tuple[float, float | None]]:
    return {s["stage"]: (float(s["t_start"]), None if s.get("t_end") is None else float(s["t_end"]))
            for s in trace.get("stages", [])}


def _channel(trace: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    ch = trace["channels"][name]
    return np.asarray(ch["t_s"], float), np.asarray(ch["samples"], float)


def _first_sustained(mask: np.ndarray, start_idx: int, n_sustain: int) -> int | None:
    """Index of the first run of ≥ ``n_sustain`` consecutive True at or after ``start_idx``."""
    m = mask[start_idx:]
    if m.size < n_sustain:
        return None
    run = 0
    for k, v in enumerate(m):
        run = run + 1 if v else 0
        if run >= n_sustain:
            return start_idx + k - n_sustain + 1
    return None


def measure_segments(trace: dict, sustain_s: float = 0.005) -> dict:
    """Per-trace segment timing and Δz; see the module docstring for every key."""
    t, z = _channel(trace, "z")
    ti, cur = _channel(trace, "current")
    meta = trace.get("meta", {})
    st = declared_stages(trace)
    event_t = float(trace.get("event_t_s", st.get("pre_roll", (0.0, 0.0))[1] or 0.0))
    dt = float(np.median(np.diff(t))) if t.size > 1 else 5e-4
    n_sus = max(2, int(round(sustain_s / dt)))
    out: dict = {"file": trace.get("_file"), "verdict_recorded": meta.get("verdict"),
                 "depth_cmd_pm": -float(meta.get("tip_lift_m", 0.0)) * 1e12,
                 "declared": {k: list(v) for k, v in st.items()}, "why": ""}
    pre = t < event_t
    if pre.sum() < 5:
        out["why"] = "no_pre_roll"
        return out
    z1 = float(np.median(z[pre]))
    mad_pm = float(np.median(np.abs(z[pre] - z1))) * 1e12
    out["z_noise_mad_pm"] = mad_pm
    exc = (z1 - z) * 1e12                         # pm, positive = pressed toward the surface
    after = t >= event_t
    d_ach = float(np.percentile(exc[after], 99)) if after.any() else 0.0
    thr_noise = 6.0 * 1.4826 * mad_pm
    if d_ach < max(thr_noise, 5.0) or d_ach < EDGE_LO * out["depth_cmd_pm"]:
        out["why"] = "no_plunge"
        return out
    out["depth_reached_frac"] = d_ach / out["depth_cmd_pm"] if out["depth_cmd_pm"] > 0 else None
    lo, hi = EDGE_LO * d_ach, EDGE_HI * d_ach
    i0 = int(np.searchsorted(t, event_t))
    k20 = _first_sustained(exc >= lo, i0, n_sus)
    k80 = _first_sustained(exc >= hi, k20, n_sus) if k20 is not None else None
    if k20 is None or k80 is None or k80 <= k20:
        out["why"] = "no_plunge_edge"
        return out
    t20, t80 = float(t[k20]), float(t[k80])
    plunge_start = t20 - (t80 - t20) / (EDGE_HI - EDGE_LO) * EDGE_LO
    plunge_end = t80 + (t80 - t20) / (EDGE_HI - EDGE_LO) * (1.0 - EDGE_HI)
    out["plunge_start_s"], out["plunge_end_s"] = plunge_start, plunge_end
    out["plunge_ramp_s"] = plunge_end - plunge_start
    if "z_ramp_1_plunge" in st:
        out["plunge_lag_s"] = plunge_start - st["z_ramp_1_plunge"][0]
    # retract: excursion falls back through 80 % then 20 %
    r80 = _first_sustained(exc <= hi, k80, n_sus)
    r20 = _first_sustained(exc <= lo, r80, n_sus) if r80 is not None else None
    t_end = float(t[-1])
    if r80 is not None and r20 is not None and r20 > r80:
        tr80, tr20 = float(t[r80]), float(t[r20])
        retract_start = tr80 - (tr20 - tr80) / (EDGE_HI - EDGE_LO) * (1.0 - EDGE_HI)
        retract_end = tr20 + (tr20 - tr80) / (EDGE_HI - EDGE_LO) * EDGE_LO
        out["retract_start_s"], out["retract_end_s"] = retract_start, retract_end
        out["press_hold_s"] = retract_start - plunge_end
        if "z_ramp_2_retract" in st:
            out["retract_lag_s"] = retract_start - st["z_ramp_2_retract"][0]
            if st["z_ramp_2_retract"][1] is not None:
                out["retract_end_lag_s"] = retract_end - st["z_ramp_2_retract"][1]
    else:
        out["why"] = "no_retract"
        retract_end = None
    # current: rise (pressing) and return (feedback restored = segment ④)
    i_pre = ti < event_t
    i_base = float(np.median(np.abs(cur[i_pre]))) if i_pre.sum() else float("nan")
    ai = np.abs(cur)
    out["i0_pa"] = i_base * 1e12
    out["i_sat_over_i0"] = float(np.max(ai) / i_base) if i_base > 0 else None
    t4 = None
    if i_base > 0 and np.any(ai >= CURRENT_RISE_FACTOR * i_base):
        k_rise = int(np.argmax(ai >= CURRENT_RISE_FACTOR * i_base))
        out["current_rise_lag_s"] = float(ti[k_rise]) - plunge_start
        k_max = int(np.argmax(ai))
        n_sus_i = max(2, int(round(0.02 / max(float(np.median(np.diff(ti))), 1e-4))))
        k_ret = _first_sustained(ai <= CURRENT_RETURN_FACTOR * i_base, k_max, n_sus_i)
        if k_ret is None:
            out["why"] = out["why"] or "no_return"
        else:
            t4 = float(ti[k_ret])
            out["t4_s"] = t4
            out["seg4_s"] = t_end - t4
    else:
        out["why"] = out["why"] or "no_press"
    # Δz of segment ③ (Z back at the commanded position, feedback still off) and ④
    if retract_end is not None:
        s3_end = t4 if t4 is not None else t_end
        m3 = (t >= retract_end + 0.02) & (t <= s3_end)
        if m3.sum() >= n_sus:
            out["dz3_pm"] = float(np.median(z[m3]) - z1) * 1e12
            out["seg3_s"] = s3_end - retract_end
    if t4 is not None and t_end - t4 >= 0.2:
        m4 = t >= t_end - 0.1
        out["dz4_pm"] = float(np.median(z[m4]) - z1) * 1e12
    if "end_wait" in st and st["end_wait"][1] is not None:
        out["capture_end_after_declared_s"] = t_end - st["end_wait"][1]
    return out


def _quantiles(vals) -> dict | None:
    x = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if x.size == 0:
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {"n": int(x.size), "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "min": float(x.min()), "max": float(x.max())}


def aggregate(measures: list[dict]) -> dict:
    """Quantiles per quantity over all traces, split by commanded depth, plus failure tallies."""
    out = {"n_traces": len(measures),
           "why": dict(Counter(m.get("why", "") or "ok" for m in measures)),
           "verdict_recorded": dict(Counter(str(m.get("verdict_recorded")) for m in measures)),
           "declared_params": {}, "overall": {}, "by_depth_pm": {}}
    for k in NUMERIC_KEYS:
        out["overall"][k] = _quantiles([m.get(k) for m in measures])
    depths = sorted({round(m.get("depth_cmd_pm", 0.0)) for m in measures})
    for d in depths:
        sub = [m for m in measures if round(m.get("depth_cmd_pm", 0.0)) == d]
        out["by_depth_pm"][str(int(d))] = {
            "n": len(sub),
            "depth_reached_frac": _quantiles([m.get("depth_reached_frac") for m in sub]),
            "dz3_pm": _quantiles([m.get("dz3_pm") for m in sub]),
            "dz4_pm": _quantiles([m.get("dz4_pm") for m in sub]),
            "press_hold_s": _quantiles([m.get("press_hold_s") for m in sub]),
            "verdict_recorded": dict(Counter(str(m.get("verdict_recorded")) for m in sub))}
    out["n_reached_segment4"] = int(sum(1 for m in measures if m.get("dz4_pm") is not None))
    return out


def run(traces_dir: Path) -> tuple[dict, list[dict]]:
    files = sorted(Path(traces_dir).glob("*.json"))
    measures: list[dict] = []
    declared: Counter = Counter()
    for f in files:
        try:
            tr = load_trace(f)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            measures.append({"file": f.name, "why": f"load:{type(exc).__name__}", "depth_cmd_pm": 0.0})
            continue
        tr["_file"] = f.name
        m = measure_segments(tr)
        meta = tr.get("meta", {})
        for k in ("switch_off_delay_s", "lift_time_1_s", "bias_settling_s", "lift_time_2_s",
                  "end_wait_s", "pre_roll_s", "post_roll_s", "poll_hz", "bias_v"):
            declared[f"{k}={meta.get(k)}"] += 1
        measures.append(m)
    agg = aggregate(measures)
    agg["declared_params"] = dict(declared)
    agg["source"] = {"dir": str(traces_dir), "n_files": len(files), "schema": SCHEMA,
                     "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    agg["notes"] = [
        "plunge_lag_s / retract_lag_s are relative to the DECLARED stage boundaries in the trace "
        "(computed from the skill parameters); the world's hw_lag literal should come from here.",
        "dz3_pm is segment ③ (Z ramped back, feedback off) and is ≈ 0 by construction; only dz4_pm "
        "(feedback restored, located by the current returning to ≤ 2× baseline) carries the poke outcome.",
        "Captures that end before the current returns are why=no_return: their dz4_pm is None, not 0.",
    ]
    return agg, measures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default=None, help="directory of TipShapeWithReadback_*.json "
                    "(default $STM_BENCH_DATA/calibration_inputs/poke_traces)")
    ap.add_argument("--out", default=str(calib_dir() / "poke_timing.json"),
                    help="default $STM_BENCH_DATA/calib/poke_timing.json")
    ap.add_argument("--per-trace", default=None, help="optional path for the per-trace measurements (JSON)")
    a = ap.parse_args(argv)
    traces = Path(a.traces) if a.traces else default_traces_dir()
    if traces is None or not traces.is_dir():
        print(f"traces directory not found: {traces} (give --traces)", file=sys.stderr)
        return 2
    agg, measures = run(traces)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=1), encoding="utf-8")
    if a.per_trace:
        Path(a.per_trace).write_text(json.dumps(measures, indent=1), encoding="utf-8")
    ov = agg["overall"]

    def fmt(k, scale=1.0, unit=""):
        q = ov.get(k)
        return f"{k}: n={q['n']} p10={q['p10']*scale:.3g} p50={q['p50']*scale:.3g} p90={q['p90']*scale:.3g}{unit}" if q else f"{k}: n=0"

    print(f"{agg['n_traces']} traces from {traces}; why={agg['why']}; reached ④: {agg['n_reached_segment4']}")
    for k in NUMERIC_KEYS:
        print("  " + fmt(k))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
