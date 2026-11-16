"""How the window writes numbers and replies (no Qt).

Readings are written in the unit of the field that sets them (bias in V, setpoint and current in pA,
positions in nm), never as ``1.2e-10`` and never switching unit with the size of the value: an operator
who reads 0.050 V types 0.050, and one who reads 1500 pA does not have to convert 1.5 nA. Instrument
time is ``H:MM:SS``. A reply is shown in the instrument's own words: the
one-line summary names what the action made (the frame, spectrum or move trace a report must cite) and the full
form is the reply as the instrument sent it.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

_HIDDEN_REPLY_KEYS = ("instrument_time_s", "budget_s")
_INT = re.compile(r"[+-]?[0-9]+", re.ASCII)
_FLOAT = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?", re.ASCII)


def hms(seconds: float | None) -> str:
    """Instrument time as ``H:MM:SS``."""
    s = int(round(max(0.0, float(seconds or 0.0))))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def num(v: Any, digits: int = 6) -> str:
    """A number without exponent noise for ordinary magnitudes (``0.05``, ``293``, ``1e-05``)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        return f"{v:.{digits}g}"
    return str(v)


def range_text(lo: Any, hi: Any) -> str:
    """A parameter's range as the action table gives it, for the line that describes the field last clicked: ``0.1 … 100000``,
    ``≥ 1`` or ``≤ 5`` where one end is open, ``""`` where both are (the unit is the field's label's)."""
    if lo is not None and hi is not None:
        return f"{num(lo)} … {num(hi)}"
    if lo is not None:
        return f"≥ {num(lo)}"
    if hi is not None:
        return f"≤ {num(hi)}"
    return ""


def value_text(v: Any) -> str:
    """Any reply value on one line: lists as ``(a, b)``, mappings as ``{k: v}``."""
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(value_text(x) for x in v) + ")"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {value_text(x)}" for k, x in v.items()) + "}"
    if v is None:
        return "-"
    return num(v)


def volts(v: float) -> str:
    """A voltage in V with at least the millivolt the bias field takes (``0.050 V``), more decimals when
    the value has them (``0.0125 V``), at most six."""
    whole, _, frac = f"{0.0 if abs(v) < 5e-7 else v:.6f}".rstrip("0").partition(".")
    return f"{whole}.{frac.ljust(3, '0')} V"


def picoamps(pa: float) -> str:
    """A current in pA, the unit of the setpoint field, whatever its size: ``1500 pA``, never ``1.5 nA``."""
    return f"{pa:.0f} pA" if abs(pa) >= 1000.0 else f"{pa:.4g} pA"


def nanometres(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f} nm"


def position(xy) -> str:
    return f"({xy[0]:.3f}, {xy[1]:.3f}) nm"


def on_off(flag: bool | None, lang: str = "en") -> str:
    """ON or OFF; ``n/a`` for a switch the instrument does not have (no qPlus sensor, so no oscillation)."""
    if flag is None:
        return "n/a"
    return ("开 ON" if flag else "关 OFF") if lang == "zh" else ("ON" if flag else "OFF")


def yes_no(flag: bool | None, lang: str = "en") -> str:
    if flag is None:
        return "-"
    return ("是 yes" if flag else "否 no") if lang == "zh" else ("yes" if flag else "no")


def instrument_state(st: dict) -> dict[str, Any]:
    """The switches and the current range as ``status()`` reports them, by the name of the action that sets
    each: what the window's On / Off pairs and range buttons show as selected. ``None``: the instrument
    has no such switch (no qPlus sensor), or the reading is missing."""
    dc = st.get("drift_comp")
    return {"feedback": st.get("feedback"), "lockin": st.get("lockin"), "pll": st.get("oscillating"),
            "drift_comp": dc.get("on") if isinstance(dc, dict) else None,
            "set_current_range": st.get("current_full_scale_na")}


def drift_comp_text(dc: dict | None, lang: str = "en") -> str:
    """The programmed drift compensation as ``status()`` reports it: OFF, or ON with the velocities it
    was programmed with (the units of the ``drift_comp`` action's fields)."""
    if not isinstance(dc, dict):
        return "n/a"
    if not dc.get("on"):
        return on_off(False, lang)
    vx, vy = dc["v_nm_per_min"]
    vz = dc.get("vz_pm_per_min") or 0.0                  # the z velocity only when there is one: the line is long enough
    return f"{on_off(True, lang)} ({num(vx, 4)}, {num(vy, 4)}) nm/min" + (f", {num(vz, 4)} pm/min" if vz else "")


def reading_items(st: dict, lang: str = "en") -> list[tuple[str, str]]:
    """The status strip's second line: ``(label key, value text)`` for every reading of ``status()``, in
    the order an operator reads them."""
    return [
        ("bias", volts(st["bias_v"])),
        ("setpoint", picoamps(st["setpoint_pa"])),
        ("feedback", on_off(st["feedback"], lang)),
        ("current", picoamps(st["current_pa"])),
        ("z", nanometres(st["z_nm"])),
        ("range", f"{num(st['current_full_scale_na'], 4)} nA"),
        ("tip", position(st["tip_nm"])),
        ("withdrawn", yes_no(st["withdrawn"], lang)),
        ("lockin", on_off(st["lockin"], lang)),
        ("oscillation", on_off(st.get("oscillating"), lang)),
        ("drift", drift_comp_text(st.get("drift_comp"), lang)),
    ]


def facts_items(info: dict) -> list[tuple[str, str]]:
    """The instrument's fixed facts (``info()``) as ``(label key, value text)`` in the units an operator
    finds them registered in; a qPlus rig adds its sensor's constants."""
    rx, ry = info["piezo_xy_range_nm"]
    lo, hi = info["bias_range_v"]
    rows = [("fact_material", str(info["material"])),
            ("fact_temperature", f"{num(info['temperature_k'], 4)} K"),
            ("fact_piezo_xy", f"{num(rx)} × {num(ry)} nm"),
            ("fact_piezo_z", f"{num(info['piezo_z_range_nm'])} nm"),
            ("fact_bias_range", f"{num(lo)} … {num(hi)} V"),
            ("fact_current_ranges", ", ".join(num(v) for v in info["current_ranges_na"]) + " nA"),
            ("fact_budget", hms(info["budget_s"]) if info.get("budget_s") else "-")]
    q = info.get("qplus")
    if q:
        rows += [("fact_qplus_f0", f"{num(q['f0_hz'], 9)} Hz"), ("fact_qplus_k", f"{num(q['k_n_per_m'], 9)} N/m"),
                 ("fact_qplus_q", num(q["q"], 9))]
    return rows


def describe_job(kind: str, label: str, args: dict | None = None) -> str:
    """What a job asks of the instrument, in a few words, for the strip while it runs: a scan by its
    centre, size and pixels, any other action by its name and the parameters it was clicked with."""
    a = dict(args or {})
    if kind == "act":
        if label == "scan" and all(k in a for k in ("center_x_nm", "center_y_nm", "size_nm")):
            px = f", {num(a['pixels'])} px" if "pixels" in a else ""
            return f"scan centre ({num(a['center_x_nm'])}, {num(a['center_y_nm'])}) size {num(a['size_nm'])} nm{px}"
        return f"{label} {params_text(a)}".strip()
    if kind == "report":
        return f"report_result {label}"
    if kind in ("view_frame", "view_spectrum"):
        return f"reading {label}"
    if kind == "boot":
        return "reading the instrument's state"
    return label


def params_text(args: dict | None) -> str:
    """The parameters of a call as sent, ``name=value`` in order."""
    return ", ".join(f"{k}={value_text(v)}" for k, v in (args or {}).items())


def call_text(label: str, args: dict | None = None) -> str:
    """A call as the instrument got it (or would have): its name and the parameters it was clicked with."""
    return f"{label} {params_text(args)}".strip()


def reply_summary(name: str, reply: Any) -> str:
    """One line for the status strip and the group's reply line; the Log keeps the whole reply."""
    if not isinstance(reply, dict):
        return f"{name} → {reply}"
    dur = reply.get("duration_s")
    tail = f" ({num(dur)} s)" if dur is not None else ""
    if "frame" in reply and name == "scan":
        return f"scan → {reply['frame']}{tail}"
    if "spectrum" in reply:
        return f"{name} → {reply['spectrum']}{tail}"
    if "trace" in reply:                                         # a move's recorded Z and current, saved like a spectrum
        return f"{name} → {reply['trace']}{tail}"
    fields = {k: v for k, v in reply.items() if k not in _HIDDEN_REPLY_KEYS and k != "duration_s"}
    body = ", ".join(f"{k}={value_text(v)}" for k, v in fields.items())
    return f"{name} → {body}{tail}" if body else f"{name} → done{tail}"


def reply_full(reply: Any) -> str:
    """The reply as the instrument sent it, on one line."""
    if isinstance(reply, str):
        return reply
    return json.dumps(reply, ensure_ascii=False, default=str)


def numeric_lists(reply: Any, min_len: int = 8) -> dict[str, list[float]]:
    """The fields of a reply that hold equal-length lists of numbers, ``min_len`` long or more: the
    most common length wins. Any reply that carries such lists is plotted from these (a move's trace is
    not one: it is saved like a spectrum and shown as one)."""
    if not isinstance(reply, dict):
        return {}
    found: dict[str, list[float]] = {}
    for k, v in reply.items():
        if isinstance(v, (list, tuple)) and len(v) >= min_len and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            found[str(k)] = [float(x) for x in v]
    if not found:
        return {}
    lengths = [len(v) for v in found.values()]
    keep = max(set(lengths), key=lambda n: (lengths.count(n), n))
    return {k: v for k, v in found.items() if len(v) == keep}


def parse_number(text: str) -> int | float | str:
    """A field's text as the instrument should get it: an integer or a number when it is one, else
    the text itself (the instrument refuses it in its own words)."""
    raw = str(text).strip()
    if _INT.fullmatch(raw):
        return int(raw)
    if _FLOAT.fullmatch(raw):
        v = float(raw)
        return v if math.isfinite(v) else raw
    return raw


def sweep_text(j: dict | None) -> str:
    """What a spectrum's sweep was, from the arguments the window sent (``v_start``, ``v_end``, ``sweep_nm``):
    ``sweep -0.550 to 0.150 V``. Empty when the window did not take the spectrum."""
    if not j:
        return ""
    if "v_start" in j and "v_end" in j:
        return f"sweep {volts(float(j['v_start']))[:-2]} to {volts(float(j['v_end']))}"
    if "sweep_nm" in j:
        return f"sweep {num(float(j['sweep_nm']), 4)} nm"
    return ""


def junction_text(j: dict | None) -> str:
    """The bias and setpoint the junction stood at when the sweep started: the status the window had read before the
    call, not anything the instrument reported about the spectrum."""
    if not j or "bias_v" not in j or "setpoint_pa" not in j:
        return ""
    return f"from {volts(j['bias_v'])} · {picoamps(j['setpoint_pa'])}"
