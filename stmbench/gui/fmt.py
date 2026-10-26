"""How the window writes numbers and replies (no Qt).

Readings are written in the unit a person reads them in (mV or V, pA or nA, nm), never as
``1.2e-10``; instrument time as ``H:MM:SS``. A reply is shown in the instrument's own words: the
one-line summary names what the action made (the frame or spectrum a report must cite) and the full
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
    """Bias or amplitude: mV below 1 V, V from there."""
    return f"{v * 1e3:.1f} mV" if abs(v) < 1.0 else f"{v:.3f} V"


def picoamps(pa: float) -> str:
    """A current given in pA, as pA below 1 nA and nA from there."""
    return f"{pa / 1e3:.4g} nA" if abs(pa) >= 1000.0 else f"{pa:.4g} pA"


def nanometres(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f} nm"


def position(xy) -> str:
    return f"({xy[0]:.3f}, {xy[1]:.3f}) nm"


def on_off(flag: bool | None, lang: str = "en") -> str:
    if flag is None:
        return "-"
    return ("开 ON" if flag else "关 OFF") if lang == "zh" else ("ON" if flag else "OFF")


def yes_no(flag: bool | None, lang: str = "en") -> str:
    if flag is None:
        return "-"
    return ("是 yes" if flag else "否 no") if lang == "zh" else ("yes" if flag else "no")


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


def params_text(args: dict | None) -> str:
    """The parameters of a call as sent, ``name=value`` in order."""
    return ", ".join(f"{k}={value_text(v)}" for k, v in (args or {}).items())


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
    most common length wins. A move-tip trace is plotted from these, as any other reply would be."""
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
