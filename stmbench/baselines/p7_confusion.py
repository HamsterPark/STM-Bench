"""The carry reader against the judge's record: confusion tables over the drags that
:mod:`stmbench.baselines.p7_validate` collected.

    python -m stmbench.baselines.p7_confusion DRAGS.pkl [DRAGS2.pkl ...]

Only drags that started within 0.3 nm of the atom they held (the judge's ``start_within_nm``) count: for
those ``followed`` is a full carry. The reader is run again on what it saw (the trace, the atom's place
in the after-frame, where the tip stopped), so thresholds can be changed without a new collection.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter

import numpy as np

from ..analysis import carry as C

START_WITHIN_NM = 0.3
GAP_FLOOR_KOHM = 286.0


def load(paths) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, "rb") as fh:
            rows += pickle.load(fh)
    return rows


def qualifying(rows) -> list[dict]:
    return [r for r in rows if r["record"].get("atom_id") is not None and r["record"].get("start_dist_nm") is not None
            and r["record"]["start_dist_nm"] <= START_WITHIN_NM and r["record"].get("feedback")]


def reading(r: dict, *, mode: str = "frame") -> bool | None:
    """The reader's verdict on a stored drag: ``"frame"`` (the real reader: frame, trace as fallback)
    or ``"trace"`` (the trace only)."""
    p = r["pull"]
    atom = p["atom"] if mode == "frame" else None
    return C.read_carry(r["spectrum"], end=p["end"], atom=atom)["carried"]


def table(rows, mode: str = "frame") -> dict:
    """tp/fp/fn/tn and the unreadable, with the rates: a false carry is a drag read as carried that
    the judge's record says was not; a false drop the other way round."""
    c = Counter()
    for r in rows:
        truth, got = bool(r["record"]["followed"]), reading(r, mode=mode)
        if got is None:
            c["unread_carry" if truth else "unread_miss"] += 1
        else:
            c[("tp" if truth else "fp") if got else ("fn" if truth else "tn")] += 1
    n_miss = c["fp"] + c["tn"] + c["unread_miss"]
    n_carry = c["tp"] + c["fn"] + c["unread_carry"]
    return {**{k: c[k] for k in ("tp", "fp", "fn", "tn", "unread_carry", "unread_miss")}, "n": len(rows), "n_carry": n_carry, "n_miss": n_miss,
            "false_carry_rate": (c["fp"] / n_miss) if n_miss else float("nan"),
            "false_drop_rate": (c["fn"] / n_carry) if n_carry else float("nan")}


def _line(name: str, t: dict) -> str:
    return (f"{name:<28} n {t['n']:>5}  carried {t['n_carry']:>5}  missed {t['n_miss']:>5}   false carry {t['fp']:>4} "
            f"({100 * t['false_carry_rate']:.2f} %)   false drop {t['fn']:>4} ({100 * t['false_drop_rate']:.2f} %)   "
            f"unread {t['unread_carry'] + t['unread_miss']}")


def report(rows, mode: str = "frame") -> str:
    q = qualifying(rows)
    out = [f"reader: {mode}; {len(rows)} drags, {len(q)} started within {START_WITHIN_NM} nm of the atom they held"]
    out.append(_line("all", table(q, mode)))
    out.append("by resistance (panel |V|/I)")
    for name, f in (("  below the gap floor (<286 k)", lambda r: r["record"]["r_kohm"] < GAP_FLOOR_KOHM),
                    ("  above (>= 286 k)", lambda r: r["record"]["r_kohm"] >= GAP_FLOOR_KOHM)):
        out.append(_line(name, table([r for r in q if f(r)], mode)))
    out.append("by the tip at the drag")
    for tip in sorted({r["tip"] for r in q}):
        out.append(_line(f"  {tip}", table([r for r in q if r["tip"] == tip], mode)))
    out.append("by how far the atom stopped from the end point (misses only: false carry rate = share read as carried)")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 5.0)):
        sub = [r for r in q if not r["record"]["followed"] and r["record"].get("end_dist_nm") is not None
               and lo <= r["record"]["end_dist_nm"] < hi]
        t = table(sub, mode)
        out.append(f"  {lo:.1f}-{hi:.1f} nm: {len(sub):>5} misses, {t['fp']} read as carried ({100 * t['false_carry_rate']:.2f} %)")
    on_tip = [r for r in q if r["record"].get("picked")]
    t = table(on_tip, mode)
    out.append(f"  lifted onto the tip: {len(on_tip)} drags, {t['fp']} read as carried")
    out.append("how it was read: " + ", ".join(f"{k} {v}" for k, v in Counter(
        C.read_carry(r["spectrum"], end=r["pull"]["end"], atom=r["pull"]["atom"] if mode == "frame" else None)["how"] for r in q).items()))
    return "\n".join(out)


def main() -> None:
    rows = load(sys.argv[1:])
    for mode in ("frame", "trace"):
        print(report(rows, mode))
        print()


if __name__ == "__main__":
    main()
