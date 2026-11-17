"""Calibration runs of the P6 reference: named protocol variants over many seeds, in one process.

The cluster's job script cannot take arguments with spaces, and the calibration needs several
variants (the careful protocol, the contrast runs, development variants) over the same seeds, so
this module runs them from one command::

    python -m stmbench.reference.p6_calibrate --variants base,a0.5 --seeds 0-29 --jobs 16 --out runs/

It writes ``<out>/<variant>.json`` (``{"summary": ..., "runs": [...]}``, as ``stmbench.cli
reference --json`` does, plus a summary per tip condition).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import p6, summarize

#: the protocol's defaults are p6.run's; a variant lists only what it changes
VARIANTS: dict[str, dict] = {
    "base": {},
    # development variants (the settled protocol is 3 mV, 100 ms, 6 + 4 sweeps, Z compensated)
    "a1": {"amplitude_mv": 1.0}, "a2": {"amplitude_mv": 2.0}, "a4": {"amplitude_mv": 4.0},
    "int200": {"integration_ms": 200.0}, "int300": {"integration_ms": 300.0},
    "s3b2": {"atom_sweeps": 3, "bare_sweeps": 2}, "s8b5": {"atom_sweeps": 8, "bare_sweeps": 5},
    "noz": {"z_comp": False}, "img": {"imaging": True},
    # contrast (a): the instrument's defaults (10 mV, 5 ms), fitted with the nominal 10 mV kernel
    "contrast_default": {"amplitude_mv": 10.0, "integration_ms": 5.0},
    # contrast (b): the settled protocol without drift compensation
    "contrast_nodrift": {"drift_comp": False},
}


def _one(args: tuple[str, int, dict]) -> dict:
    variant, seed, kw = args
    try:
        r = p6.run(seed, **kw)
    except Exception as exc:  # noqa: BLE001 — a failed seed is a result
        return {"paper": "P6", "seed": int(seed), "success": False, "error": f"{type(exc).__name__}: {exc}"}
    r["variant"] = variant
    return r


def _seeds(text: str) -> list[int]:
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base")
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.variants.split(",")
    seeds = _seeds(a.seeds)
    tasks = [(v, s, VARIANTS[v]) for v in names for s in seeds]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(_one, tasks))
    for v in names:
        # failed seeds carry no variant: attribute rows to variants by position
        mine = [r for (vv, _, _), r in zip(tasks, rows) if vv == v]
        by_tip = {}
        for tip in sorted({str(r.get("tip_condition")) for r in mine if "error" not in r}):
            by_tip[tip] = summarize([r for r in mine if str(r.get("tip_condition")) == tip])
        (out / f"{v}.json").write_text(json.dumps({"summary": summarize(mine), "by_tip": by_tip, "runs": mine},
                                                  ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        s = summarize(mine)
        print(v, f"n={s['n']} success={s['success_rate']:.2f} failed={len(s['errors'])}",
              {c: (round(d['pass_rate'], 2), round(d['err_median'], 3), round(d['err_p90'], 3)) for c, d in s["claims"].items()},
              f"t_med={s['instrument_s']['median']:.0f} t_p90={s['instrument_s']['p90']:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
