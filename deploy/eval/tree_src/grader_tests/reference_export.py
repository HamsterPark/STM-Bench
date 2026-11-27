#!/usr/bin/env python3
"""Make the correct submissions of the stm tasks: the privileged reference solution of STM-Bench
(stmbench/reference/<module>.py, by the scenario of gt.json: see REFERENCE) runs one episode per task through the
instrument with the export on, as the
console writes it. Needs Python 3.13, the STM-Bench package of the delivered commit (PYTHONPATH or
--code) and numpy, scipy and PyYAML at the pins of deploy/osworld/grader/requirements.txt.

    python reference_export.py --private <dir holding reference_private/> --out <fixtures dir> [--tasks ID ...]

For each task it reads the scenario and seed of ``reference_private/<id>/gt.json`` and writes
``<out>/<id>/export/`` (session.json, actions.jsonl, digests.jsonl, report.json) and
``<out>/<id>/reference_run.json`` (how the run went, without the seed). Prints no seed.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shutil
import sys
from pathlib import Path


#: scenario id -> module of stmbench/reference; None: the reference solution does not exist yet (P1, P2)
REFERENCE = {"P1_barth1990_au111": None, "P2_standing_waves_cu111": None, "P3_corral_cu111": "p3",
             "P3_corral_repair_cu111": "p3_repair", "P4_atom_positioning_cu111": "p4",
             "P5_sader_jarvis_qplus": "p5", "P7_pull_threshold_cu111": "p7"}


def run_variant(kind: str, seed: int, scenario: str, _common) -> dict:
    """A negative case of the repair task, played through the instrument with the export on (the patched Session
    of main() writes it). Records what the judge said per claim, read after the episode (privileged)."""
    from stmbench.analysis.corral import find_ring
    from stmbench.reference import p3, p3_repair

    if kind == "incomplete":
        orig = p3_repair._pair
        p3_repair._pair = lambda s, sv, plan: orig(s, sv, plan)[:1]       # only the first spare is handled
        try:
            r = p3_repair.run(seed, scenario=scenario)
        finally:
            p3_repair._pair = orig
        r.pop("seed", None)
        r["variant"] = kind
        return r
    # unrepaired: the corral as it arrives, measured by P3's own protocol (no drift compensation is running)
    s = _common.Session(scenario, seed)
    inst, w = s.inst, s.w
    ring = w.surface.corral_snapshot()
    centre = tuple(v * 1e-9 for v in ring["centre_nm"])
    radius_m = float(ring["radius_nm"]) * 1e-9
    log = {"tip_rounds": 0}
    p3_repair._clean_tip(s, centre, radius_m, log, spectroscopy=True)
    m = p3.measure(s, centre, radius_m, compensated=False, ring_fit=find_ring)
    got = m.pop("values")
    out = inst.finish()
    return {"variant": kind, "success": bool(out["success"]), "partial": float(out.get("partial", 0.0)),
            "values": got, "claims_ok": _common.claims_ok(inst),
            "claims_detail": (inst.verdict.details or {}).get("claims", {}),
            "ring_occupancy": float(w.truth()["corral"]["ring_occupancy"]), "gap_sites": len(ring["gap_sites"]),
            "instrument_s": float(inst.used_s), **log}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--code", type=Path, help="directory holding the stmbench and stmsim packages")
    ap.add_argument("--force", action="store_true", help="replace an existing fixture")
    ap.add_argument("--variant", choices=("unrepaired", "incomplete"),
                    help="a negative case of the repair task instead of the correct episode: 'unrepaired' measures the "
                         "broken corral without repairing it, 'incomplete' runs the repair reference with only the first "
                         "spare handled; written to <out>/<id>/variants/<variant>/")
    a = ap.parse_args()
    if a.code:
        sys.path.insert(0, str(a.code))

    from stmbench.instrument.export import ExportWriter
    from stmbench.reference import _common

    #: the instrument's calls a console makes; each is followed by a sync, as the console's session does
    calls = ("act", "report", "finish", "status", "frame", "spectrum", "info", "list_data")
    holder: dict = {}
    original_init = _common.Session.__init__

    def init(self, scenario, seed, out_dir=None):
        original_init(self, scenario, seed, out_dir)
        exp = ExportWriter(self.inst, holder["export_dir"])
        for name in calls:
            fn = getattr(self.inst, name)

            def wrapped(*args, _fn=fn, **kw):
                try:
                    return _fn(*args, **kw)
                finally:
                    exp.sync()

            setattr(self.inst, name, wrapped)

    _common.Session.__init__ = init
    root = a.private / "reference_private"
    ids = a.tasks or sorted(p.name for p in root.iterdir() if (p / "gt.json").is_file())
    if a.variant:
        for tid in ids:
            gt = json.loads((root / tid / "gt.json").read_text(encoding="utf-8"))
            if not str(gt["scenario"]).startswith("P3_corral_repair"):
                print(f"{tid}: variants exist only for the repair task", file=sys.stderr)
                return 1
            dest = a.out / tid / "variants" / a.variant
            if dest.exists():
                if not a.force:
                    print(f"{tid}: variant {a.variant} exists, kept")
                    continue
                shutil.rmtree(dest)
            holder["export_dir"] = dest / "export"
            r = run_variant(a.variant, int(gt["seed"]), str(gt["scenario"]), _common)
            (dest / "variant_run.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
            print(f"{tid} [{a.variant}]: success={r['success']} partial={r['partial']:.2f} "
                  f"ring_occupancy={r['ring_occupancy']:.3f} claims_ok={r['claims_ok']} "
                  f"instrument_s={r['instrument_s']:.0f}")
        return 0
    for tid in ids:
        gt = json.loads((root / tid / "gt.json").read_text(encoding="utf-8"))
        dest = a.out / tid
        if dest.exists():
            if not a.force:
                print(f"{tid}: fixture exists, kept")
                continue
            shutil.rmtree(dest)
        holder["export_dir"] = dest / "export"
        paper = REFERENCE.get(str(gt["scenario"]))
        if paper is None:
            print(f"{tid}: skipped: no reference solution yet for scenario {gt['scenario']}")
            continue
        fn = importlib.import_module(f"stmbench.reference.{paper}").run
        # P5's reference has one scenario and takes no ``scenario`` argument
        kw = {"scenario": str(gt["scenario"])} if "scenario" in inspect.signature(fn).parameters else {}
        r = fn(int(gt["seed"]), **kw)
        r.pop("seed", None)
        (dest / "reference_run.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        n = sum(1 for _ in (dest / "export" / "actions.jsonl").open(encoding="utf-8"))
        print(f"{tid}: success={r['success']} partial={r['partial']} instrument_s={r['instrument_s']:.0f} "
              f"records={n}")
        if not r["success"]:
            print(f"{tid}: the reference solution did not reproduce this world", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
