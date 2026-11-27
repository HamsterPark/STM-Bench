#!/usr/bin/env python3
"""Grader tests of the stm tasks of P3's repair variant (close the gap in a quantum corral, then measure it).

PASS: the reference solution's export; each peak moved by half the tolerance of the rule card (and the radius, if the
card asks for one), in either direction where the reference's own error leaves it inside; an episode that never reached
``finish``; a digest hash differing in its last bits. FAIL: the damaged logs of ``stm_cases.common_cases``; a log
stopped after the repair but before the first spectrum; a peak 25 meV off; the radius off by more than its tolerance;
the two peaks swapped; peaks citing only an off-centre spectrum or none; a radius citing the first frame (taken before
the repair); and two negative episodes played through the instrument by pieces of the reference solution
(``reference_export.py --variant``): the broken corral measured without repair, and the repair stopped after the first
spare. The tolerances are read from the rule card in the task file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stm_cases as sc  # noqa: E402
from stm_cases import FAIL  # noqa: E402

LO, HI, RADIUS = "peak_lo_mev", "peak_hi_mev", "radius_nm"
VARIANTS = (("unrepaired", "measured without repair (the broken ring measured, no atom moved)"),
            ("incomplete", "ring left incomplete (only the first spare placed, then measured)"))


def cases(task, tasks, fx):
    e = fx[task]
    ptol = sc.card_abs_tol(task, LO, "meV", 10.0)
    rtol = sc.card_abs_tol(task, RADIUS, "nm", 0.15)
    has_radius = bool(sc.reports(e, RADIUS))
    spectra, frames = sc.spectrum_names(e), sc.frame_names(e)
    out = sc.common_cases(task, tasks, fx)
    out += sc.abs_shift_cases(e, task, (LO, HI), ptol, "meV")
    if has_radius:
        out += sc.abs_shift_cases(e, task, (RADIUS,), rtol, "nm")
        out += [(f"radius {rtol + 0.25:.3g} nm off (outside {rtol:g})", FAIL, sc.shifted(e, RADIUS, rtol + 0.25)),
                ("radius cites the first frame, taken before the repair", FAIL,
                 sc.set_evidence(e, (RADIUS,), frames[:1], frame=frames[0]))]
    out += [
        ("repaired and looked at, stopped before the first spectrum", FAIL, sc.cut_before_first(e, "sts")),
        ("the two peaks swapped (not ascending)", FAIL, sc.swapped(e, LO, HI)),
        ("peaks cite only a spectrum taken away from the centre", FAIL, sc.set_evidence(e, (LO, HI), spectra[:1])),
        ("peaks cite no spectrum", FAIL, sc.set_evidence(e, (LO, HI), [])),
    ]
    if ptol <= 12.0:
        # the lowest resonances can be 44 meV apart: 25 meV is outside a 10 meV tolerance and clear of the next level
        out += [(f"lower peak 25 meV off (outside {ptol:g})", FAIL, sc.shifted(e, LO, 25.0)),
                (f"higher peak 25 meV off (outside {ptol:g})", FAIL, sc.shifted(e, HI, 25.0))]
    else:
        print(f"NOTE: the rule card's peak tolerance is {ptol:g} meV: the 'peak 25 meV off' cases are left out", flush=True)
    for kind, label in VARIANTS:
        d = sc.FIXTURES / task / "variants" / kind
        if (d / "export" / "actions.jsonl").is_file():
            note = ""
            if (d / "variant_run.json").is_file():
                vr = json.loads((d / "variant_run.json").read_text(encoding="utf-8"))
                bad = sorted(k for k, ok in (vr.get("claims_ok") or {}).items() if not ok)
                note = (f" [judge when made: ring_occupancy {float(vr.get('ring_occupancy', float('nan'))):.3f}, "
                        f"failed claims {bad}]")
            out.append((label + note, FAIL, sc.Export.load(d / "export")))
    return out


if __name__ == "__main__":
    sys.exit(sc.run_suite("P3 repair", "stm_eval_p3_corral_repair_[0-9][0-9]", cases))
