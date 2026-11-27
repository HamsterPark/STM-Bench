#!/usr/bin/env python3
"""Grader tests of the stm tasks of P3 (the resonances of a quantum corral), carried over from the delivered suite.

PASS: the reference solution's export; both peaks moved by half the 10 meV allowed and the radius by half the 0.15 nm
(each in either direction where the reference's own error leaves it inside); an episode that never reached ``finish``;
a digest hash differing in its last bits. FAIL: the damaged logs of ``stm_cases.common_cases``; a peak 25 meV off; the
radius 0.4 nm off; the two peaks swapped; peaks citing only a spectrum taken away from the corral centre or none;
a radius citing only a frame narrower than the 22 nm the card asks for.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stm_cases as sc  # noqa: E402
from stm_cases import FAIL  # noqa: E402

LO, HI, RADIUS = "peak_lo_mev", "peak_hi_mev", "radius_nm"


def cases(task, tasks, fx):
    e = fx[task]
    ptol = sc.card_abs_tol(task, LO, "meV", 10.0)
    rtol = sc.card_abs_tol(task, RADIUS, "nm", 0.15)
    spectra, frames = sc.spectrum_names(e), sc.frame_names(e)
    assert len(spectra) >= 2 and len(frames) >= 2, "the fixture holds too few spectra or frames"
    out = sc.common_cases(task, tasks, fx)
    out += sc.abs_shift_cases(e, task, (LO, HI), ptol, "meV")
    out += sc.abs_shift_cases(e, task, (RADIUS,), rtol, "nm")
    out += [
        (f"lower peak 25 meV off (outside {ptol:g})", FAIL, sc.shifted(e, LO, 25.0)),
        (f"higher peak 25 meV off (outside {ptol:g})", FAIL, sc.shifted(e, HI, 25.0)),
        (f"radius 0.4 nm off (outside {rtol:g})", FAIL, sc.shifted(e, RADIUS, 0.4)),
        ("the two peaks swapped (not ascending)", FAIL, sc.swapped(e, LO, HI)),
        # the first spectrum is taken on bare terrace well outside the ring: right kind, wrong place
        ("peaks cite only a spectrum taken away from the centre", FAIL, sc.set_evidence(e, (LO, HI), spectra[:1])),
        ("peaks cite no spectrum", FAIL, sc.set_evidence(e, (LO, HI), [])),
        # the first frame is one of the two 12 nm frames of the drift measurement
        ("radius cites only a 12 nm frame (rule card: 22 nm)", FAIL,
         sc.set_evidence(e, (RADIUS,), frames[:1], frame=frames[0])),
    ]
    return out


if __name__ == "__main__":
    sys.exit(sc.run_suite("P3", "stm_eval_p3_corral_[0-9][0-9]", cases))
