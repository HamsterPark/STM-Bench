"""P1 through the instrument and the grader. The orientation claims are judged on the direction of
the lines at the reported place (one arm of the zigzag is one rotational domain, ``local: true``),
in the episode and in the host's replay of its export alike: two neighbouring arms are two domains,
two places on one arm are one."""
from __future__ import annotations

from pathlib import Path

import pytest

from stmbench.instrument import Instrument
from stmbench.instrument.export import ExportWriter
from stmbench.instrument.grade import grade_export
from stmbench.trackB import claims as C

P1 = "P1_barth1990_au111"
SEED = 0
#: one frame for every claim: >= 100 nm at <= 0.4 nm/px (the period), >= 40 nm at <= 0.5 nm/px (orientations)
SIZE_NM, PIXELS = 100.0, 256


def _scan(inst: Instrument) -> tuple[str, dict]:
    """The frame's name and its scan_saved event (the drift the judge maps reported places with)."""
    name = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=SIZE_NM, pixels=PIXELS,
                    line_time_s=0.5)["frame"]
    ev = [e for e in inst.world.events if e.get("kind") == "scan_saved" and str(e.get("path", "")).endswith(name)]
    assert ev, name
    return name, ev[-1]


def _places(inst: Instrument, ev: dict) -> list[tuple[float, float, float]]:
    """Scan-frame places (nm) inside the frame, each with the line direction the judge finds there,
    keeping only places whose neighbours 5 nm away share it (the middle of an arm)."""
    surf = inst.world.surface

    def local(x: float, y: float) -> float | None:
        sx, sy = C.to_sample_frame(ev, x, y)
        o = surf.orientation_at(sx * 1e-9, sy * 1e-9) or {}
        return o.get("stripe_local_deg")

    out = []
    half = SIZE_NM / 2 - 10.0
    steps = [-half + i * (2 * half / 12) for i in range(13)]
    for x in steps:
        for y in steps:
            d = local(x, y)
            if d is None:
                continue
            around = [local(x + dx, y + dy) for dx, dy in ((5, 0), (-5, 0), (0, 5), (0, -5))]
            if all(a is not None and C.angle_diff(a, d) < 0.5 for a in around):
                out.append((x, y, float(d)))
    return out


def _episode(path: Path, *, same_arm: bool) -> dict:
    """Report the truth at two places in the middle of two neighbouring arms (``same_arm``: two
    places on one arm), export the episode and return the instrument's own summary."""
    inst = Instrument(P1, SEED)
    exp = ExportWriter(inst, path)
    name, ev = _scan(inst)
    places = _places(inst, ev)
    a = places[0]
    others = [p for p in places if p is not a and (C.angle_diff(p[2], a[2]) < 0.5) == same_arm
              and (same_arm or C.angle_diff(p[2], a[2]) >= 50.0)
              and abs(p[0] - a[0]) + abs(p[1] - a[1]) > 10.0]
    assert others, "no second place of the kind asked for in the frame"
    b = others[0]
    truth = inst.world.truth()
    inst.report("stripe_period_nm", C.truth_at_path(truth, "herringbone.period_along_arm_nm"), evidence=[name])
    for cid, (x, y, d) in (("stripe_orientation_deg", a), ("domain_a_orientation_deg", a),
                           ("domain_b_orientation_deg", b)):
        inst.report(cid, d, x_nm=x, y_nm=y, frame=name, evidence=[name])
    summary = inst.finish()
    exp.sync()
    return summary


@pytest.mark.parametrize("same_arm", [False, True])
def test_a_p1_export_is_graded_on_the_line_direction_at_the_reported_places(same_arm, tmp_path):
    summary = _episode(tmp_path / "export", same_arm=same_arm)
    # two neighbouring arms are two domains; two places on one arm are one, and distinct_from fails
    assert summary["success"] is (not same_arm), summary
    assert summary["claims_verified"] == (4 if not same_arm else 3)
    g = grade_export(tmp_path / "export", P1, SEED)
    assert g["replay"]["consistent"], g["replay"]
    assert (g["success"], g["partial"], g["claims_verified"]) == (
        summary["success"], summary["partial"], summary["claims_verified"])
