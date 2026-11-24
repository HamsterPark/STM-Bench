"""The scripted P7 baseline: two pull thresholds from what the instrument shows, no truth."""
from __future__ import annotations

from stmbench.baselines import p7
from stmbench.baselines._common import Blind
from stmbench.instrument import Instrument

P7 = "P7_pull_threshold_cu111"


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


def test_the_baseline_measures_two_atoms_through_what_an_agent_gets():
    """Seed 1 arrives with a good tip: three pokes, the drift, a survey, two series of drags each
    read from a frame after it, a logistic fit, and a report that cites the series' move traces. The
    judge accepts the evidence of both claims and both values (development seed: the tolerance is
    10 %, the baseline's error on this seed is about 2 %)."""
    inst = Instrument(P7, 1)
    log: dict = {}
    got = p7.play(Blind(_NoWorld(inst)), log)
    assert log["tip_first"] == "ok" and len(got["r_kohm"]) == 2
    for cid, val, ev in zip(("r_pull_1_kohm", "r_pull_2_kohm"), got["r_kohm"], got["evidence"]):
        assert len(ev) == p7.PROTO.drags_per_atom
        inst.report(cid, val, unit="kΩ", evidence=ev)
    out = inst.finish()
    det = inst.verdict.details["claims"]
    assert all(det[c]["evidence_ok"] for c in ("r_pull_1_kohm", "r_pull_2_kohm")), det
    assert out["success"], det
    assert inst.used_s < inst.max_sim_s


def test_a_lifted_atom_is_not_read_as_carried():
    """A frame that shows no atom on the path after the drag is a miss, whatever the trace says (a
    lifted atom steps Z down, and a trace that ends high can follow it)."""
    from stmbench.analysis.carry import read_carry
    import numpy as np
    z = np.linspace(0.0, 5e-12, 21)
    sp = {"z_abs": z, "dist": np.linspace(0.0, 1e-9, 21), "feedback": True}
    assert read_carry(sp, end=(0.0, 0.0), atom=None)["carried"] is True               # no frame: the trace
    assert read_carry(sp, end=(0.0, 0.0), atom=None, framed=True)["carried"] is False
