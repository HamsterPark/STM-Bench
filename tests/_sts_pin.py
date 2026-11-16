"""What one bias sweep of a shipped scenario leaves behind, as short digests.

Used by tests/test_sts_resolution.py to pin that the ``sts_resolution`` code path does not touch
a scenario that has the switch off. The pinned values were computed with the code of commit
14bdc55 (before the sts_resolution sweep existed), by ``python -m tests._sts_pin``.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def array_digest(a) -> str:
    """Digest of an array to 9 significant digits: stable across machines whose last bits differ."""
    return _h(",".join(f"{float(x):.8e}" for x in np.asarray(a, float).ravel()))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return f"{float(o):.8e}"
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def rng_digest(rng) -> str:
    return _h(json.dumps(_jsonable(rng.bit_generator.state), sort_keys=True))


def sweep_digest(scenario_id: str, seed: int = 0) -> dict:
    """Two sweeps (lock-in off, then on at 5 mV) through the Instrument, and the end state."""
    from stmbench.instrument import Instrument
    inst = Instrument(scenario_id, seed)
    w = inst.world
    out: dict = {}
    for label, pre in (("off", None), ("on", {"on": True, "amplitude_mv": 5.0})):
        if pre is not None:
            inst.act("lockin", **pre)
        inst.act("sts", v_start=-0.05, v_end=0.05, points=48, settle_ms=2.0, integration_ms=8.0)
        rec = w.sts_records[-1]
        out[label] = {"v": array_digest(rec["v"]), "i": array_digest(rec["i"]), "didv": array_digest(rec["didv"]),
                      "record": _h(json.dumps(_jsonable({k: v for k, v in rec.items()
                                                         if k not in ("v", "i", "didv", "path")}), sort_keys=True)),
                      "event": _h(json.dumps(_jsonable([e for e in w.events if e.get("kind") == "sts"][-1]),
                                             sort_keys=True))}
    out["rng"] = {"world": rng_digest(w.rng), "noise": rng_digest(w.noise.rng), "tip": rng_digest(w.tip.rng),
                  "move": rng_digest(w.move_rng), "df": rng_digest(w.df_rng),
                  "outscan": rng_digest(w.rng_outscan), "recovery": rng_digest(w.rng_recovery),
                  "aniso": rng_digest(w.rng_aniso)}
    return out


if __name__ == "__main__":
    res = {sid: sweep_digest(sid) for sid in ("P2_standing_waves_cu111", "P4_atom_positioning_cu111")}
    print(json.dumps(res, indent=1, sort_keys=True))
