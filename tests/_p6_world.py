"""A P6-like scenario for the physics tests: Co adatoms on Cu(111), the Kondo resonance, the
``sts_resolution`` switch on, and the hidden draws of the P6 design (BRIEF_physics.md) - built in a
temporary folder from the P4 scenario, so that no shipped scenario file is needed."""
from __future__ import annotations

from pathlib import Path

import yaml

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"

KONDO_HIDDEN = {"gamma_mev": {"range": [3.0, 10.0], "log": True}, "q": [0.0, 1.2], "e_k_mev": [-2.0, 6.0],
                "amplitude": [0.12, 0.30], "sigma_nm": [0.30, 0.45], "q_sigma_nm": [0.45, 0.70]}


def p6_like(tmp_path, *, kondo: dict | None = None, lockin_attenuation: float | None = 1.0,
            sts_resolution: bool = True, species: str = "Co", draw: bool = False, tip: dict | None = None,
            name: str = "p6like") -> Path:
    """Write the scenario and return its path. ``kondo`` are fixed parameters (the ``initial.kondo``
    block, in meV and nm); ``draw`` instead draws them per seed from the P6 ranges. The tip is a
    good, stable one unless ``tip`` says otherwise."""
    d = yaml.safe_load((SCEN / "P4_atom_positioning_cu111.yaml").read_text(encoding="utf-8"))
    d["id"] = name
    d["rig"] = "reference-stm"
    ini = d["initial"]
    ini["physics"] = {"sts_resolution": bool(sts_resolution)}
    ini["tip"] = {"radius_nm": 1.0, "apex_sigma_nm": 0.08, "lambda_per_s": 0.0, **(tip or {})}
    ini["adatoms"] = {**ini["adatoms"], "species": species}
    hidden: dict = {}
    if draw:
        hidden["kondo"] = KONDO_HIDDEN
        hidden["lockin"] = {"attenuation": [0.80, 1.00]}
        ini["kondo"] = {"species": species}
    else:
        ini["kondo"] = {"species": species, **(kondo or {})}
        if lockin_attenuation is not None:
            ini["lockin"] = {"attenuation": lockin_attenuation}
    d["hidden"] = hidden
    p = Path(tmp_path) / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return p


#: a mid-range Co resonance: Γ 6 meV, q 0.3, ε_K 1.5 meV, a0 0.25 (the fitting tests use this)
FIXED = {"gamma_mev": 6.0, "q": 0.3, "e_k_mev": 1.5, "amplitude": 0.25, "sigma_nm": 0.35, "q_sigma_nm": 0.55}
