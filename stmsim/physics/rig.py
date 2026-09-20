"""Rig profile: the per-instrument numbers, loaded from ``stmsim/profiles/*.yaml``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class RigProfile:
    def __init__(self, data: dict[str, Any]):
        self.data = data

    @classmethod
    def load(cls, name_or_path: str | Path = "reference-stm") -> "RigProfile":
        p = Path(name_or_path)
        if not p.exists():
            p = PROFILES_DIR / f"{name_or_path}.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        # a variant profile (e.g. the qPlus rig) states only what differs from its parent, so
        # the two cannot drift apart; lists replace wholesale, mappings merge key by key
        parent = data.pop("extends", None)
        if parent:
            base = cls.load(parent).data
            data = _deep_merge(base, data)
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    # ── derived conveniences ──
    @property
    def name(self) -> str:
        return str(self.data.get("name", "generic"))

    @property
    def temperature_k(self) -> float:
        return float(self.get("environment.temperature_k", 300.0))

    @property
    def z_range_m(self) -> float:
        cold = self.temperature_k < 20.0
        return float(self.get("z.range_m_lhe" if cold else "z.range_m_rt", 360e-9))

    @property
    def extend_sign(self) -> int:
        return int(self.get("z.extend_sign", -1))

    @property
    def z_cal_error(self) -> float:
        return float(self.get("z.calibration_error", 0.0))

    @property
    def xy_range_m(self) -> tuple[float, float]:
        return (float(self.get("xy.range_x_m", 2.6e-6)), float(self.get("xy.range_y_m", 2.6e-6)))

    @property
    def preamp_full_scale_a(self) -> float:
        return float(self.get("preamp.full_scale_a", 10e-9))

    @property
    def noise_floor_a(self) -> float:
        return float(self.get("preamp.noise_floor_a", 0.05e-12))

    @property
    def desat_tau_s(self) -> float:
        return float(self.get("preamp.desaturation_tau_s", 0.45))

    @property
    def modules_loaded(self) -> set[str]:
        return set(self.get("modules_loaded", []) or [])

    @property
    def modules_not_loaded(self) -> set[str]:
        return set(self.get("modules_not_loaded", []) or [])

    @property
    def osci_dt_s(self) -> float:
        return float(self.get("timing.osci_dt_s", 0.0005))

    @property
    def osci1t_timebase_s(self) -> list[float]:
        return [float(x) for x in self.get("timing.osci1t_timebase_s", [6.4, 2.56, 1.28, 0.64, 0.256, 0.128])]

    @property
    def rt_freq_hz(self) -> float:
        return float(self.get("timing.rt_freq_hz", 20000.0))
