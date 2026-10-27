"""Physics a scenario switches on: mechanisms that most scenarios do not have.

A scenario's ``initial`` block may hold::

    physics: {tip_changes_outside_scans: true, tip_recovery: true,
              tip_anisotropy: true, crash_over_range: true}

Every key is optional and false when missing, so a scenario without the block builds the world
it always built: same draws, same records. An unknown key raises: a typo would otherwise switch
nothing, which looks exactly like "the mechanism had no effect".

* ``tip_changes_outside_scans`` — the tip's spontaneous-change hazard also runs while the
  instrument tunnels outside scans and drags (spectroscopy, waiting, moving, settling);
* ``tip_recovery`` — the tip also gets better on its own (a metastable apex settles, a
  secondary apex is lost, an adsorbate desorbs, a crashed cluster stops flickering);
* ``tip_anisotropy`` — the tip's ``axis_ratio`` and ``angle`` shape the images, and a blunt
  tip, a crash, a bad pulse or a deep poke make it anisotropic;
* ``crash_over_range`` — a setpoint above the preamplifier's range cannot be reached, and the
  loop drives the tip into the sample.

Each stochastic mechanism draws only from its own generator, seeded
``default_rng([seed, KEY])`` with the keys below, so that switching one on moves no other
stream (AGENTS.md invariant 1). The move trace's noise has the key ``0x7ACE`` (not used here).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

#: tip changes (and, with ``tip_recovery`` on, improvements) in the intervals outside scans
TIP_OUTSCAN_STREAM = 0x71D1
#: tip recovery: when a tip improves, and which way
TIP_RECOVERY_STREAM = 0x71D2
#: the anisotropy a crash, a pulse or a deep poke leaves on the tip
TIP_ANISO_STREAM = 0x71D3


@dataclass(frozen=True)
class PhysicsSwitches:
    tip_changes_outside_scans: bool = False
    tip_recovery: bool = False
    tip_anisotropy: bool = False
    crash_over_range: bool = False

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_config(cls, cfg, *, where: str = "initial.physics") -> "PhysicsSwitches":
        """The switches a scenario's ``initial.physics`` block names; all off for no block."""
        if cfg is None:
            return cls()
        if not isinstance(cfg, dict):
            raise ValueError(f"{where} must be a mapping of switch -> true/false, got {type(cfg).__name__}")
        unknown = sorted(set(cfg) - set(cls.keys()))
        if unknown:
            raise ValueError(f"{where}: unknown physics switch(es) {unknown} (known: {list(cls.keys())})")
        for k, v in cfg.items():
            if not isinstance(v, bool):
                raise ValueError(f"{where}.{k} must be true or false, got {v!r}")
        return cls(**cfg)

    def any(self) -> bool:
        return any(getattr(self, k) for k in self.keys())

    def as_dict(self) -> dict[str, bool]:
        return {k: bool(getattr(self, k)) for k in self.keys()}
