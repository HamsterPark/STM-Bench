"""qPlus / PLL panel state (controller Oscillation Control).

The sensor's own properties — resonance ``f0``, stiffness ``k``, quality factor ``Q`` — live on
the :class:`~stmsim.physics.tip.Tip`, because they belong to the installed tuning fork. What
lives here is the *controller*: whether the drive is on, what amplitude it holds, how it is
excited. Two behaviours matter physically:

* the oscillation takes ``τ = Q/(π f0)`` to ring up after the output is switched on (≈0.3 s at
  25 kHz / Q 25000), so an amplitude read a millisecond after enabling the output is not the
  setpoint yet;
* touching the surface damps it to a few per cent of free amplitude — which is exactly the
  signature MAST's ``CheckTipCrashByAmplitude`` looks for.

With the drive off the amplitude channel reads the calibrated 7–8 pm noise floor and the
excitation reads zero, which is what MAST's qPlus skills interpret as "no oscillation control".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

CRASH_DAMPING = 0.05                # amplitude fraction left when the tip is in contact
CONTACT_GAP_M = 0.1e-9


@dataclass
class PLLState:
    f0_hz: float = 25296.2              # calibration measurement (2026-08-04)
    q: float = 25000.0
    amp_setpoint_m: float = 50e-12
    undriven_floor_m: float = 8e-12
    input_calibration_m_per_v: float = 4.0e-9
    amp_to_excitation_m_per_v: float = 2.0e-8
    output_on: bool = False
    amp_ctrl_on: bool = True
    phas_ctrl_on: bool = True
    excitation_v: float = 0.0
    center_freq_hz: float = 25296.2
    df_setpoint_hz: float = 0.0
    freq_range_hz: float = 1000.0
    exc_range_idx: int = 1
    inp_range_m: float = 1e-9
    inp_diff: int = 0
    inp_div10: int = 0
    demod_filter_order: int = 3
    demod_harmonic: int = 1
    demod_input: int = 1
    demod_freq_gen: int = 1
    demod_phase_ref_deg: float = 0.0
    amp_p_v_per_m: float = 1e7
    amp_tc_s: float = 1e-3
    amp_bw_hz: float = 20.0
    phas_p_hz_per_deg: float = 5.0
    phas_tc_s: float = 1e-3
    phas_bw_hz: float = 100.0
    add_on: int = 0
    exc_overwrite_idx: int = -1
    freq_overwrite_idx: int = -1
    t_on_wall: float = 0.0
    swp_n: int = 200
    swp_period_s: float = 0.02
    swp_settle_s: float = 0.1

    @classmethod
    def from_rig(cls, rig, tip=None) -> "PLLState":
        f0 = float(getattr(tip, "qplus_f0_hz", 0.0) or rig.get("qplus.f0_hz", 25296.2))
        q = float(getattr(tip, "qplus_q", 0.0) or rig.get("qplus.q", 25000.0))
        return cls(f0_hz=f0, q=q, center_freq_hz=f0,
                   amp_setpoint_m=float(rig.get("qplus.amplitude_default_m", 50e-12)),
                   undriven_floor_m=float(rig.get("qplus.undriven_amplitude_floor_m", 8e-12)),
                   input_calibration_m_per_v=float(rig.get("qplus.input_calibration_m_per_v", 4.0e-9)),
                   amp_to_excitation_m_per_v=float(rig.get("qplus.amp_to_excitation_m_per_v", 2.0e-8)))

    @property
    def tau_s(self) -> float:
        return self.q / (math.pi * max(self.f0_hz, 1.0))

    def excitation_for(self, amplitude_m: float) -> float:
        return float(amplitude_m) / max(self.amp_to_excitation_m_per_v, 1e-30)

    def set_output(self, on: bool, wall_s: float) -> None:
        if on and not self.output_on:
            self.t_on_wall = float(wall_s)
            if self.amp_ctrl_on:
                self.excitation_v = self.excitation_for(self.amp_setpoint_m)
        if not on:
            self.excitation_v = 0.0
        self.output_on = bool(on)

    def amplitude_now(self, wall_s: float, gap_close_m: float | None = None) -> float:
        """Oscillation amplitude right now (0 when the drive is off — the caller adds noise)."""
        if not self.output_on or self.excitation_v <= 0:
            return 0.0
        target = self.amp_setpoint_m if self.amp_ctrl_on else \
            self.excitation_v * self.amp_to_excitation_m_per_v
        ring = 1.0 - math.exp(-max(wall_s - self.t_on_wall, 0.0) / max(self.tau_s, 1e-6))
        damp = 1.0 if (gap_close_m is None or gap_close_m > CONTACT_GAP_M) else CRASH_DAMPING
        return float(target * ring * damp)

    def snapshot(self) -> dict:
        return {"f0_hz": self.f0_hz, "q": self.q, "amp_setpoint_m": self.amp_setpoint_m,
                "output_on": self.output_on, "amp_ctrl_on": self.amp_ctrl_on,
                "phas_ctrl_on": self.phas_ctrl_on, "excitation_v": self.excitation_v,
                "center_freq_hz": self.center_freq_hz}
