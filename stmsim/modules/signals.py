"""controller global signal table (128 slots) and live evaluation.

Index conventions follow the controller surface (the crash-check client defaults 0 = Current,
14 = Z); the rest are plausible V5e names. Values are evaluated on demand from the world.
"""
from __future__ import annotations

import math

SIGNAL_NAMES: list[str] = ["" for _ in range(128)]
_FIXED = {
    0: "Current (A)", 1: "Input 2 (V)", 2: "Input 3 (V)", 3: "Input 4 (V)", 4: "Input 5 (V)",
    5: "Input 6 (V)", 6: "Input 7 (V)", 7: "Input 8 (V)", 8: "X (m)", 9: "Y (m)",
    10: "Internal 10 (V)", 11: "Internal 11 (V)", 12: "Internal 12 (V)",
    13: "Internal 13 (V)", 14: "Z (m)", 15: "Excitation (V)",
    # Oscillation-control channels use the controller's abbreviations: clients match them by name
    # ("amplitude" next to "oc"/"osc"/"pll", "freq. shift"), and so does the .dat corpus
    16: "OC D1 Amplitude (m)", 17: "OC M1 Freq. Shift (Hz)", 18: "OC D1 Phase (deg)",
    19: "OC M1 Excitation (V)",
    20: "LI Demod 1 X (A)", 21: "LI Demod 1 Y (A)", 22: "LI Demod 2 X (A)", 23: "LI Demod 2 Y (A)",
    24: "Bias (V)", 25: "Z-Ctrl Setpoint (A)", 26: "Internal 26 (V)", 27: "Temperature 1 (K)",
    28: "Temperature 2 (K)", 29: "Pressure (Pa)", 30: "Z Error (m)",
}
for _i, _n in _FIXED.items():
    SIGNAL_NAMES[_i] = _n
for _i in range(31, 128):
    SIGNAL_NAMES[_i] = f"Internal {_i} (V)"


class SignalTable:
    def __init__(self, world):
        self.w = world

    def names(self) -> list[str]:
        return list(SIGNAL_NAMES)

    def value(self, index: int) -> float:
        w = self.w
        if index == 0:
            return w.current_now()
        if index == 14:
            return w.z_now()
        if index == 24:
            return w.bias_v
        if index in (8,):
            return w.tip_x
        if index in (9,):
            return w.tip_y
        if index == 16:
            # oscillation amplitude: the free amplitude while the drive is on, the sensor's
            # own noise floor when it is off (the client uses this to detect absent control)
            if getattr(w, "pll", None) is None:
                return 50e-12 + w.noise.rng.normal(0, 8e-12)
            return w.amplitude_now()
        if index == 17:
            if getattr(w, "pll", None) is None:
                return w.noise.rng.normal(0, 0.05)
            return w.df_now()
        if index == 18:
            pll = getattr(w, "pll", None)
            if pll is not None and pll.phas_ctrl_on and pll.output_on:
                return float(w.df_rng.normal(0, 0.5))
            return w.noise.rng.normal(0, 1e-5)
        if index == 19:
            pll = getattr(w, "pll", None)
            if pll is not None:
                return float(pll.excitation_v + w.df_rng.normal(0, 1e-4))
            return 0.01 + w.noise.rng.normal(0, 1e-4)
        if index == 15:
            return 0.01 + w.noise.rng.normal(0, 1e-4)
        if index in (20, 22):
            amp = w.lockin_amp_v if w.lockin_on else 0.0
            return w.current_now() * 0.05 * (amp / 0.01) if amp else w.noise.rng.normal(0, 1e-13)
        if index in (21, 23):
            return w.noise.rng.normal(0, 1e-13)
        if index == 25:
            return w.zctrl.setpoint_a
        if index == 27:
            return w.temperature_k + w.noise.rng.normal(0, 0.002)
        if index == 28:
            return w.temperature_k + 0.3
        if index == 29:
            return w.pressure_pa
        if index == 30:
            return w.noise.rng.normal(0, 3e-12)
        if 1 <= index <= 7:
            return w.noise.rng.normal(0, 1e-4)
        return w.noise.rng.normal(0, 1e-5)


def sxm_channel_name(index: int) -> tuple[str, str]:
    """(Name, Unit) as controller writes them in :DATA_INFO: — 'Z (m)' → ('Z', 'm')."""
    full = SIGNAL_NAMES[index] if 0 <= index < len(SIGNAL_NAMES) else f"Sig{index}"
    if " (" in full and full.endswith(")"):
        name, unit = full.rsplit(" (", 1)
        return name.replace(" ", "_"), unit[:-1]
    return full.replace(" ", "_"), ""
