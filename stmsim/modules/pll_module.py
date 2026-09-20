"""controller PLL / Oscillation Control, and the frequency sweep that measures f0 and Q.

Only bound on a profile that loads the ``PLL`` module (``reference-stm-qplus``); on the
plain STM profile every verb here still answers ``NeedModule``, because the module gate in
the wire server fires before the handler lookup. All 51 verbs are implemented even though an
agent needs six of them: the compatibility client calls most of the surface, and
the end-to-end contract test counts an unimplemented verb as a wire error.

One modulator, one demodulator: the index is validated (it must be ≥ 1, as controller counts
from one) but every index shares the same state. Additional indices are outside the
benchmark surface.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..physics.world import World
from ..wire.errors import BadArguments


def _idx(i, verb: str) -> int:
    v = int(i)
    if v < 1:
        raise BadArguments(verb, f"modulator/demodulator index must be ≥ 1, got {v}")
    return v


def bind_pll(d, w: World):
    if w.pll is None:
        from ..physics.qplus import PLLState
        w.pll = PLLState.from_rig(w.rig, w.tip)
    p = w.pll

    @d.handles("PLL_CenterFreqSet")
    def _cfset(i, f):
        _idx(i, "PLL.CenterFreqSet")
        p.center_freq_hz = float(f)

    d.register("PLL_CenterFreqGet", lambda i: float(p.center_freq_hz) if _idx(i, "PLL.CenterFreqGet") else 0.0)
    d.register("PLL_FreqShiftGet", lambda i: float(w.df_now()) if _idx(i, "PLL.FreqShiftGet") else 0.0)

    @d.handles("PLL_FreqShiftSet")
    def _fsset(i, f):
        _idx(i, "PLL.FreqShiftSet")
        p.df_setpoint_hz = float(f)

    @d.handles("PLL_FreqShiftAutoCenter")
    def _autocenter(i):
        """Re-zero Δf where the tip is now: shifts the centre frequency, not the physics."""
        _idx(i, "PLL.FreqShiftAutoCenter")
        with w.lock:
            if p.output_on and not w.withdrawn:
                amp = p.amplitude_now(w.clock.wall())
                gap = w.current_z_tip() - w.surface_height_here()
                p.center_freq_hz = float(w.tip.qplus_f0_hz + w.df_physical(gap, amp))
            else:
                p.center_freq_hz = float(w.tip.qplus_f0_hz)

    @d.handles("PLL_OutOnOffSet")
    def _outset(i, on):
        _idx(i, "PLL.OutOnOffSet")
        p.set_output(bool(int(on)), w.clock.wall())

    d.register("PLL_OutOnOffGet", lambda i: int(bool(p.output_on)) if _idx(i, "PLL.OutOnOffGet") or True else 0)

    @d.handles("PLL_AmpCtrlSetpntSet")
    def _ampset(i, sp):
        _idx(i, "PLL.AmpCtrlSetpntSet")
        p.amp_setpoint_m = float(sp)
        if p.output_on and p.amp_ctrl_on:
            p.excitation_v = p.excitation_for(p.amp_setpoint_m)

    d.register("PLL_AmpCtrlSetpntGet", lambda i: float(p.amp_setpoint_m) if _idx(i, "PLL.AmpCtrlSetpntGet") or True else 0.0)

    @d.handles("PLL_AmpCtrlOnOffSet")
    def _ampon(i, s):
        _idx(i, "PLL.AmpCtrlOnOffSet")
        p.amp_ctrl_on = bool(int(s))

    d.register("PLL_AmpCtrlOnOffGet", lambda i: int(bool(p.amp_ctrl_on)))

    @d.handles("PLL_PhasCtrlOnOffSet")
    def _phason(i, s):
        _idx(i, "PLL.PhasCtrlOnOffSet")
        p.phas_ctrl_on = bool(int(s))

    d.register("PLL_PhasCtrlOnOffGet", lambda i: int(bool(p.phas_ctrl_on)))

    @d.handles("PLL_ExcitationSet")
    def _excset(i, v):
        _idx(i, "PLL.ExcitationSet")
        p.excitation_v = float(v)

    d.register("PLL_ExcitationGet", lambda i: float(p.excitation_v))

    @d.handles("PLL_AmpCtrlGainSet")
    def _ampgain(i, pg, tc):
        _idx(i, "PLL.AmpCtrlGainSet")
        p.amp_p_v_per_m, p.amp_tc_s = float(pg), float(tc)

    d.register("PLL_AmpCtrlGainGet",
               lambda i: [p.amp_p_v_per_m, p.amp_tc_s, p.amp_p_v_per_m / max(p.amp_tc_s, 1e-12)])
    d.register("PLL_AmpCtrlBandwidthSet", lambda i, bw: setattr(p, "amp_bw_hz", float(bw)))
    d.register("PLL_AmpCtrlBandwidthGet", lambda i: float(p.amp_bw_hz))

    @d.handles("PLL_PhasCtrlGainSet")
    def _phasgain(i, pg, tc):
        _idx(i, "PLL.PhasCtrlGainSet")
        p.phas_p_hz_per_deg, p.phas_tc_s = float(pg), float(tc)

    d.register("PLL_PhasCtrlGainGet", lambda i: [p.phas_p_hz_per_deg, p.phas_tc_s])
    d.register("PLL_PhasCtrlBandwidthSet", lambda i, bw: setattr(p, "phas_bw_hz", float(bw)))
    d.register("PLL_PhasCtrlBandwidthGet", lambda i: float(p.phas_bw_hz))
    d.register("PLL_ExcRangeSet", lambda i, r: setattr(p, "exc_range_idx", int(r)))
    d.register("PLL_ExcRangeGet", lambda i: int(p.exc_range_idx))
    d.register("PLL_FreqRangeSet", lambda i, f: setattr(p, "freq_range_hz", float(f)))
    d.register("PLL_FreqRangeGet", lambda i: float(p.freq_range_hz))
    d.register("PLL_InpCalibrSet", lambda i, c: setattr(p, "input_calibration_m_per_v", float(c)))
    d.register("PLL_InpCalibrGet", lambda i: float(p.input_calibration_m_per_v))
    d.register("PLL_InpRangeSet", lambda i, r: setattr(p, "inp_range_m", float(r)))
    d.register("PLL_InpRangeGet", lambda i: float(p.inp_range_m))

    @d.handles("PLL_InpPropsSet")
    def _inpprops(i, diff, div10):
        p.inp_diff, p.inp_div10 = int(diff), int(div10)

    d.register("PLL_InpPropsGet", lambda i: [int(p.inp_diff), int(p.inp_div10)])
    d.register("PLL_DemodFilterSet", lambda i, o: setattr(p, "demod_filter_order", int(o)))
    d.register("PLL_DemodFilterGet", lambda i: int(p.demod_filter_order))
    d.register("PLL_DemodHarmonicSet", lambda i, h: setattr(p, "demod_harmonic", int(h)))
    d.register("PLL_DemodHarmonicGet", lambda i: int(p.demod_harmonic))

    @d.handles("PLL_DemodInputSet")
    def _demodin(i, inp, gen):
        p.demod_input, p.demod_freq_gen = int(inp), int(gen)

    d.register("PLL_DemodInputGet", lambda i: [int(p.demod_input), int(p.demod_freq_gen)])
    d.register("PLL_DemodPhasRefSet", lambda i, ph: setattr(p, "demod_phase_ref_deg", float(ph)))
    d.register("PLL_DemodPhasRefGet", lambda i: float(p.demod_phase_ref_deg))
    d.register("PLL_AddOnOffSet", lambda i, a: setattr(p, "add_on", int(a)))
    d.register("PLL_AddOnOffGet", lambda i: int(p.add_on))

    @d.handles("PLL_FreqExcOverwriteSet")
    def _overwrite(i, exc, freq):
        p.exc_overwrite_idx, p.freq_overwrite_idx = int(exc), int(freq)

    d.register("PLL_FreqExcOverwriteGet", lambda i: [int(p.exc_overwrite_idx), int(p.freq_overwrite_idx)])
    d.register("PLL_PerfectPLLUpdtZTC", lambda i: None)


def bind_pllfreqswp(d, w: World):
    """The resonance sweep used to read f0 and Q from the sensor."""
    if w.pll is None:
        from ..physics.qplus import PLLState
        w.pll = PLLState.from_rig(w.rig, w.tip)
    p = w.pll

    d.register("PLLFreqSwp_Open", lambda i: None)
    d.register("PLLFreqSwp_Stop", lambda i: None)

    @d.handles("PLLFreqSwp_ParamsSet")
    def _params(i, n, period, settle):
        p.swp_n, p.swp_period_s, p.swp_settle_s = int(n), float(period), float(settle)

    d.register("PLLFreqSwp_ParamsGet", lambda i: [int(p.swp_n), float(p.swp_period_s),
                                                  float(p.swp_settle_s)])

    @d.handles("PLLFreqSwp_Start")
    def _start(i, get_data, direction):
        _idx(i, "PLLFreqSwp.Start")
        n = max(int(p.swp_n), 8)
        f0 = float(w.tip.qplus_f0_hz)
        q = float(w.tip.qplus_q)
        span = float(p.freq_range_hz)
        f = np.linspace(f0 - span / 2, f0 + span / 2, n)
        drive = p.amp_setpoint_m if p.output_on else p.undriven_floor_m
        x = f / f0
        amp = drive / np.sqrt((1 - x ** 2) ** 2 + (x / q) ** 2)
        phase = np.degrees(np.arctan2(x / q, 1 - x ** 2)) - 90.0
        amp = amp + w.df_rng.normal(0, 1e-13, n)
        time.sleep(min(n * p.swp_period_s, 4.0))
        names = ["Frequency (Hz)", "Amplitude (m)", "Phase (deg)"]
        data = np.vstack([f, amp, phase]).astype(np.float32)
        f_res = f0 * (1.0 + float(w.df_rng.normal(0, 1e-3)))
        if not int(get_data):
            return [None, None, [], 0, 0, [], f_res, q, 0.0, 0.0, 0, 0]
        return [None, None, names, len(names), n, data.tolist(), float(f_res), float(q),
                -90.0, float(p.amp_to_excitation_m_per_v), 0, 0]
