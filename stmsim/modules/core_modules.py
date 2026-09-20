"""Bias / ZCtrl / Current / FolMe / Piezo / Signals / Util / SafeTip / LockIn / Marks / DataLog.

Each ``bind_*`` registers handlers on a :class:`Dispatcher`. Handlers get decoded
positional arguments and return values aligned with the local command registry;
``None`` count slots are filled by the codec.
"""
from __future__ import annotations

import math
import time

from ..wire.errors import BadArguments, WireError
from ..physics.world import ZCTRL_OFF, ZCTRL_ON, World
from .signals import SignalTable

_BIAS_RANGES = ["10 V", "1 V", "100 mV"]
_GAIN_NAMES = ["1E6", "1E7", "1E8", "1E9", "1E10", "1E11"]
_FILTER_NAMES = ["none", "1 kHz", "10 kHz"]
W_TILT_MAX_DEG = 45.0       # Piezo.TiltSet acceptance window per axis (a real sample is off by ≪ 5°)


def bind_bias(d, w: World):
    @d.handles("Bias_Set")
    def _set(v):
        if abs(v) > 10.0:
            raise BadArguments("Bias.Set", "outside ±10 V range")
        w.set_bias(v)

    d.register("Bias_Get", lambda: w.bias_v)
    d.register("Bias_RangeSet", lambda idx: setattr(w, "bias_range_index", int(idx)))
    d.register("Bias_RangeGet", lambda: [None, None, _BIAS_RANGES, w.bias_range_index])
    d.register("Bias_CalibrGet", lambda: [1.0, 0.0])

    @d.handles("Bias_Pulse")
    def _pulse(wait, width_s, value_v, z_hold, rel_abs):
        v = float(value_v) if int(rel_abs) == 1 else w.bias_v + float(value_v)
        w.bias_pulse(float(width_s), v, bool(int(z_hold)), int(rel_abs))
        if int(wait):
            time.sleep(min(float(width_s), 4.0))


def bind_zctrl(d, w: World):
    z = w.zctrl
    d.register("ZCtrl_OnOffSet", lambda on: w.zctrl_set(bool(int(on))))
    d.register("ZCtrl_OnOffGet", lambda: 1 if z.on else 0)
    d.register("ZCtrl_SetpntSet", lambda i: w.set_setpoint(float(i)))
    d.register("ZCtrl_SetpntGet", lambda: z.setpoint_a)
    d.register("ZCtrl_ZPosSet", lambda zz: w.set_z_position(float(zz)))
    d.register("ZCtrl_ZPosGet", lambda: w.z_now())

    @d.handles("ZCtrl_Withdraw")
    def _withdraw(wait, timeout_ms):
        w.withdraw()

    d.register("ZCtrl_Home", lambda: w.home())
    d.register("ZCtrl_StatusGet", lambda: z.status if not z.on else ZCTRL_ON)
    d.register("ZCtrl_CtrlListGet", lambda: [None, None, z.controllers, z.active_index])

    @d.handles("ZCtrl_ActiveCtrlSet")
    def _active(idx):
        z.active_index = int(idx)

    @d.handles("ZCtrl_GainSet")
    def _gain_set(p, t, i):
        z.p_m, z.time_const_s, z.i_m_per_s = float(p), float(t), float(i)

    d.register("ZCtrl_GainGet", lambda: [z.p_m, z.time_const_s, z.i_m_per_s])
    d.register("ZCtrl_LimitsGet", lambda: [z.limit_high_m, z.limit_low_m])

    @d.handles("ZCtrl_LimitsSet")
    def _lim(hi, lo):
        z.limit_high_m, z.limit_low_m = float(hi), float(lo)

    d.register("ZCtrl_LimitsEnabledSet", lambda on: setattr(z, "limits_enabled", bool(int(on))))
    d.register("ZCtrl_LimitsEnabledGet", lambda: 1 if z.limits_enabled else 0)
    d.register("ZCtrl_TipLiftSet", lambda m: setattr(z, "tip_lift_m", float(m)))
    d.register("ZCtrl_TipLiftGet", lambda: z.tip_lift_m)
    d.register("ZCtrl_SwitchOffDelaySet", lambda s: setattr(z, "switch_off_delay_s", float(s)))
    d.register("ZCtrl_SwitchOffDelayGet", lambda: z.switch_off_delay_s)
    d.register("ZCtrl_HomePropsSet", lambda rel, pos: None)
    d.register("ZCtrl_HomePropsGet", lambda: [1, 0.0])
    rate = {"v": 30e-9}
    d.register("ZCtrl_WithdrawRateSet", lambda v: rate.__setitem__("v", float(v)))
    d.register("ZCtrl_WithdrawRateGet", lambda: rate["v"])


def bind_current(d, w: World):
    d.register("Current_Get", lambda: w.current_now())
    d.register("Current_100Get", lambda: w.current_now())
    d.register("Current_GainsGet", lambda: [None, None, _GAIN_NAMES, w.preamp.gain_index, None, None, _FILTER_NAMES, 0])

    @d.handles("Current_GainSet")
    def _gain(idx, filt):
        w.preamp.gain_index = int(idx)
        w.preamp.full_scale_a = 10.0 / (10.0 ** (6 + int(idx)))

    d.register("Current_CalibrGet", lambda idx: [10.0 ** (-(6 + int(idx))), 0.0])


def bind_folme(d, w: World):
    st = {"speed": 293e-9, "custom": 0, "oversampl": 10, "ps": 0}

    @d.handles("FolMe_XYPosSet")
    def _set(x, y, wait):
        rx, ry = w.rig.xy_range_m
        if abs(x) > rx / 2 or abs(y) > ry / 2:
            raise BadArguments("FolMe.XYPosSet", "outside piezo range")
        dist = ((x - w.tip_x) ** 2 + (y - w.tip_y) ** 2) ** 0.5
        # A FolMe move is the path used to drag an atom along: integrate it
        # through the adatom registry at the FolMe speed. Before the path integration fix,
        # (``move_xy``) moved directly, so no manipulation reached the physics from the wire;
        # every MoveAtomTo, in every mode, reported "moved" over an atom that had not moved.
        # The 2026-09-11 P4/P3-repair trials exposed this discrepancy. The path integration
        # keeps wire-level atom manipulation connected to the physics model.
        w.move_xy_path(float(x), float(y), speed_m_s=float(st["speed"]))
        if int(wait):
            time.sleep(min(dist / max(st["speed"], 1e-9), 4.0))

    d.register("FolMe_XYPosGet", lambda wait: [w.tip_x, w.tip_y])

    @d.handles("FolMe_SpeedSet")
    def _speed(s, custom):
        st["speed"], st["custom"] = float(s), int(custom)

    d.register("FolMe_SpeedGet", lambda: [st["speed"], st["custom"]])
    d.register("FolMe_OversamplSet", lambda o: st.__setitem__("oversampl", int(o)))
    d.register("FolMe_OversamplGet", lambda: [st["oversampl"], w.rig.rt_freq_hz / st["oversampl"]])
    d.register("FolMe_PSOnOffSet", lambda on: st.__setitem__("ps", int(on)))
    d.register("FolMe_PSOnOffGet", lambda: st["ps"])
    d.register("FolMe_Stop", lambda: None)
    d.register("FolMe_PSPropsGet", lambda: [0, 0, None, "", None, "", 0.0])


def bind_piezo(d, w: World):
    rx, ry = w.rig.xy_range_m
    st = {"rx": rx, "ry": ry, "rz": 2 * w.rig.z_range_m,
          "drift": [0, 0.0, 0.0, 0.0, 0, 0, 0, 0.9]}
    d.register("Piezo_RangeGet", lambda: [st["rx"], st["ry"], st["rz"]])

    @d.handles("Piezo_RangeSet")
    def _rset(x, y, zz):
        st["rx"], st["ry"], st["rz"] = float(x), float(y), float(zz)

    d.register("Piezo_SensGet", lambda: [float(w.rig.get("xy.sens_x_m_per_v", 8.9e-9)),
                                         float(w.rig.get("xy.sens_y_m_per_v", 8.7e-9)), 3.0e-9])
    d.register("Piezo_TiltGet", lambda: list(w.piezo_tilt_deg))
    # Piezo.CalibrGet: m/V of the ±10 V low-voltage signals (before the HV amplifier), so
    # half range = calibration × 10 V — imaging and readback clients use this value.
    # Piezo_RangeGet reports FULL ranges here, hence /20.
    d.register("Piezo_CalibrGet", lambda: [float(st["rx"]) / 20.0, float(st["ry"]) / 20.0,
                                           float(st["rz"]) / 20.0])

    @d.handles("Piezo_TiltSet")
    def _tilt(tx, ty):
        # Piezo.TiltSet(Tilt_X_deg, Tilt_Y_deg): the scanner's tilt correction — a plane
        # added to the Z output (World.set_piezo_tilt documents the convention). controller
        # takes ±90° in the panel; anything approaching that is a typo, not a levelling.
        tx, ty = float(tx), float(ty)
        if not (math.isfinite(tx) and math.isfinite(ty)) or max(abs(tx), abs(ty)) > W_TILT_MAX_DEG:
            raise BadArguments("Piezo.TiltSet", f"tilt outside ±{W_TILT_MAX_DEG:g}°")
        w.set_piezo_tilt(tx, ty)

    d.register("Piezo_DriftCompGet", lambda: list(st["drift"]))
    # Piezo.XYZLimitsGet: [enabled, X low, X high, Y low, Y high, Z low, Z high] in **volts**
    # at the ±10 V DAC — not in metres. The client turns them into a reachable half range as
    # calibration × min(10 V, |limits|), so metres here would report a scanner with no travel
    # and every ConfigureScan would be refused. The XY limits are the full DAC swing, which
    # makes the half range exactly Piezo.RangeGet / 2; Z carries the controller's own limits.
    def _limits():
        cz = st["rz"] / 20.0
        zv = [w.zctrl.limit_low_m / cz, w.zctrl.limit_high_m / cz] if cz else [-10.0, 10.0]
        zlo, zhi = (max(-10.0, min(10.0, v)) for v in zv)
        return [1, -10.0, 10.0, -10.0, 10.0, zlo, zhi]

    d.register("Piezo_XYZLimitsGet", _limits)

    @d.handles("Piezo_DriftCompSet")
    def _drift(on, vx, vy, vz, sat):
        # not just panel state: this drives the physics. World.set_drift_comp documents the
        # sign — v is the velocity at which features are observed to move, so a measured
        # two-frame drift estimate can be applied to stop the apparent motion.
        st["drift"] = [int(on), float(vx), float(vy), float(vz), 0, 0, 0, float(sat)]
        w.set_drift_comp(bool(int(on)), float(vx), float(vy), float(vz), float(sat))


def bind_signals(d, w: World):
    tab = SignalTable(w)
    d.register("Signals_NamesGet", lambda: [None, None, tab.names()])
    d.register("Signals_ValGet", lambda idx, wait: tab.value(int(idx)))
    d.register("Signals_ValsGet", lambda idxs, wait: [None, [tab.value(int(i)) for i in idxs]])
    d.register("Signals_CalibrGet", lambda idx: [1.0, 0.0])
    d.register("Signals_RangeGet", lambda idx: [10.0, -10.0])
    d.register("Signals_MeasNamesGet", lambda: [None, None, tab.names()[:24]])


def bind_util(d, w: World):
    d.register("Util_VersionGet", lambda: ["controller SPM Controller (stmsim)", "Generic 5e", 15016, 15016])
    d.register("Util_SessionPathGet", lambda: [None, str(w.session_dir)])

    @d.handles("Util_SessionPathSet")
    def _sp(path, save):
        from pathlib import Path
        w.session_dir = Path(path)

    d.register("Util_RTFreqGet", lambda: w.rig.rt_freq_hz)
    d.register("Util_AcqPeriodGet", lambda: w.rig.osci_dt_s)
    d.register("Util_RTOversamplGet", lambda: int(round(w.rig.rt_freq_hz * w.rig.osci_dt_s)))
    d.register("Util_Lock", lambda: None)
    d.register("Util_UnLock", lambda: None)
    d.register("Util_SettingsLoad", lambda path, sess: None)
    d.register("Util_SettingsSave", lambda path, sess: None)


def bind_safetip(d, w: World):
    st = {"on": 1, "auto_rec": 1, "auto_pause": 1, "thr": 5e-9}
    d.register("SafeTip_OnOffSet", lambda on: st.__setitem__("on", int(on)))
    d.register("SafeTip_OnOffGet", lambda: st["on"])
    d.register("SafeTip_SignalGet", lambda: abs(w.current_now()))

    @d.handles("SafeTip_PropsSet")
    def _props(rec, pause, thr):
        st.update(auto_rec=int(rec), auto_pause=int(pause), thr=float(thr))

    d.register("SafeTip_PropsGet", lambda: [st["auto_rec"], st["auto_pause"], st["thr"]])


def bind_lockin(d, w: World):
    mods: dict[int, dict] = {}
    demods: dict[int, dict] = {}

    def m(i):
        return mods.setdefault(int(i), {"on": 0, "signal": 24, "amp": 0.01, "freq": 973.0,
                                        "phasreg": 1, "harm": 1, "phase": 0.0})

    def dm(i):
        return demods.setdefault(int(i), {"signal": 0, "harm": 1, "hp": [1, 0.0], "lp": [1, 100.0],
                                          "phasreg": 1, "phase": 0.0, "sync": 0, "rt": 0})

    @d.handles("LockIn_ModOnOffSet")
    def _on(i, on):
        m(i)["on"] = int(on)
        if int(i) == 1:
            w.lockin_on = bool(int(on))

    d.register("LockIn_ModOnOffGet", lambda i: m(i)["on"])
    d.register("LockIn_ModSignalSet", lambda i, s: m(i).__setitem__("signal", int(s)))
    d.register("LockIn_ModSignalGet", lambda i: m(i)["signal"])

    @d.handles("LockIn_ModAmpSet")
    def _amp(i, a):
        m(i)["amp"] = float(a)
        if int(i) == 1:
            w.lockin_amp_v = float(a)

    d.register("LockIn_ModAmpGet", lambda i: m(i)["amp"])

    @d.handles("LockIn_ModPhasFreqSet")
    def _freq(i, f):
        m(i)["freq"] = float(f)
        if int(i) == 1:
            w.lockin_freq_hz = float(f)

    d.register("LockIn_ModPhasFreqGet", lambda i: m(i)["freq"])
    d.register("LockIn_ModPhasRegSet", lambda i, r: m(i).__setitem__("phasreg", int(r)))
    d.register("LockIn_ModPhasRegGet", lambda i: m(i)["phasreg"])
    d.register("LockIn_ModHarmonicSet", lambda i, h: m(i).__setitem__("harm", int(h)))
    d.register("LockIn_ModHarmonicGet", lambda i: m(i)["harm"])
    d.register("LockIn_ModPhasSet", lambda i, p: m(i).__setitem__("phase", float(p)))
    d.register("LockIn_ModPhasGet", lambda i: m(i)["phase"])
    d.register("LockIn_DemodSignalSet", lambda i, s: dm(i).__setitem__("signal", int(s)))
    d.register("LockIn_DemodSignalGet", lambda i: dm(i)["signal"])
    d.register("LockIn_DemodHarmonicSet", lambda i, h: dm(i).__setitem__("harm", int(h)))
    d.register("LockIn_DemodHarmonicGet", lambda i: dm(i)["harm"])
    d.register("LockIn_DemodHPFilterSet", lambda i, o, f: dm(i).__setitem__("hp", [int(o), float(f)]))
    d.register("LockIn_DemodHPFilterGet", lambda i: list(dm(i)["hp"]))
    d.register("LockIn_DemodLPFilterSet", lambda i, o, f: dm(i).__setitem__("lp", [int(o), float(f)]))
    d.register("LockIn_DemodLPFilterGet", lambda i: list(dm(i)["lp"]))
    d.register("LockIn_DemodPhasRegSet", lambda i, r: dm(i).__setitem__("phasreg", int(r)))
    d.register("LockIn_DemodPhasRegGet", lambda i: dm(i)["phasreg"])
    d.register("LockIn_DemodPhasSet", lambda i, p: dm(i).__setitem__("phase", float(p)))
    d.register("LockIn_DemodPhasGet", lambda i: dm(i)["phase"])
    d.register("LockIn_DemodSyncFilterSet", lambda i, s: dm(i).__setitem__("sync", int(s)))
    d.register("LockIn_DemodSyncFilterGet", lambda i: dm(i)["sync"])
    d.register("LockIn_DemodRTSignalsSet", lambda i, s: dm(i).__setitem__("rt", int(s)))
    d.register("LockIn_DemodRTSignalsGet", lambda i: dm(i)["rt"])


def bind_marks(d, w: World):
    d.register("Marks_PointsGet", lambda: [0, [], [], 0, [], [], []])
    d.register("Marks_LinesGet", lambda: [0, [], [], [], [], [], []])
    d.register("Marks_PointsErase", lambda i: None)
    d.register("Marks_LinesErase", lambda i: None)
    d.register("Marks_PointsVisibleSet", lambda i, s: None)
    d.register("Marks_LinesVisibleSet", lambda i, s: None)


def bind_datalog(d, w: World):
    st = {"running": 0, "chs": [0, 14], "props": [0, 0, 0, 0.0, 1, "log", "", []]}
    d.register("DataLog_Open", lambda: None)
    d.register("DataLog_Start", lambda: st.__setitem__("running", 1))
    d.register("DataLog_Stop", lambda: st.__setitem__("running", 0))
    d.register("DataLog_StatusGet", lambda: [None, "", st["running"], 0, 0.0, None, "", None, "", 0])
    d.register("DataLog_ChsSet", lambda chs: st.__setitem__("chs", list(chs)))
    d.register("DataLog_ChsGet", lambda: [None, st["chs"]])

    @d.handles("DataLog_PropsSet")
    def _props(mode, h, m, s, avg, base, comment, modules):
        st["props"] = [int(mode), int(h), int(m), float(s), int(avg), base, comment, modules]

    d.register("DataLog_PropsGet", lambda: [st["props"][0], st["props"][1], st["props"][2], st["props"][3],
                                            st["props"][4], None, st["props"][5], None, st["props"][6]])
