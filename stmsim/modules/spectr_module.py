"""BiasSpectr (I(V), and I(z) via z_offset the way MeasureBarrierHeight does it) and ZSpectr."""
from __future__ import annotations

import time

import numpy as np

from ..physics.world import World
from ..wire.errors import WireError
from .signals import SIGNAL_NAMES

#: every command must answer inside the client's 5 s recv timeout
MAX_REPLY_S = 4.0

_PARAM_NAMES = ["Sweep Start", "Sweep End", "X (m)", "Y (m)", "Z (m)", "Z offset (m)", "Settling time (s)",
                "Integration time (s)", "Z-Ctrl hold", "Final Z (m)"]


def bind_biasspectr(d, w: World):
    st = {"open": False, "running": False, "chs": [0], "save_all": 0, "sweeps": 1, "bwd": 1,
          "points": 200, "z_offset": 0.0, "autosave": 0, "dialog": 0,
          "v0": -1.0, "v1": 1.0,
          "timing": [0.1, 0.0, 0.3, 1.0, 0.005, 0.005, 0.1, 0.1],   # z_avg, z_off, init_settle, slew, settle, integ, end, zctrl
          "adv": [1, 1, 0, 0], "altz": [0, 50e-12, 0.1], "ttl": [0, 0, 0.0, 0.0],
          "mls_per_seg": 0, "mls_mode": "Linear"}

    d.register("BiasSpectr_Open", lambda: st.__setitem__("open", True))
    d.register("BiasSpectr_Stop", lambda: st.__setitem__("running", False))
    d.register("BiasSpectr_StatusGet", lambda: None)
    d.register("BiasSpectr_ChsSet", lambda chs: st.__setitem__("chs", [int(c) for c in chs]))
    d.register("BiasSpectr_ChsGet", lambda: [None, st["chs"], None, None, [SIGNAL_NAMES[c] for c in st["chs"]]])

    @d.handles("BiasSpectr_PropsSet")
    def _props(save_all, sweeps, bwd, points, z_off, autosave, dialog):
        if int(save_all) != 0:
            st["save_all"] = 1 if int(save_all) == 1 else 0
        if int(sweeps) > 0:
            st["sweeps"] = int(sweeps)
        if int(bwd) != 0:
            st["bwd"] = 1 if int(bwd) == 1 else 0
        if int(points) > 0:
            st["points"] = int(points)
        st["z_offset"] = float(z_off)
        if int(autosave) != 0:
            st["autosave"] = 1 if int(autosave) == 1 else 0
        if int(dialog) != 0:
            st["dialog"] = 1 if int(dialog) == 1 else 0

    d.register("BiasSpectr_PropsGet", lambda: [st["save_all"], st["sweeps"], st["bwd"], st["points"],
                                               None, None, [SIGNAL_NAMES[c] for c in st["chs"]],
                                               None, None, _PARAM_NAMES, None, None, ["Sweep Start", "Sweep End"]])

    @d.handles("BiasSpectr_LimitsSet")
    def _lim(v0, v1):
        st["v0"], st["v1"] = float(v0), float(v1)

    d.register("BiasSpectr_LimitsGet", lambda: [st["v0"], st["v1"]])

    @d.handles("BiasSpectr_TimingSet")
    def _timing(*vals):
        st["timing"] = [float(v) for v in vals]
        st["z_offset"] = float(vals[1])

    d.register("BiasSpectr_TimingGet", lambda: list(st["timing"]))

    @d.handles("BiasSpectr_AdvPropsSet")
    def _adv(*vals):
        st["adv"] = [int(v) for v in vals]

    d.register("BiasSpectr_AdvPropsGet", lambda: list(st["adv"]))

    @d.handles("BiasSpectr_AltZCtrlSet")
    def _altz(on, sp, settle):
        st["altz"] = [int(on), float(sp), float(settle)]

    d.register("BiasSpectr_AltZCtrlGet", lambda: list(st["altz"]))

    @d.handles("BiasSpectr_TTLSyncSet")
    def _ttl(*vals):
        st["ttl"] = [int(vals[0]), int(vals[1]), float(vals[2]), float(vals[3])]

    d.register("BiasSpectr_TTLSyncGet", lambda: list(st["ttl"]))
    d.register("BiasSpectr_MLSLockinPerSegSet", lambda v: st.__setitem__("mls_per_seg", int(v)))
    d.register("BiasSpectr_MLSLockinPerSegGet", lambda: st["mls_per_seg"])
    d.register("BiasSpectr_MLSModeSet", lambda m: st.__setitem__("mls_mode", str(m)))
    d.register("BiasSpectr_MLSModeGet", lambda: [None, st["mls_mode"]])
    d.register("BiasSpectr_MLSValsSet", lambda *a: None)
    d.register("BiasSpectr_MLSValsGet", lambda: [1, [st["v0"]], [st["v1"]], [0.3], [0.005], [0.005], [st["points"]], [0]])

    @d.handles("BiasSpectr_Start")
    def _start(get_data, basename):
        if w.withdrawn:
            raise WireError("Bias Spectroscopy: no tunnelling junction (tip withdrawn)")
        n = st["points"]
        settle, integ = st["timing"][4], st["timing"][5]
        z_off = st["z_offset"] if st["z_offset"] else st["timing"][1]
        st["running"] = True
        t_compute = time.perf_counter()
        cur = w.sts_curve(st["v0"], st["v1"], n, z_offset_m=z_off, settle_s=settle, integ_s=integ)
        duration = st["timing"][2] + n * (settle + integ) * (2 if st["bwd"] else 1) + st["timing"][6]
        # the acquisition's wall-clock cost counts toward the sweep time, so a spectrum that
        # was expensive to compute does not also wait the full nominal duration on top and
        # blow the client's 5 s reply window
        time.sleep(max(0.0, min(duration, MAX_REPLY_S) - (time.perf_counter() - t_compute)))
        st["running"] = False
        names = ["Bias calc (V)"]
        rows = [cur["v"]]
        for c in st["chs"]:
            nm = SIGNAL_NAMES[c]
            if c in (0, 26):
                rows.append(cur["i"])
                names.append(nm)
                if st["bwd"]:
                    rows.append(cur["i"][::-1] * (1 + w.rng.normal(0, 0.02, n)))
                    names.append("Current [bwd] (A)")
            elif c in (20, 22):
                rows.append(cur["didv"] * (w.lockin_amp_v if w.lockin_on else 1e-3))
                names.append(nm)
            else:
                rows.append(np.zeros(n))
                names.append(nm)
        data = np.vstack(rows).astype(np.float32)
        # controller autosaves the sweep; every analysis skill takes a .dat path, so the
        # simulator has to leave one behind or the analysis half of a task is unreachable
        if st["autosave"] or str(basename or "").strip():
            from ..io.dat_writer import save_bias_spectrum
            path = save_bias_spectrum(w, str(basename or ""), v=cur["v"], i=cur["i"],
                                      didv=cur["didv"])
            w.dats_saved.append(path)
            if w.sts_records:
                w.sts_records[-1]["path"] = path
            w.sts_last["path"] = path
            w._event("dat_saved", path=path, experiment="bias spectroscopy",
                     idx=len(w.dats_saved), sts_idx=w.sts_counter)
        params = [st["v0"], st["v1"], w.tip_x, w.tip_y, w.zctrl.z_n, z_off, settle, integ,
                  float(st["adv"][1]), w.zctrl.z_n]
        if not int(get_data):
            return [None, None, [], 0, 0, [], None, []]
        return [None, None, names, None, None, data.tolist(), None, params]


def bind_zspectr(d, w: World):
    """Z spectroscopy — I(z) on an STM rig, Δf(z) once a qPlus sensor is oscillating.

    ``Z rel`` follows the controller-file convention: 0 at the start height and
    **negative toward the surface**, with ``Sweep End (m) = −distance`` in the header."""
    st = {"chs": [0], "bwd": 1, "points": 100, "sweeps": 1, "autosave": 0, "dialog": 0, "save_all": 0,
          "timing": [0.1, 0.3, 1e-6, 0.005, 0.005, 0.1, 0.1], "retract": [0, 0.0, 0, 0],
          "adv": [0.0, 0, 0, 1], "z_offset": 0.0, "z_sweep": None, "running": False,
          "retract_delay": 0.0, "retract_second": [0, 0.0, 0, 0], "ttl": [0, 0, 0.0, 0.0],
          "dig_sync": 0, "pulse_seq": [0, 0]}
    d.register("ZSpectr_Open", lambda: None)
    d.register("ZSpectr_Stop", lambda: st.__setitem__("running", False))
    d.register("ZSpectr_StatusGet", lambda: 1 if st["running"] else 0)
    d.register("ZSpectr_ChsSet", lambda chs: st.__setitem__("chs", [int(c) for c in chs]))
    d.register("ZSpectr_ChsGet", lambda: [None, st["chs"], None, None, [SIGNAL_NAMES[c] for c in st["chs"]]])

    @d.handles("ZSpectr_PropsSet")
    def _props(bwd, points, sweeps, autosave, dialog, save_all):
        # controller tri-state: 0 = leave alone, 1 = on, 2 = off
        if int(bwd):
            st["bwd"] = 1 if int(bwd) == 1 else 0
        if int(points) > 0:
            st["points"] = int(points)
        if int(sweeps) > 0:
            st["sweeps"] = int(sweeps)
        if int(autosave):
            st["autosave"] = 1 if int(autosave) == 1 else 0
        if int(dialog):
            st["dialog"] = 1 if int(dialog) == 1 else 0
        if int(save_all):
            st["save_all"] = 1 if int(save_all) == 1 else 0

    d.register("ZSpectr_PropsGet", lambda: [st["bwd"], st["points"], None, None, [SIGNAL_NAMES[c] for c in st["chs"]],
                                            None, None, _PARAM_NAMES, st["autosave"], st["save_all"]])

    @d.handles("ZSpectr_RangeSet")
    def _range(z_offset, z_sweep):
        st["z_offset"] = float(z_offset)
        st["z_sweep"] = abs(float(z_sweep))

    d.register("ZSpectr_RangeGet", lambda: [st["z_offset"], _sweep()])

    @d.handles("ZSpectr_TimingSet")
    def _timing(*vals):
        st["timing"] = [float(v) for v in vals]

    d.register("ZSpectr_TimingGet", lambda: list(st["timing"]))

    @d.handles("ZSpectr_RetractSet")
    def _retract(en, thr, sig, comp):
        st["retract"] = [int(en), float(thr), int(sig), int(comp)]

    d.register("ZSpectr_RetractGet", lambda: list(st["retract"]))
    d.register("ZSpectr_RetractDelaySet", lambda s: st.__setitem__("retract_delay", float(s)))
    d.register("ZSpectr_RetractDelayGet", lambda: st["retract_delay"])

    @d.handles("ZSpectr_RetractSecondSet")
    def _retract2(cond, thr, sig, comp):
        st["retract_second"] = [int(cond), float(thr), int(sig), int(comp)]

    d.register("ZSpectr_RetractSecondGet", lambda: list(st["retract_second"]))

    @d.handles("ZSpectr_TTLSyncSet")
    def _ttl(line, pol, t_on, dur):
        st["ttl"] = [int(line), int(pol), float(t_on), float(dur)]

    d.register("ZSpectr_TTLSyncGet", lambda: list(st["ttl"]))
    d.register("ZSpectr_DigSyncSet", lambda v: st.__setitem__("dig_sync", int(v)))
    d.register("ZSpectr_DigSyncGet", lambda: st["dig_sync"])

    @d.handles("ZSpectr_PulseSeqSyncSet")
    def _pulse(nr, periods):
        st["pulse_seq"] = [int(nr), int(periods)]

    d.register("ZSpectr_PulseSeqSyncGet", lambda: list(st["pulse_seq"]))

    @d.handles("ZSpectr_AdvPropsSet")
    def _adv(t, rec, li, reset):
        st["adv"] = [float(t), int(rec), int(li), int(reset)]

    d.register("ZSpectr_AdvPropsGet", lambda: list(st["adv"]))

    def _sweep() -> float:
        if st["z_sweep"] is not None:
            return float(st["z_sweep"])
        return float(getattr(w, "z_sweep_m", None) or w.rig.get("spectroscopy.z_sweep_m", 0.2e-9))

    @d.handles("ZSpectr_Start")
    def _start(get_data, basename):
        if w.withdrawn:
            raise WireError("Z Spectroscopy: no tunnelling junction (tip withdrawn)")
        n = st["points"]
        sweep = _sweep()
        settle, integ = st["timing"][3], st["timing"][4]
        st["running"] = True
        cur = w.zspec_curve(n, sweep, z_offset_m=st["z_offset"], bwd=bool(st["bwd"]),
                            sweeps=st["sweeps"], settle_s=settle, integ_s=integ,
                            retract=tuple(st["retract"]))
        duration = st["timing"][1] + n * (settle + integ) * (2 if st["bwd"] else 1) * st["sweeps"] \
            + st["timing"][5]
        time.sleep(min(duration, 4.0))
        st["running"] = False
        z_rel, i, df, amp = cur["z_rel"], cur["i"], cur["df"], cur["amp"]
        z_abs = np.asarray(cur["z_n"], float)
        names = ["Z rel (m)"]
        rows = [z_rel]
        for c in st["chs"]:
            if c in (0, 26):
                col = i
            elif c == 17:
                col = df
            elif c == 16:
                col = amp
            elif c == 14:
                col = z_abs
            elif c == 24:
                col = np.full(n, w.bias_v)
            elif c == 19:
                col = np.full(n, w.pll.excitation_v if w.pll is not None else 0.0)
            else:
                col = np.zeros(n)
            rows.append(col)
            names.append(SIGNAL_NAMES[c])
            if st["bwd"]:
                rows.append(np.asarray(col, float)[::-1])
                names.append(SIGNAL_NAMES[c].replace(" (", " [bwd] ("))
        data = np.vstack(rows).astype(np.float32)
        if st["autosave"] or str(basename or "").strip():
            from ..io.dat_writer import save_z_spectrum
            path = save_z_spectrum(w, str(basename or ""), z_rel=z_rel, i=i,
                                   df=df if any(c == 17 for c in st["chs"]) else None,
                                   amp=amp if any(c == 16 for c in st["chs"]) else None,
                                   z_abs=z_abs)
            w.dats_saved.append(path)
            if w.zspec_records:
                w.zspec_records[-1]["path"] = path
            cur["path"] = path
            w._event("dat_saved", path=path, experiment="Z spectroscopy",
                     idx=len(w.dats_saved), zspec_idx=cur["idx"])
        params = [0.0, -sweep, w.tip_x, w.tip_y, w.zctrl.z_n, st["z_offset"], settle, integ, 1.0,
                  w.zctrl.z_n]
        if not int(get_data):
            return [None, None, [], 0, 0, [], None, []]
        return [None, None, names, None, None, data.tolist(), None, params]
