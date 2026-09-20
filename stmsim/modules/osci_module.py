"""Oscilloscopes: Osci1T (one channel) and Osci2T (two channels), the current monitor's feed.

Contract: ``DataGet(0)`` returns immediately with the latest
screen; ``TimebaseGet`` → (index, count, table of whole-screen durations); sample period
``dt`` = screen / samples (2 kHz at the 0.128 s screen with 256 samples).
Screens are generated from the world's noise model at call time, stamped with a
monotonically increasing ``t0`` so the pump's duplicate detection sees fresh data.
"""
from __future__ import annotations

import numpy as np

from ..physics.world import World
from .signals import SignalTable

_SAMPLES_1T = 256


def _screen(w: World, sig: SignalTable, ch: int, n: int, dt: float):
    t0 = w.clock.wall()
    t = t0 + np.arange(n) * dt
    if ch in (0, 26):
        i_dc = w._dc_current() if not w._active_transient() else w.current_now()
        with w.lock:
            data = w.preamp.clamp(w.noise.current_series(i_dc, t))
    elif ch in (14, 10):
        z0 = w.z_now()
        data = z0 + w.noise.z_series(n)
    else:
        data = np.array([sig.value(ch) for _ in range(min(n, 8))] * (n // min(n, 8) + 1))[:n]
    return t0, data


def bind_osci(d, w: World):
    sig = SignalTable(w)
    table = w.rig.osci1t_timebase_s
    st1 = {"ch": 0, "tb": len(table) - 1, "trig": [0, 0, 0.0, 0.0]}
    st2 = {"chs": [0, 14], "tb": 0, "ovs": 0, "trig": [0, 0, 0, 0.0, 0.0, 0.0]}

    # ── Osci1T ──
    d.register("Osci1T_ChSet", lambda ch: st1.__setitem__("ch", int(ch)))
    d.register("Osci1T_ChGet", lambda: st1["ch"])
    d.register("Osci1T_TimebaseSet", lambda idx: st1.__setitem__("tb", int(idx)))
    d.register("Osci1T_TimebaseGet", lambda: [st1["tb"], None, list(table)])

    @d.handles("Osci1T_TrigSet")
    def _t1(mode, slope, level, hyst):
        st1["trig"] = [int(mode), int(slope), float(level), float(hyst)]

    d.register("Osci1T_TrigGet", lambda *a: None)
    d.register("Osci1T_Run", lambda: None)

    @d.handles("Osci1T_DataGet")
    def _data1(data_to_get):
        screen = table[st1["tb"]]
        dt = screen / _SAMPLES_1T
        t0, data = _screen(w, sig, st1["ch"], _SAMPLES_1T, dt)
        return [t0, dt, None, data.tolist()]

    # ── Osci2T ──
    @d.handles("Osci2T_ChsSet")
    def _chs(a, b):
        st2["chs"] = [int(a), int(b)]

    d.register("Osci2T_ChsGet", lambda: list(st2["chs"]))
    d.register("Osci2T_TimebaseSet", lambda idx: st2.__setitem__("tb", int(idx)))
    d.register("Osci2T_TimebaseGet", lambda: [st2["tb"], None, list(table)])
    d.register("Osci2T_OversamplSet", lambda idx: st2.__setitem__("ovs", int(idx)))
    d.register("Osci2T_OversamplGet", lambda: st2["ovs"])

    @d.handles("Osci2T_TrigSet")
    def _t2(mode, ch, slope, level, hyst, pos):
        st2["trig"] = [int(mode), int(ch), int(slope), float(level), float(hyst), float(pos)]

    d.register("Osci2T_TrigGet", lambda *a: None)
    d.register("Osci2T_Run", lambda: None)

    @d.handles("Osci2T_DataGet")
    def _data2(data_to_get):
        screen = table[st2["tb"]]
        dt = w.rig.osci_dt_s * (2 ** st2["ovs"])
        n = max(int(round(screen / dt)), 16)
        t0, a = _screen(w, sig, st2["chs"][0], n, dt)
        _, b = _screen(w, sig, st2["chs"][1], n, dt)
        return [t0, dt, None, a.tolist(), None, b.tolist()]
