"""Scan module: frame geometry, buffer, props, speed, start/stop, live grab, save, wait."""
from __future__ import annotations

import time

import numpy as np

from ..physics.world import World
from ..wire.errors import BadArguments
from .signals import SIGNAL_NAMES

_ACTION_START, _ACTION_STOP, _ACTION_PAUSE, _ACTION_RESUME = 0, 1, 2, 3


def bind_scan(d, w: World):
    st = w.scan

    @d.handles("Scan_Action")
    def _action(action, direction):
        a = int(action)
        if a == _ACTION_START:
            w.scan_start(direction_up=bool(int(direction)))
        elif a == _ACTION_STOP:
            w.scan_stop()
        elif a in (_ACTION_PAUSE, _ACTION_RESUME):
            pass

    d.register("Scan_StatusGet", lambda: 1 if w.scan_running() else 0)

    @d.handles("Scan_FrameSet")
    def _frame_set(cx, cy, width, height, angle):
        rx, ry = w.rig.xy_range_m
        if width <= 0 or height <= 0:
            raise BadArguments("Scan.FrameSet", "non-positive size")
        if abs(cx) + width / 2 > rx / 2 * 1.02 or abs(cy) + height / 2 > ry / 2 * 1.02:
            raise BadArguments("Scan.FrameSet", "frame exceeds piezo range")
        st.cx, st.cy, st.w, st.h, st.angle_deg = float(cx), float(cy), float(width), float(height), float(angle)
        w.move_xy(st.cx, st.cy)

    d.register("Scan_FrameGet", lambda: [st.cx, st.cy, st.w, st.h, st.angle_deg])

    @d.handles("Scan_BufferSet")
    def _buf_set(chs, pixels, lines):
        if chs:
            st.channels = [int(c) for c in chs]
        if int(pixels) > 0:
            st.nx = int(pixels)
        if int(lines) > 0:
            st.ny = int(lines)

    d.register("Scan_BufferGet", lambda: [None, list(st.channels), st.nx, st.ny])

    @d.handles("Scan_PropsSet")
    def _props_set(cont, bouncy, autosave, series, comment, modules, autopaste):
        # controller semantics: 0 = no change, 1 = on, 2 = off
        def tri(v, cur):
            v = int(v)
            return cur if v == 0 else (v == 1)
        st.continuous = tri(cont, st.continuous)
        st.bouncy = tri(bouncy, st.bouncy)
        st.autosave = tri(autosave, st.autosave)
        st.autopaste = tri(autopaste, st.autopaste)
        if series:
            st.series_name = str(series)
        if comment:
            st.comment = str(comment)
        if modules:
            st.modules_names = list(modules) if isinstance(modules, list) else [str(modules)]

    def _props_get():
        mods = st.modules_names or ["Bias", "Z-Controller", "Scan"]
        params = [["Bias (V)", "", ""], ["Setpoint", "P gain", "I gain"], ["Scanfield", "pixels/line", "lines"]]
        while len(params) < len(mods):
            params.append(["", "", ""])
        params = params[:len(mods)]
        counts = [len([p for p in row if p]) for row in params]
        return [1 if st.continuous else 0, 1 if st.bouncy else 0, 1 if st.autosave else 0,
                None, st.series_name, None, st.comment, None, None, mods, None, counts,
                None, None, params, 1 if st.autopaste else 0]

    d.register("Scan_PropsGet", _props_get)

    @d.handles("Scan_SpeedSet")
    def _speed_set(fwd_speed, bwd_speed, fwd_time, bwd_time, keep_const, ratio):
        keep = int(keep_const)
        if keep == 1 or (fwd_time > 0 and fwd_speed <= 0):
            st.line_time_fwd_s = float(fwd_time) if fwd_time > 0 else st.line_time_fwd_s
            st.line_time_bwd_s = float(bwd_time) if bwd_time > 0 else st.line_time_fwd_s
        else:
            if fwd_speed > 0:
                st.line_time_fwd_s = st.w / float(fwd_speed)
            if bwd_speed > 0:
                st.line_time_bwd_s = st.w / float(bwd_speed)
            elif fwd_speed > 0:
                st.line_time_bwd_s = st.line_time_fwd_s
        st.speed_fwd_m_per_s = st.w / st.line_time_fwd_s
        st.speed_bwd_m_per_s = st.w / st.line_time_bwd_s

    d.register("Scan_SpeedGet", lambda: [st.w / st.line_time_fwd_s, st.w / st.line_time_bwd_s,
                                         st.line_time_fwd_s, st.line_time_bwd_s, 1,
                                         st.line_time_bwd_s / max(st.line_time_fwd_s, 1e-9)])

    @d.handles("Scan_FrameDataGrab")
    def _grab(channel_index, data_direction):
        fr = w.frame
        ch = int(channel_index)
        name = SIGNAL_NAMES[ch] if 0 <= ch < len(SIGNAL_NAMES) else f"Sig{ch}"
        if fr is None:
            data = np.zeros((st.ny, st.nx), dtype=np.float32)
            return [None, name, None, None, data.tolist(), 0 if st.scan_dir == "down" else 1]
        fr.progress(w.clock.sim())
        if ch not in fr.data:
            raise BadArguments("Scan.FrameDataGrab", f"channel {ch} not in scan buffer")
        buf = fr.live_buffer(ch, "fwd" if int(data_direction) == 1 else "bwd")
        return [None, name, None, None, buf.astype(np.float32).tolist(),
                0 if fr.settings.scan_dir == "down" else 1]

    @d.handles("Scan_WaitEndOfScan")
    def _wait(timeout_ms):
        budget = min(max(float(timeout_ms), 0.0) / 1000.0, 4.0) if int(timeout_ms) >= 0 else 4.0
        t0 = time.monotonic()
        while w.scan_running():
            if time.monotonic() - t0 >= budget:
                return [1, None, ""]
            time.sleep(0.02)
        path = w.frame.saved_path if (w.frame is not None and w.frame.saved_path) else ""
        return [0, None, path]

    @d.handles("Scan_Save")
    def _save(wait, timeout_ms):
        w.save_frame()
        return 0

    d.register("Scan_XYPosGet", lambda wait: [w.tip_x, w.tip_y])
    d.register("Scan_BackgroundPaste", lambda wait, timeout: 0)
