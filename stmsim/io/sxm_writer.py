"""Write controller ``.sxm`` files the way ``mast.io.nanonis_files.read_sxm`` expects them.

Layout: text header of ``:KEY:`` lines, ``:SCANIT_END:``, then ``\\x1a\\x04``, then
big-endian float32 frames channel by channel (forward then backward for ``both``).

Conventions (from the reader and its tests):

* rows are stored in **acquisition order** — ``:SCAN_DIR: up`` files therefore have the
  bottom edge in row 0 (``sxm_oriented_frames`` flips them);
* the **backward block is stored mirrored** along x (the reader does ``fliplr``);
* rows not acquired are **NaN** (``frame_validity.acquired_row_mask``);
* ``:DATA_INFO:`` rows are tab-separated, header row first; the reader reads columns
  Channel / Name / Unit / Direction and ignores Unit;
* ``:COMMENT:`` is written GBK (the controller file convention);
* ``:REC_DATE:`` / ``:REC_TIME:`` are the frame **start** time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..modules.signals import sxm_channel_name


def write_sxm(path: str | Path, frame, world, *, channels: list[int] | None = None) -> Path:
    st = frame.settings
    chans = channels or list(st.channels)
    rec = world.clock.sim_datetime(frame.rec_sim_s)
    ny, nx = st.ny, st.nx
    lines = [
        ":NANONIS_VERSION:", "2",
        ":SCANIT_TYPE:", "              FLOAT            MSBFIRST",
        ":REC_DATE:", f" {rec.day:02d}.{rec.month:02d}.{rec.year:04d}",
        ":REC_TIME:", rec.strftime("%H:%M:%S"),
        ":REC_TEMP:", f"            {world.temperature_k:.1f}",
        ":ACQ_TIME:", f"        {st.frame_time_s:.1f}",
        ":SCAN_PIXELS:", f"{nx:>10d}{ny:>10d}",
        ":SCAN_FILE:", str(path),
        ":SCAN_TIME:", f"{st.line_time_fwd_s:>19.6E}{st.line_time_bwd_s:>19.6E}",
        ":SCAN_RANGE:", f"{st.w:>19.6E}{st.h:>19.6E}",
        ":SCAN_OFFSET:", f"{st.cx:>19.6E}{st.cy:>19.6E}",
        ":SCAN_ANGLE:", f"{st.angle_deg:>19.6E}",
        ":SCAN_DIR:", st.scan_dir,
        ":BIAS:", f"{world.bias_v:.6E}",
        ":Z-CONTROLLER:",
        "\tName\ton\tSetpoint\tP-gain\tI-gain\tT-const",
        f"\t{world.zctrl.controllers[world.zctrl.active_index]}\t{1 if world.zctrl.on else 0}\t"
        f"{world.zctrl.setpoint_a:.4E} A\t{world.zctrl.p_m:.4E} m\t{world.zctrl.i_m_per_s:.4E} m/s\t{world.zctrl.time_const_s:.4E} s",
        ":Z-CONTROLLER>SETPOINT:", f"{world.zctrl.setpoint_a:.6E}",
        ":COMMENT:", st.comment or "",
        ":Bias>Bias (V):", f"{world.bias_v:.6E}",
        ":Z-Controller>Z (m):", f"{world.zctrl.z_n:.6E}",
        ":Z-Controller>Setpoint:", f"{world.zctrl.setpoint_a:.6E} A",
        ":Z-Controller>Controller status:", "ON" if world.zctrl.on else "OFF",
        ":Z-Controller>P gain:", f"{world.zctrl.p_m:.4E} m",
        ":Z-Controller>I gain:", f"{world.zctrl.i_m_per_s:.4E} m/s",
        ":Scan>Scanfield:", f"{st.cx:.6E};{st.cy:.6E};{st.w:.6E};{st.h:.6E};{st.angle_deg:.6E}",
        ":Scan>series name:", st.series_name,
        ":Scan>channels:", ";".join("%s (%s)" % sxm_channel_name(c) for c in chans),
        ":Scan>pixels/line:", str(nx),
        ":Scan>lines:", str(ny),
        ":Scan>speed forw. (m/s):", f"{st.w / max(st.line_time_fwd_s, 1e-9):.6E}",
        ":Scan>speed backw. (m/s):", f"{st.w / max(st.line_time_bwd_s, 1e-9):.6E}",
        ":DATA_INFO:",
        "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset",
    ]
    for c in chans:
        name, unit = sxm_channel_name(c)
        lines.append(f"\t{c}\t{name}\t{unit}\tboth\t1.000E+0\t0.000E+0")
    lines += [":SCANIT_END:", "", ""]
    header_text = "\n".join(lines)
    # COMMENT is GBK in the controller file convention; everything else is ASCII
    header = header_text.encode("gbk", errors="replace")
    blobs = []
    for c in chans:
        fwd, bwd = frame.saved_arrays(c)
        # the renderer already keeps the backward pass in ACQUISITION order (right→left), which is
        # exactly the mirrored layout controller writes; the reader un-mirrors it with fliplr
        blobs.append(np.ascontiguousarray(fwd, dtype=">f4").tobytes())
        blobs.append(np.ascontiguousarray(bwd, dtype=">f4").tobytes())
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(header + b"\x1a\x04" + b"".join(blobs))
    return p
