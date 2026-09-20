"""controller ``.dat`` writer for spectroscopy sweeps.

Every analysis skill that reads a spectrum — the compatibility client's ``AssessShockleyOnset``,
``AssessSpectrum``, the dispersion fit, the Sader–Jarvis inversion — takes a **file path**,
because on a real rig that is what the controller autosave leaves behind. The simulator has to
produce the same thing or the whole analysis half of a paper task is unreachable.

Layout (matched against controller files, ``stmsim/calibrate/iz_templates.parse_dat_text``
and MAST's ``mast.io.nanonis_files.read_dat``)::

    <key>\\t<value>\\t
    <key>\\t<value>\\t
    <blank>
    [DATA]
    <col>\\t<col>\\t<col>
    1.234560E-01\\t...

Two conventions the readers rely on: the **column-name row comes after** ``[DATA]``, and the
first data column is the sweep axis. Files are ASCII so both readers' encodings agree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_MARKER = "[DATA]"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) or isinstance(v, np.floating):
        return f"{float(v):.6E}"
    return str(v)


def write_dat(path: str | Path, *, header: dict, names: list[str],
              columns: list[np.ndarray]) -> Path:
    """Write one sweep. ``columns[0]`` must be the sweep axis."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}\t{_fmt(v)}\t" for k, v in header.items()]
    lines.append("")
    lines.append(DATA_MARKER)
    lines.append("\t".join(names))
    data = np.column_stack([np.asarray(c, float) for c in columns])
    for row in data:
        lines.append("\t".join("%.6E" % v for v in row))
    p.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    return p


def common_header(world, *, experiment: str) -> dict:
    """The header keys both readers actually branch on, plus the ones the corpus carries."""
    rec = world.clock.sim_datetime(world.clock.sim())
    stamp = f"{rec.day:02d}.{rec.month:02d}.{rec.year:04d} {rec:%H:%M:%S}"
    sx, sy = world.sample_xy()
    return {
        "Experiment": experiment,
        "Date": stamp,
        "Saved Date": stamp,
        "User": "stmsim",
        "X (m)": world.tip_x,
        "Y (m)": world.tip_y,
        "Z (m)": world.z_tip_from_ctrl(world.zctrl.z_n),
        "Bias>Bias (V)": world.bias_v,
        "Bias>Calibration (V/V)": 1.0,
        "Z-Controller>Setpoint": world.zctrl.setpoint_a,
        "Z-Controller>Controller status": "ON" if world.zctrl.on else "OFF",
        "Z-Controller>Z (m)": float(world.zctrl.z_n),
        "Current>Calibration (A/V)": 1.0,
        "Current>Gain": world.preamp.gain_index,
        "Temperature 1>Temperature 1 (K)": world.temperature_k,
        "Comment01": "",
    }


def oscillation_header(world) -> dict:
    """``Oscillation Control>…`` keys — where the inversion skill finds f0, A and Q."""
    pll = getattr(world, "pll", None)
    if pll is None:
        return {}
    return {
        "Oscillation Control>Center Frequency (Hz)": pll.center_freq_hz,
        "Oscillation Control>Amplitude Setpoint (m)": pll.amp_setpoint_m,
        "Oscillation Control>Q-Factor": pll.q,
        "Oscillation Control>Excitation (V)": pll.excitation_v,
        "Oscillation Control>Output on": bool(pll.output_on),
        "Oscillation Control>Amplitude Controller on": bool(pll.amp_ctrl_on),
        "Oscillation Control>Phase Controller on": bool(pll.phas_ctrl_on),
        "Oscillation Control>Input Calibration (m/V)": pll.input_calibration_m_per_v,
        "Oscillation Control>Reference Phase (deg)": pll.demod_phase_ref_deg,
        "Oscillation Control>Lock-In Filter Order": pll.demod_filter_order,
        "Oscillation Control>Harmonic": pll.demod_harmonic,
    }


def next_path(world, basename: str, default: str) -> Path:
    """``<basename><NNNNN>.dat`` in the session directory, one counter per basename."""
    base = (basename or default).strip() or default
    n = world._dat_counters.get(base, 0) + 1
    world._dat_counters[base] = n
    return Path(world.session_dir) / f"{base}{n:05d}.dat"


def save_bias_spectrum(world, basename: str, *, v, i, didv=None, i_bwd=None) -> str:
    header = common_header(world, experiment="bias spectroscopy")
    v = np.asarray(v, float)
    header.update({
        "Bias Spectroscopy>Sweep Start (V)": float(v[0]),
        "Bias Spectroscopy>Sweep End (V)": float(v[-1]),
        "Bias Spectroscopy>Num Pixel": int(v.size),
        "Bias Spectroscopy>Z offset (m)": 0.0,
        "Bias Spectroscopy>Z-controller hold": True,
        "Lock-in>Lock-in status": "ON" if world.lockin_on else "OFF",
        "Lock-in>Amplitude": world.lockin_amp_v,
        "Lock-in>Frequency (Hz)": world.lockin_freq_hz,
        "Lock-in>Modulated signal": "Bias (V)",
    })
    names = ["Bias calc (V)", "Current (A)"]
    cols = [v, np.asarray(i, float)]
    if i_bwd is not None:
        names.append("Current [bwd] (A)")
        cols.append(np.asarray(i_bwd, float))
    if didv is not None:
        names.append("LI Demod 1 X (A)")
        cols.append(np.asarray(didv, float) * (world.lockin_amp_v if world.lockin_on else 1e-3))
    path = next_path(world, basename, "Bias Spectroscopy")
    write_dat(path, header=header, names=names, columns=cols)
    return str(path)


def save_z_spectrum(world, basename: str, *, z_rel, i, df=None, amp=None, z_abs=None) -> str:
    header = common_header(world, experiment="Z spectroscopy")
    header.update(oscillation_header(world))
    z_rel = np.asarray(z_rel, float)
    header.update({
        "Z Spectroscopy>Sweep Start (m)": float(z_rel[0]),
        "Z Spectroscopy>Sweep End (m)": float(z_rel[-1]),
        "Z Spectroscopy>Num Pixel": int(z_rel.size),
    })
    names = ["Z rel (m)", "Current (A)"]
    cols = [z_rel, np.asarray(i, float)]
    if df is not None:
        names.append("OC M1 Freq. Shift (Hz)")
        cols.append(np.asarray(df, float))
    if amp is not None:
        names.append("OC D1 Amplitude (m)")
        cols.append(np.asarray(amp, float))
    if z_abs is not None:
        names.append("Z (m)")
        cols.append(np.asarray(z_abs, float))
    path = next_path(world, basename, "Z-Spectroscopy")
    write_dat(path, header=header, names=names, columns=cols)
    return str(path)
