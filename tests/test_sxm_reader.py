"""stmsim's own reader for the files it writes: round trips through the instrument, and
files that are not frames are refused."""
from __future__ import annotations

import numpy as np
import pytest

from stmbench.instrument import Instrument
from stmsim.io import sxm_reader as R


def _session(tmp_path):
    inst = Instrument("P5_sader_jarvis_qplus", 1, out_dir=tmp_path / "run")
    a = inst.act("scan", center_x_nm=3.0, center_y_nm=-2.0, size_nm=15.0, pixels=40, angle_deg=20.0)
    b = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=10.0, pixels=32, direction="up")
    inst.act("pll", on=True, amplitude_pm=50.0)
    z = inst.act("zspec", sweep_nm=0.4, points=64)
    s = inst.act("sts", v_start=-0.3, v_end=0.3, points=50)
    return inst, tmp_path / "run" / "session", [a["frame"], b["frame"]], [z["spectrum"], s["spectrum"]]


def test_frames_read_back_in_image_orientation(tmp_path):
    inst, sess, frames, _ = _session(tmp_path)
    for name in frames:
        scan = R.read_sxm(sess / name)
        rec = inst.frames[name]
        h = scan["header"]
        assert h["scan_pixels"] == [rec.geometry["pixels"]] * 2
        assert h["channel_names"][:2] and set(h["channel_directions"]) == {"both"}
        assert h["scan_dir"] == rec.geometry["direction"]
        for ch, fwd, bwd in (("Z", rec.z_fwd, rec.z_bwd), ("Current", rec.i_fwd, rec.i_bwd)):
            fr = R.oriented_frames(scan, ch)
            # the file holds float32: equal to the instrument's arrays to that precision
            assert np.allclose(fr["forward"], fwd, rtol=1e-6, atol=1e-12 * (ch == "Z") + 1e-18)
            assert np.allclose(fr["backward"], bwd, rtol=1e-6, atol=1e-12 * (ch == "Z") + 1e-18)
        assert R.oriented_frames(scan, "z")["forward"] is not None           # case-insensitive
        assert R.read_sxm_header(sess / name) == h


def test_spectra_read_back(tmp_path):
    inst, sess, _, spectra = _session(tmp_path)
    z = R.read_dat(sess / spectra[0])
    assert z["header"] and len(next(iter(z["columns"].values()))) == 64
    s = R.read_dat(sess / spectra[1])
    v = next(iter(s["columns"].values()))
    assert np.allclose(v, inst.spectra[spectra[1]]["v"], atol=1e-6)


def test_files_that_are_not_frames_are_refused(tmp_path):
    p = tmp_path / "x.sxm"
    p.write_bytes(b":SCAN_PIXELS:\n 10 10\n")
    with pytest.raises(ValueError):
        R.read_sxm(p)
    assert R.read_sxm_header(p) == {}
    assert R.read_dat(tmp_path / "missing.dat" if False else p)["columns"] == {}
