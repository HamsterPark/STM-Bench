"""Which factor lowers MAST's trace/retrace correlation on 100 nm sim frames?"""
from __future__ import annotations

import time

import numpy as np

from stmbench.paths import data_path
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World


def render(w, hyst_frac, fb_on, noise_scale):
    from mast.vision.tip_metrics import trace_retrace_correlation
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    w.hyst_frac = hyst_frac
    w.noise.z_floor_m = 2e-12 * noise_scale
    w.zctrl.on = fb_on
    w.scan.nx = w.scan.ny = 128
    w.scan.w = w.scan.h = 100e-9
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.586
    w.scan_start()
    while w.scan_running():
        time.sleep(0.01)
    path = w.save_frame()
    fr = sxm_oriented_frames(read_sxm(path), "Z")
    f, b = fr["forward"], fr["backward"]
    return trace_retrace_correlation(f, b), f, b


w = World(rig=RigProfile.load("reference-stm"), seed=5, session_dir=data_path("tmp_dbg2"), time_scale=500.0)
w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
w.withdrawn = False
w.zctrl_set(True)
w.set_bias(1.0); w.set_setpoint(100e-12); time.sleep(0.8)
for label, hy, fb, ns in [("baseline", 0.025, True, 1.0), ("no hysteresis", 0.0, True, 1.0),
                          ("no noise", 0.025, True, 0.0), ("hyst 6px", 0.05, True, 1.0),
                          ("no hyst no noise", 0.0, True, 0.0)]:
    c, f, b = render(w, hy, fb, ns)
    print(f"{label:20s} corr={c:.3f}  rowshift_est={np.argmax(np.correlate(f[64]-f[64].mean(), b[64]-b[64].mean(), 'full'))-127}")
w.zctrl.on = True
# steps vs flat terrace: scan inside one terrace
w.scan.cx, w.scan.cy = 0.0, 0.0
from stmsim.physics.surface import Surface
w2 = World(rig=RigProfile.load("reference-stm"), seed=5, session_dir=data_path("tmp_dbg2"), time_scale=500.0)
w2.surface = Surface("Au(111)", seed=5, adsorbate_density_per_um2=0.0)
w2.surface.site.terrace_w = 5e-6   # effectively one terrace
w2.coarse.coarse_gap_m = w2.surface_height_here() + 0.6e-9
w2.withdrawn = False
w2.zctrl_set(True); w2.set_bias(1.0); w2.set_setpoint(100e-12); time.sleep(0.8)
c, f, b = render(w2, 0.025, True, 1.0)
print(f"{'single terrace':20s} corr={c:.3f}")
