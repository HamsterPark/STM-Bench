"""Ad-hoc fidelity probes: run MAST's own trace/retrace similarity on simulator frames, and
watch what a series of pulses does to the Z position."""
from __future__ import annotations

import sys
import time

import numpy as np

from stmbench.paths import data_path
from stmsim.physics.rig import RigProfile
from stmsim.physics.tip import Apex, Tip
from stmsim.physics.world import World


def frame_similarity(world: World, w_nm: float, px: int, line_s: float):
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.tip_metrics import trace_retrace_correlation, assess_tip_classical
    world.scan.nx = world.scan.ny = px
    world.scan.w = world.scan.h = w_nm * 1e-9
    world.scan.line_time_fwd_s = world.scan.line_time_bwd_s = line_s
    world.scan_start()
    while world.scan_running():
        time.sleep(0.02)
    path = world.save_frame()
    fr = sxm_oriented_frames(read_sxm(path), "Z")
    f, b = fr["forward"], fr["backward"]
    corr = trace_retrace_correlation(f, b)
    raw = np.corrcoef(f.ravel(), b.ravel())[0, 1]
    m = assess_tip_classical(f, b, nm_per_px=w_nm / px)
    return corr, raw, m


def main():
    w = World(rig=RigProfile.load("reference-stm"), seed=5, session_dir=data_path("tmp_dbg"), time_scale=200.0)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.set_bias(1.0)
    w.set_setpoint(100e-12)
    time.sleep(0.8)
    print("good tip:")
    for (wn, px, ls) in [(100, 128, 0.586), (50, 128, 0.293), (20, 128, 0.15)]:
        c, raw, m = frame_similarity(w, wn, px, ls)
        print(f"  {wn} nm/{px}px: trace_retrace_corr={c:.3f} raw_corr={raw:.3f} instab={m.fwd_bwd_instability} "
              f"resolution_nm={m.resolution_nm} fft_sharp={m.fft_sharpness}")
    w.tip.apexes = [Apex(0, 0, 0, 1.0), Apex(2.5e-9, -1.5e-9, -0.2354e-9, 0.8)]
    w.tip.radius_m = 8e-9
    c, raw, m = frame_similarity(w, 100, 128, 0.586)
    print(f"blunt+double tip 100 nm: corr={c:.3f} raw={raw:.3f} instab={m.fwd_bwd_instability}")
    print("pulses:")
    for k in range(4):
        w.bias_pulse(0.5, 10.0, True, 1)
        time.sleep(3.2)
        print(f"  pulse {k}: outcome={w.events[-1]['outcome']} dz_nm={w.events[-1].get('dz_m',0)*1e9:.1f} "
              f"length_nm={w.tip.length_m*1e9:.1f} h_nm={w.surface_height_here()*1e9:.2f} z_n_nm={w.z_now()*1e9:.1f} "
              f"I_pA={w.current_now()*1e12:.1f}")


if __name__ == "__main__":
    sys.exit(main())
