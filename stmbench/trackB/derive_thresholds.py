"""Derive the truth thresholds that must not be arbitrary (DESIGN.md §5.2).

* ``r_star_nm`` — the largest tip radius at which the simulator's Au(111) frames still
  pass MAST's atomic-phase gate (``mast.vision.atomic_phase.assess_atomic_phase``) in at
  least ``pass_frac`` of seeds, at the imaging point MAST itself prescribes for Au(111)
  (20 mV / 500 pA, 5 nm / 256 px ⇒ 0.0195 nm/px, inside the ``< 0.02`` scale gate).
  The gate is MAST's and the tip is the simulator's; r* indicates the largest radius
  supporting the requested atomic-resolution condition.

* ``lambda_star_per_s`` — an UPPER bound on the acceptable spontaneous tip-change rate,
  read off real complete frames: per frame ``rate = n_row_jumps / frame_time`` where
  ``n_row_jumps = rowjump_frac × rows``; λ* is the ``lambda_quantile`` of that rate over
  clean, complete frames (row jumps also come from feedback and noise, so this is an
  upper bound, and it is reported as one).

Outputs ``$STM_BENCH_DATA/calib/thresholds.json`` (``stmbench.paths.calib_dir()``, or
``--out``) with the full sweep so a reader can see where the cliff is, not just the
number. Run:

    python -m stmbench.trackB.derive_thresholds --seeds 8 --out "$STM_BENCH_DATA/calib/thresholds.json"
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from stmbench.paths import calib_dir, corpus_index_dir, data_path

SIGMAS_NM = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.20, 0.25)


def radius_implied_by_sigma(sigma_nm: float) -> float | None:
    """Invert the mean of the ASSUMED σ_a(R) map in ``Tip.apex_sigma_from_radius``."""
    r = (sigma_nm - 0.06) / 0.04
    return r if r > 0 else None


def _render(world, w_nm: float, px: int, line_s: float, centre=(0.0, 0.0)):
    world.scan.nx = world.scan.ny = px
    world.scan.w = world.scan.h = w_nm * 1e-9
    world.scan.cx, world.scan.cy = float(centre[0]), float(centre[1])
    world.scan.line_time_fwd_s = world.scan.line_time_bwd_s = line_s
    world.scan_start()
    while world.scan_running():
        time.sleep(0.01)
    return world.save_frame()


def _flat_centre(world, w_nm: float, *, max_step_pm: float = 20.0, reach_nm: float = 400.0):
    """A scan centre whose window holds no step — the target of FindFlatRegion
    before an atomic-resolution frame. Uses the simulator's own terrace truth (this is a
    threshold derivation, not an episode). Steps inside a 5 nm atomic frame put extra
    peaks into the FFT and MAST rightly calls them ``peaks_not_one_lattice``."""
    import itertools
    half = 0.5 * w_nm * 1e-9
    g = np.linspace(-half, half, 9)
    gx, gy = np.meshgrid(g, g)
    cands = [(0.0, 0.0)]
    for r in range(1, int(reach_nm // 10) + 1):
        for dx, dy in itertools.product((-r, 0, r), repeat=2):
            if (dx, dy) != (0, 0):
                cands.append((dx * 10e-9, dy * 10e-9))
    for cx, cy in cands:
        h = world.surface.terrace_height(gx + cx, gy + cy)
        # remove the terrace tilt (a plane) before asking for step-free
        A = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
        coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
        resid = h.ravel() - A @ coef
        if (resid.max() - resid.min()) * 1e12 < max_step_pm:
            return cx, cy
    return 0.0, 0.0


def sweep_sigma_star(seeds: int, session_dir: Path, *, w_nm: float = 5.0, px: int = 256,
                     line_s: float = 0.1, sigmas=SIGMAS_NM, pass_frac: float = 0.9,
                     material: str = "Au(111)", bias_v: float = 0.02, setpoint_a: float = 500e-12,
                     radius_nm: float = 1.0, verbose: bool = True) -> dict:
    """Sweep the apex smearing σ_a against MAST's atomic-phase gate; σ* = the largest σ_a
    that still passes in ≥ ``pass_frac`` of seeds. The mesoscopic radius is held fixed."""
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.atomic_phase import assess_atomic_phase
    from mast.vision.lattice_calibration import first_order_period_nm

    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World

    expect = first_order_period_nm(material)
    nmpp = w_nm / px
    rows = []
    for s_nm in sigmas:
        passed = 0
        details = []
        for seed in range(seeds):
            w = World(rig=RigProfile.load("reference-stm"), seed=100 + seed, material=material,
                      session_dir=session_dir / f"s{s_nm}_{seed}", time_scale=1000.0)
            w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
            w.withdrawn = False
            w.zctrl_set(True)
            w.transients.clear()
            w.set_bias(bias_v)
            w.set_setpoint(setpoint_a)
            w.tip.radius_m = radius_nm * 1e-9
            w.tip.apex_sigma_m = s_nm * 1e-9
            w.tip.apex_radius_ref_m = w.tip.radius_m   # pin: no resampling during the sweep
            w.tip.lambda_per_s = 0.0                   # this sweep isolates sharpness
            w.tip.metastable = False
            w.achievable_z_tip()
            path = _render(w, w_nm, px, line_s, centre=_flat_centre(w, w_nm))
            fr = sxm_oriented_frames(read_sxm(path), "Z")
            res_f = assess_atomic_phase(fr["forward"], nm_per_px=nmpp, expected_a_nm=expect)
            res_b = assess_atomic_phase(fr["backward"], nm_per_px=nmpp, expected_a_nm=expect)
            ok = bool(res_f.passed and res_b.passed)
            passed += ok
            details.append({"seed": seed, "passed": ok,
                            "fwd": {"passed": bool(res_f.passed),
                                    "reasons": list(getattr(res_f, "reasons", ()) or ()),
                                    "concentration": getattr(res_f, "angular_concentration", None),
                                    "snr": getattr(res_f, "snr", None),
                                    "sharpness": getattr(res_f, "fft_sharpness", None),
                                    "period_nm": getattr(res_f, "period_nm", None),
                                    "period_fast_axis_nm": getattr(res_f, "period_fast_axis_nm", None)},
                            "bwd_passed": bool(res_b.passed)})
        frac = passed / seeds
        transfer = w.tip.atomic_transfer(w.surface.material.first_order_period_m)
        rows.append({"apex_sigma_nm": s_nm, "pass_frac": frac, "n": seeds,
                     "transfer": transfer,
                     "imaged_corrugation_pm": transfer * w.surface.material.corrugation_m * 1e12,
                     "details": details})
        if verbose:
            reasons = [d["fwd"]["reasons"][:1] for d in details if not d["passed"]]
            print(f"sigma_a={s_nm:.2f} nm  transfer {transfer:.3f}  pass {passed}/{seeds}  {reasons[:3]}",
                  flush=True)
    ok_s = [row["apex_sigma_nm"] for row in rows if row["pass_frac"] >= pass_frac]
    sigma_star = max(ok_s) if ok_s else None
    return {"sigma_star_nm": sigma_star,
            "r_star_nm_implied": radius_implied_by_sigma(sigma_star) if sigma_star else None,
            "pass_frac_required": pass_frac, "imaging": {
                "material": material, "bias_v": bias_v, "setpoint_a": setpoint_a, "size_nm": w_nm,
                "pixels": px, "nm_per_px": nmpp, "line_s": line_s, "expected_a_nm": expect,
                "radius_nm_held": radius_nm},
            "sweep": rows}


def lambda_star_from_index(index_path: str, *, quantile: float = 0.85,
                           material_filter: str = "Au") -> dict:
    """Upper bound on the acceptable tip-change rate from complete reference frames."""
    import pandas as pd

    df = pd.read_parquet(index_path)
    cols = set(df.columns)
    need = {"rowjump_frac", "ny", "t_fwd_s", "t_bwd_s", "acq_frac", "fb_corr_flip", "path"}
    if not need <= cols:
        return {"lambda_star_per_s": None, "error": f"index lacks {sorted(need - cols)}",
                "columns": sorted(cols)}
    d = df[df["err"].isna() | (df["err"].astype(str) == "")] if "err" in cols else df
    if material_filter:
        hay = d["path"].astype(str)
        if "comment" in cols:
            hay = hay + " " + d["comment"].astype(str)
        d = d[hay.str.contains(material_filter, case=False, na=False)]
    d = d[d["acq_frac"] >= 0.99]              # complete frames only
    d = d[d["fb_corr_flip"] >= 0.9]           # MAST's good-tip trace/retrace band
    rows_n = d["ny"].astype(float)
    t_frame = rows_n * (d["t_fwd_s"].astype(float) + d["t_bwd_s"].astype(float))
    keep = (t_frame > 0) & np.isfinite(t_frame)
    d, t_frame, rows_n = d[keep], t_frame[keep], rows_n[keep]
    n_jumps = d["rowjump_frac"].astype(float) * rows_n
    rate = (n_jumps / t_frame).replace([np.inf, -np.inf], np.nan).dropna()
    if rate.empty:
        return {"lambda_star_per_s": None, "error": "no frames survived the filters"}
    q = float(np.quantile(rate.values, quantile))
    return {"lambda_star_per_s": q, "quantile": quantile, "n_frames": int(rate.size),
            "rate_median_per_s": float(np.median(rate.values)),
            "rate_p50_p85_p95": [float(np.quantile(rate.values, p)) for p in (0.5, 0.85, 0.95)],
            "frac_frames_with_any_jump": float((n_jumps > 0).mean()),
            "t_frame_median_s": float(np.median(t_frame.values)),
            "filters": {"material": material_filter, "acq_frac": ">=0.99", "fb_corr_flip": ">=0.9"},
            "note": ("upper bound: the row-jump detector fires on feedback/noise too "
                     "(clean frames still show a median rowjump_frac of a few percent)")}


def lambda_star_from_task(*, verify_rows: int = 256, verify_line_s: float = 0.586,
                          max_interrupt_p: float = 0.10) -> dict:
    """λ* defined by what the task needs: MAST's verify frame (100 nm / 256 px / 0.586 s per
    line, both directions ⇒ ≈300 s) must survive without a tip change in ≥ 90 % of frames.
    P(no change in T) = exp(−λT) ≥ 1 − p  ⇒  λ* = −ln(1 − p) / T."""
    t_frame = verify_rows * 2.0 * verify_line_s
    lam = -math.log(1.0 - max_interrupt_p) / t_frame
    return {"lambda_star_per_s": lam, "t_verify_frame_s": t_frame,
            "max_interrupt_p": max_interrupt_p,
            "note": "task-derived: a Poisson tip-change rate at which MAST's own verify frame "
                    "is interrupted in at most max_interrupt_p of attempts"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=str(calib_dir() / "thresholds.json"),
                    help="default $STM_BENCH_DATA/calib/thresholds.json")
    ap.add_argument("--session-dir", default=str(data_path("tmp_thresholds")),
                    help="where the sweep's sim .sxm go (default $STM_BENCH_DATA/tmp_thresholds)")
    ap.add_argument("--index", default=str(corpus_index_dir() / "sxm_index_full.parquet"),
                    help="reference frame index parquet (default <corpus index>/sxm_index_full.parquet)")
    ap.add_argument("--skip-lambda", action="store_true")
    ap.add_argument("--skip-rstar", action="store_true")
    args = ap.parse_args(argv)
    out: dict = {}
    if not args.skip_rstar:
        out["sigma_star"] = sweep_sigma_star(args.seeds, Path(args.session_dir))
        print("sigma* =", out["sigma_star"]["sigma_star_nm"], "nm;  implied r* =",
              out["sigma_star"]["r_star_nm_implied"], "nm")
    if not args.skip_lambda:
        out["lambda_star_task"] = lambda_star_from_task()
        try:
            out["lambda_star_index"] = lambda_star_from_index(args.index)
        except Exception as exc:  # noqa: BLE001
            out["lambda_star_index"] = {"lambda_star_per_s": None, "error": repr(exc)}
        print("lambda* (task) =", out["lambda_star_task"]["lambda_star_per_s"])
        print("lambda* (index upper bound) =", out["lambda_star_index"].get("lambda_star_per_s"),
              out["lambda_star_index"].get("error", ""), out["lambda_star_index"].get("n_frames"))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
