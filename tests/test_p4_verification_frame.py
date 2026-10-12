"""P4's verification frame (≤ 0.2 nm/px) is fine enough to place the moved atom: the rule asks
for the measurement the claim needs, not for more."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from stmbench.trackB import claims as C
from stmsim.physics.scanner import SIG_Z
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


def _placement_errors(nm_per_px: float, tmp_path, n_seeds: int = 2, n_atoms: int = 3) -> list[float]:
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    seeds = [s for s in range(80) if sc.draw_hidden(s)["tip"]["condition"] == "good"][:n_seeds]
    out = []
    for seed in seeds:
        w = sc.build_world(seed, session_dir=tmp_path / f"s{seed}")
        w.clock.pause()
        reg = w.surface.site.adatoms
        atoms = sorted(reg.atoms.values(), key=lambda a: math.hypot(*reg.site_xy(a.i, a.j, a.sub)))[:n_atoms]
        for atom in atoms:
            ax, ay = (v * 1e9 for v in reg.site_xy(atom.i, atom.j, atom.sub))
            n = int(round(8.0 / nm_per_px))
            st = w.scan
            st.cx, st.cy, st.w, st.h, st.nx, st.ny = (ax + 0.37) * 1e-9, (ay - 0.21) * 1e-9, 8e-9, 8e-9, n, n
            st.line_time_fwd_s = st.line_time_bwd_s = 0.3
            w.scan_start()
            fr = w.frame
            fr.rows_done, fr.finished = fr.rows_total, True
            w.save_frame()
            ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
            # look within 1 nm of where the atom should appear, then take the bump's centroid
            ex, ey = C.to_scan_frame(ev, ax, ay)
            ie = (ex - st.cx * 1e9 + 4.0) / 8.0 * n - 0.5
            je = (4.0 - (ey - st.cy * 1e9)) / 8.0 * n - 0.5
            z = fr.data[SIG_Z][0]
            z = z - np.median(z, axis=1, keepdims=True)
            rr = int(round(1.0 / nm_per_px))
            ka, ia = max(int(je) - rr, 0), max(int(ie) - rr, 0)
            sub = z[ka:int(je) + rr + 1, ia:int(ie) + rr + 1]
            k, i = np.unravel_index(int(np.argmax(sub)), sub.shape)
            k, i = k + ka, i + ia
            r = max(2, int(round(0.6 / nm_per_px)))
            k0, i0 = max(k - r, 0), max(i - r, 0)
            win = z[k0:k + r + 1, i0:i + r + 1]
            win = np.clip(win - 0.5 * win.max(), 0, None)
            yy, xx = np.mgrid[k0:k0 + win.shape[0], i0:i0 + win.shape[1]]
            kc, ic = float((win * yy).sum() / win.sum()), float((win * xx).sum() / win.sum())
            px, py = (float(q) * 1e9 for q in st.pixel_xy(ic, kc))
            sx, sy = C.to_sample_frame(ev, px, py)
            out.append(math.hypot(sx - ax, sy - ay))
    return out


def test_a_frame_at_the_rule_limit_places_the_atom(tmp_path):
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    limit = next(c for c in sc.claims if c["id"] == "target_site")["evidence"]["max_nm_per_px"]
    assert limit == 0.2
    errs = _placement_errors(limit, tmp_path)
    assert float(np.median(errs)) < 0.05
    # far inside the Cu site spacing (0.255 nm) the claim turns on
    assert sum(e < 0.1 for e in errs) >= len(errs) - 1, errs
