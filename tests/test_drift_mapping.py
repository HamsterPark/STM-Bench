"""A reported scan-frame position is placed on the sample through the drift of the row that
imaged it: the renderer adds one drift/creep offset per acquisition row, and each saved frame
now records them (``row_offsets_nm``)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from stmbench.trackB import claims as C
from stmsim.physics.scanner import SIG_Z
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"

FRAME = {"kind": "scan_saved", "sim_s": 10.0, "idx": 1, "complete": True, "path": "session/sim001.sxm",
         "cx_m": 0.0, "cy_m": 0.0, "w_m": 60e-9, "h_m": 60e-9, "angle_deg": 0.0,
         "nx": 256, "ny": 256, "drift_m": [0.0, 0.0]}


def _rows(ny: int, dx_per_row: float, dy_per_row: float = 0.0, x0: float = 0.0) -> list:
    return [[x0 + k * dx_per_row, k * dy_per_row] for k in range(ny)]


# ── the frame's own scan → sample map ────────────────────────────────────────
def test_a_late_row_is_placed_through_its_own_drift(tmp_path):
    """Regression: a frame recorded only the drift at its start, so an atom imaged near the
    bottom of a slow frame was placed where the sample had been when the frame began.

    Rendered on the simulator with 0.02 nm/s of drift: the atom sits ~85 % down a 128 s
    frame. Through the frame-start offset it lands 2.2 nm from its true place; through the
    offset of the row that imaged it, within 0.05 nm."""
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    seed = next(s for s in range(40) if sc.draw_hidden(s)["tip"]["condition"] == "good")
    w = sc.build_world(seed, session_dir=tmp_path / "s")
    w.clock.pause()                                   # no wall time leaks into the frame
    w.drift_v_m_per_s = np.array([0.02e-9, 0.0, 0.0])
    reg = w.surface.site.adatoms
    atom = min(reg.atoms.values(), key=lambda a: math.hypot(*reg.site_xy(a.i, a.j, a.sub)))
    ax, ay = (v * 1e9 for v in reg.site_xy(atom.i, atom.j, atom.sub))
    st = w.scan
    st.cx, st.cy, st.w, st.h, st.nx, st.ny = ax * 1e-9, (ay + 7.0) * 1e-9, 20e-9, 20e-9, 128, 128
    st.line_time_fwd_s = st.line_time_bwd_s = 0.5
    w.scan_start()
    fr = w.frame
    fr.rows_done, fr.finished = fr.rows_total, True
    w.save_frame()
    ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
    assert len(ev["row_offsets_nm"]) == 128 and ev["scan_dir"] == "down"
    # where the atom appears: intensity centroid of the bump in the row-flattened topograph
    z = fr.data[SIG_Z][0]
    z = z - np.median(z, axis=1, keepdims=True)
    k, i = np.unravel_index(int(np.argmax(z)), z.shape)
    win = z[k - 4:k + 5, i - 4:i + 5]
    win = np.clip(win - 0.5 * win.max(), 0, None)
    yy, xx = np.mgrid[k - 4:k + 5, i - 4:i + 5]
    kc, ic = float((win * yy).sum() / win.sum()), float((win * xx).sum() / win.sum())
    px, py = (float(v) * 1e9 for v in st.pixel_xy(ic, kc))
    assert kc > 96                                    # imaged in the last quarter of the frame
    sx, sy = C.to_sample_frame(ev, px, py)
    assert math.hypot(sx - ax, sy - ay) < 0.05
    start = (px + ev["drift_m"][0] * 1e9, py + ev["drift_m"][1] * 1e9)
    assert math.hypot(start[0] - ax, start[1] - ay) > 1.5


@pytest.mark.parametrize("angle,scan_dir", [(0.0, "down"), (30.0, "down"), (-75.0, "up")])
def test_sample_and_scan_maps_are_inverse(angle, scan_dir):
    fr = {**FRAME, "angle_deg": angle, "scan_dir": scan_dir,
          "row_offsets_nm": _rows(256, 0.01, -0.004, x0=3.0)}
    for x, y in [(0.0, 0.0), (12.5, -20.0), (-28.0, 27.0)]:
        sx, sy = C.to_sample_frame(fr, x, y)
        assert C.to_scan_frame(fr, sx, sy) == pytest.approx((x, y), abs=1e-6)


def test_the_row_a_point_falls_in_sets_its_offset():
    """Top row first when scanning down, bottom row first when scanning up."""
    rows = _rows(256, 0.02)                            # 0 … 5.1 nm along x
    down = {**FRAME, "scan_dir": "down", "row_offsets_nm": rows}
    up = {**down, "scan_dir": "up"}
    top, bottom = (0.0, 29.9), (0.0, -29.9)
    assert C.to_sample_frame(down, *top)[0] == pytest.approx(0.0, abs=0.02)
    assert C.to_sample_frame(down, *bottom)[0] == pytest.approx(5.1, abs=0.02)
    assert C.to_sample_frame(up, *top)[0] == pytest.approx(5.1, abs=0.02)
    assert C.to_sample_frame(up, *bottom)[0] == pytest.approx(0.0, abs=0.02)


def test_frames_without_row_records_keep_their_start_offset():
    old = {**FRAME, "drift_m": [7e-9, -3e-9]}
    assert C.to_sample_frame(old, 5.0, 2.0) == pytest.approx((12.0, -1.0))
    assert C.to_scan_frame(old, 12.0, -1.0) == pytest.approx((5.0, 2.0))
