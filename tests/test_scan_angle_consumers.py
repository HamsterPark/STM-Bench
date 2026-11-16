"""The consumers of the scan angle that this branch added (desktop console, web console), checked against the
scanner's clockwise-positive convention (PR #18, DESIGN.md section on the scan angle).

What a console shows for a pixel must be what the judge maps for that pixel, at any angle (AGENTS.md invariant 6).
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from stmbench.instrument import Instrument
from stmbench.render import pixel_to_scan_nm
from stmsim.physics.scanner import ScanSettings

P4 = "P4_atom_positioning_cu111"
INDEX_HTML = Path(__file__).resolve().parents[1] / "stmbench" / "console" / "static" / "index.html"


def _geom(angle: float) -> dict:
    return {"cx_nm": 3.0, "cy_nm": -2.0, "w_nm": 8.0, "h_nm": 8.0, "angle_deg": angle, "nx": 32, "ny": 32}


@pytest.mark.parametrize("angle", [-120.0, -30.0, 0.0, 30.0, 90.0])
@pytest.mark.parametrize("col,row", [(0.0, 0.0), (10.0, 20.0), (31.0, 5.5), (-0.5, 32.5)])
def test_the_desktop_console_maps_scan_nm_back_to_the_pixel_it_came_from(angle, col, row):
    pytest.importorskip("PySide6")
    from stmbench.gui.widgets import frame_view as fvm
    g = _geom(angle)
    x, y = pixel_to_scan_nm(g, col, row)
    assert fvm.scan_to_pixel(g, x, y) == pytest.approx((col, row), abs=1e-9)
    # and against the scanner itself, not only against render's own map
    st = ScanSettings(cx=3e-9, cy=-2e-9, w=8e-9, h=8e-9, nx=32, ny=32, angle_deg=angle)
    sx, sy = st.pixel_xy(col, row)
    assert fvm.scan_to_pixel(g, float(sx) * 1e9, float(sy) * 1e9) == pytest.approx((col, row), abs=1e-6)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("angle", [-120.0, 30.0, 90.0])
def test_the_web_console_reads_a_pointer_as_the_judge_maps_that_pixel(angle, tmp_path):
    """Runs the page's own ``pointFrom`` under node with a stub frame and a stub pointer."""
    src = INDEX_HTML.read_text(encoding="utf-8")
    start = src.index("function pointFrom(")
    body = src[start:src.index("\nfunction hover(", start)]
    n, size, cx, cy = 32, 8.0, 3.0, -2.0
    probes = [(0.1, 0.2), (0.9, 0.4), (0.5, 0.5), (0.3, 0.95)]
    frame = json.dumps({"pixels": n, "size_nm": size, "center_nm": [cx, cy], "angle_deg": angle})
    script = (f"const S={{sel:{{name:'f'}}}}; const frameOf=()=>({frame});\n"
              f"{body}\nconst out=[];\nfor (const [fx,fy] of {json.dumps(probes)}) "
              "{ const img={getBoundingClientRect:()=>({left:0,top:0,width:100,height:100})};"
              " const p=pointFrom({clientX:fx*100,clientY:fy*100},img); out.push([p.x,p.y,p.col,p.row]); }\n"
              "console.log(JSON.stringify(out));")
    f = tmp_path / "probe.js"
    f.write_text(script, encoding="utf-8")
    res = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=60, check=True)
    for (x, y, col, row) in json.loads(res.stdout):
        want = pixel_to_scan_nm(_geom(angle), col, row)
        assert (x, y) == pytest.approx(want, abs=1e-9)


def _peak_scan_nm(inst: Instrument, angle: float) -> tuple[float, float, float]:
    out = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=12.0, pixels=96,
                            angle_deg=angle, line_time_s=0.05)
    fr = inst.frame(out["frame"])
    z = np.asarray(fr["array"], float)
    z = z - np.nanmedian(z)
    # the brightest pixel inside the disc that every rotation of the window shares
    rr, cc = np.mgrid[0:96, 0:96]
    inside = np.hypot(rr - 47.5, cc - 47.5) < 0.42 * 96
    z = np.where(inside & np.isfinite(z), z, -np.inf)
    row, col = np.unravel_index(int(np.argmax(z)), z.shape)
    g = {"cx_nm": 0.0, "cy_nm": 0.0, "w_nm": 12.0, "h_nm": 12.0, "angle_deg": angle, "nx": 96, "ny": 96}
    x, y = pixel_to_scan_nm(g, col, row)
    return x, y, float(z[row, col])


def test_a_pixel_read_on_an_atom_in_a_rotated_frame_is_the_atoms_scan_position(tmp_path):
    """The same atom, found in a frame at 0 and in one at 30 degrees, reads the same scan coordinates.

    A frame read with the old anticlockwise map would put the 30 degree reading several nm away.
    """
    inst = Instrument(P4, 0, out_dir=tmp_path / "run")
    x0, y0, h0 = _peak_scan_nm(inst, 0.0)
    assert h0 > 0
    x1, y1, _ = _peak_scan_nm(inst, 30.0)
    assert math.hypot(x1 - x0, y1 - y0) < 0.8, ((x0, y0), (x1, y1))
