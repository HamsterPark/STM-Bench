"""The viewer's contrast switch (``FrameView.set_contrast``): ``auto`` is the grey scale an MCP agent's image
has (0.5-99.5 percentiles), ``full`` spreads it over the frame's whole range, so that the top of an atom is
not one flat white. Only the grey scale and the colour bar change: what is read off the image does not.

(The rest of the viewer's tests are in ``test_gui_widgets.py``; this file is the console's own, written when
the switch was added.)"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui.widgets import FrameView  # noqa: E402
from stmbench.gui.widgets import frame_view as fv  # noqa: E402
from stmbench.instrument.render import display_array  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def atom_frame(name="sim001.sxm", peak_pm=100.0, tilt_pm=0.0) -> dict:
    """A 64 x 64 z frame (metres): quiet noise, one Gaussian atom (sigma 1.5 px) at (40, 24), and optionally a
    tilt. The atom covers more than 0.5 % of the frame above a quarter of its height, so the percentile
    mapping saturates its top."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:64, 0:64]
    z = rng.normal(0.0, 1.0, (64, 64)) * 1e-12
    z += peak_pm * 1e-12 * np.exp(-((xx - 40) ** 2 + (yy - 24) ** 2) / (2 * 1.5 ** 2))
    z += tilt_pm * 1e-12 * (xx - 32) / 32.0
    return {"frame": name, "array": z, "center_nm": (0.0, 0.0), "size_nm": 10.0, "angle_deg": 0.0,
            "channel": "z", "direction": "forward", "unit": "m", "pixels": 64}


def make_view(frame=None) -> FrameView:
    view = FrameView()
    view.resize(640, 560)
    if frame is not None:
        view.set_frame(frame)
    return view


def grey_at(view: FrameView, col: int, row: int) -> int:
    pt = view.pixel_to_widget(col, row).toPoint()
    return view.grab().toImage().pixelColor(pt).red()


def test_the_default_is_the_agents_image_and_full_maps_the_whole_range(app):
    frame = atom_frame()
    view = make_view(frame)
    assert view.contrast() == "auto"
    shown = view._shown
    # auto: exactly display_array's image and the percentile range on the colour bar
    assert np.array_equal(shown.grey, display_array(frame["array"], "z"))
    flat = frame["array"] - np.median(frame["array"], axis=1, keepdims=True)
    lo, hi = np.percentile(flat, [0.5, 99.5])
    assert shown.lo == pytest.approx(lo * 1e12) and shown.hi == pytest.approx(hi * 1e12)
    assert hi < 0.5 * flat.max()                                        # the top of the atom is above the mapped range
    ring = (24, 42)                                                     # (row, col): 2 px from the centre, 40 % of the peak
    assert shown.grey[24, 40] == 255 and shown.grey[ring] == 255        # both saturate: the top is one flat white
    view.set_contrast("full")
    assert view.contrast() == "full"
    assert shown.lo == pytest.approx(flat.min() * 1e12) and shown.hi == pytest.approx(flat.max() * 1e12)
    assert shown.grey.min() == 0 and shown.grey.max() == 255
    assert shown.grey[24, 40] == 255 and 80 < shown.grey[ring] < 240    # now the shoulder is told from the centre
    # the numbers read off the image are the same in both
    assert np.array_equal(shown.values, flat * 1e12)
    view.set_contrast("auto")
    assert np.array_equal(shown.grey, display_array(frame["array"], "z")) and shown.hi == pytest.approx(hi * 1e12)


def test_the_painted_image_and_the_colour_bar_follow_the_mode(app):
    view = make_view(atom_frame())
    assert grey_at(view, 42, 24) == 255                                          # 2 px from the atom's centre
    ticks_auto = [t for _, t in view._layout().bar_ticks]
    view.set_contrast("full")
    assert 80 < grey_at(view, 42, 24) < 240 and grey_at(view, 40, 24) == 255  # the picture is drawn with the new scale
    ticks_full = [t for _, t in view._layout().bar_ticks]
    assert ticks_full != ticks_auto                                              # and the bar says the new range
    assert ticks_full[0] == fv.format_number(view._shown.hi, 2) or float(ticks_full[0]) == pytest.approx(view._shown.hi, rel=2e-2)


def test_the_mode_holds_for_the_next_frame_and_every_flatten_and_a_bad_mode_is_refused(app):
    view = make_view(atom_frame("sim001.sxm"))
    view.set_contrast("full")
    view.set_frame(atom_frame("sim002.sxm", peak_pm=60.0))                      # the next frame is shown in full too
    flat2 = view._shown.values
    assert view.contrast() == "full" and view._shown.hi == pytest.approx(flat2.max()) and view._shown.lo == pytest.approx(flat2.min())
    for mode in fv.FLATTEN_MODES:
        view.set_flatten(mode)
        assert view.contrast() == "full"
        s = view._shown
        assert s.lo == pytest.approx(np.nanmin(s.values)) and s.hi == pytest.approx(np.nanmax(s.values)), mode
        assert s.grey.min() == 0 and s.grey.max() == 255, mode
    with pytest.raises(ValueError, match="contrast mode"):
        view.set_contrast("bright")
    assert view.contrast() == "full"
    empty = FrameView()                                                          # no frame yet: the mode is kept for the first one
    empty.set_contrast("full")
    empty.set_frame(atom_frame())
    assert empty._shown.grey.max() == 255 and empty._shown.hi == pytest.approx(empty._shown.values.max())


def test_a_marker_reads_the_same_value_in_both_modes_and_a_frame_with_holes_still_maps(app):
    view = make_view(atom_frame())
    p_auto = view._point_at(view.pixel_to_widget(40, 24))
    view.set_contrast("full")
    p_full = view._point_at(view.pixel_to_widget(40, 24))
    assert p_auto["value"] == p_full["value"] and p_auto["unit"] == p_full["unit"] == "pm"
    holes = atom_frame()
    holes["array"][50:, :] = np.nan                                              # rows never acquired
    view.set_frame(holes)
    s = view._shown
    assert np.isfinite(s.lo) and np.isfinite(s.hi) and s.hi > s.lo
    assert s.nan_mask is not None and s.nan_mask[60, 5] and not s.nan_mask[10, 5]
    flat_finite = s.values[np.isfinite(s.values)]
    assert s.lo == pytest.approx(flat_finite.min()) and s.hi == pytest.approx(flat_finite.max())
    constant = atom_frame()
    constant["array"] = np.zeros((16, 16))
    view.set_frame(constant)                                                     # a flat frame: no division by zero
    assert view._shown.hi > view._shown.lo and np.isfinite(view._shown.grey).all()
