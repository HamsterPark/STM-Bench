"""The viewer's contrast switch (``FrameView.set_contrast``): ``auto`` is the grey scale an MCP agent's image
has (0.5-99.5 percentiles), ``full`` spreads it over the frame's whole range, so that the top of an atom is
not one flat white, ``visible`` over the range of the part in view (zoomed onto a terrace, a step elsewhere in
the frame no longer sets the scale). Only the grey scale and the colour bar change: what is read off the image
does not.

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


def stepped_frame(name="sim001.sxm") -> dict:
    """The atom frame with a 600 pm block raised in its top left corner (rows and columns 0-19): a step far from
    the atom, which sets the whole frame's range and leaves the atom a dark bump beside it."""
    frame = atom_frame(name)
    frame["array"] = frame["array"].copy()
    frame["array"][0:20, 0:20] += 600e-12
    return frame


def test_visible_contrast_is_full_for_the_whole_frame_and_the_range_of_the_part_in_view_when_zoomed(app):
    view = make_view(stepped_frame())
    view.set_contrast("full")
    full_grey, full_range = view._shown.grey.copy(), (view._shown.lo, view._shown.hi)
    assert full_range[1] > 500 and full_grey[24, 40] < 60                        # the block sets it; the atom is dark
    view.set_contrast("visible")
    s = view._shown
    assert view.contrast() == "visible" and fv.CONTRAST_MODES == ("auto", "full", "visible")
    assert np.array_equal(s.grey, full_grey) and (s.lo, s.hi) == full_range      # nothing zoomed: the same as full
    view.set_zoom(4, view.pixel_to_widget(40, 24))                               # the atom, at x4: its 16 x 16 pixels
    r0, r1, c0, c1 = view._region()
    assert r1 - r0 <= 17 and c1 - c0 <= 17 and r0 <= 24 < r1 and c0 <= 40 < c1 and (r0, c0) != (0, 0)
    part = s.values[r0:r1, c0:c1]
    assert s.lo == pytest.approx(part.min()) and s.hi == pytest.approx(part.max())
    assert 90 < s.hi < 150 and s.grey[24, 40] == 255 and 80 < s.grey[24, 42] < 240   # the atom's own range: its shoulder shows
    assert grey_at(view, 40, 24) == 255 and 80 < grey_at(view, 42, 24) < 240          # and so does the painted image
    ticks = [t for _, t in view._layout().bar_ticks]
    assert float(ticks[0]) == pytest.approx(s.hi, rel=2e-2)                       # the colour bar says the part's range
    # the readings are the same in every mode: they come from the flattened data
    values = s.values.copy()
    view.set_contrast("auto")
    assert np.array_equal(view._shown.values, values) and view.zoom() == 4
    flat = view._shown.values
    assert view._shown.hi == pytest.approx(np.percentile(flat, 99.5))             # auto does not depend on the view
    view.set_contrast("visible")
    # panned to the corner with the block, the range is the block's; panned to the quiet corner, the noise's
    view.set_tool("hand")
    view.pan_by(1e4, 1e4)
    assert view.view_rect()[:2] == (0.0, 0.0) and view._shown.hi > 500
    view.pan_by(-1e4, -1e4)
    assert view.view_rect()[:2] == (48.0, 48.0)
    quiet = view._shown.values[48:64, 48:64]
    assert view._shown.hi == pytest.approx(quiet.max()) and view._shown.lo == pytest.approx(quiet.min()) and view._shown.hi < 10
    # back to the whole frame: full again
    view.fit()
    assert np.array_equal(view._shown.grey, full_grey) and (view._shown.lo, view._shown.hi) == full_range
    for mode in ("plane", "none", "line"):                                        # and every flatten mode follows the part in view
        view.set_flatten(mode)
        view.set_zoom(2, view.pixel_to_widget(50, 50))
        r0, r1, c0, c1 = view._region()
        part = view._shown.values[r0:r1, c0:c1]
        assert view._shown.lo == pytest.approx(np.nanmin(part)) and view._shown.hi == pytest.approx(np.nanmax(part)), mode
        view.fit()


def test_visible_contrast_follows_the_view_into_another_channel_and_ends_with_a_new_frame(app):
    view = make_view(stepped_frame("sim001.sxm"))
    view.set_contrast("visible")
    view.set_zoom(4, view.pixel_to_widget(40, 24))
    corner = view.view_rect()
    other = stepped_frame("sim001.sxm")                                           # the same frame, another channel
    other["channel"] = "current"
    other["array"] = other["array"] * 1e3
    view.set_frame(other)
    r0, r1, c0, c1 = view._region()
    part = view._shown.values[r0:r1, c0:c1]
    assert view.zoom() == 4 and view.view_rect() == corner
    assert view._shown.lo == pytest.approx(part.min()) and view._shown.hi == pytest.approx(part.max())
    view.set_frame(stepped_frame("sim002.sxm"))                                   # the next frame is shown whole, in full
    assert view.zoom() == 1 and view._region() is None
    assert view._shown.lo == pytest.approx(view._shown.values.min()) and view._shown.hi == pytest.approx(view._shown.values.max())


def test_visible_contrast_over_a_part_with_no_data_falls_back_to_the_frame(app):
    holes = stepped_frame()
    holes["array"][50:, :] = np.nan                                               # rows never acquired
    view = make_view(holes)
    view.set_contrast("visible")
    view.set_zoom(8, view.pixel_to_widget(10, 60))
    r0, r1, c0, c1 = view._region()
    assert r0 >= 50 and not np.isfinite(view._shown.values[r0:r1, c0:c1]).any()    # the view holds nothing but holes
    s = view._shown
    finite = s.values[np.isfinite(s.values)]
    assert np.isfinite(s.lo) and np.isfinite(s.hi) and s.hi > s.lo
    assert s.lo == pytest.approx(finite.min()) and s.hi == pytest.approx(finite.max())   # the frame's own range, not nothing
