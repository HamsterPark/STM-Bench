"""Markers placed on another frame are drawn faded, as an outline.

Written for the round after screen trial 24 (2026-11-26): markers from an earlier frame, drawn on every frame shown, were
hard to tell from new ones.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from stmbench.gui.widgets import frame_view as fvm  # noqa: E402
from tests.test_gui_zoom import app, bump_frame, inside, make_view  # noqa: E402, F401

LEFT = Qt.MouseButton.LeftButton
NO_MOD = Qt.KeyboardModifier.NoModifier


def gradient_frame(name: str, channel: str = "z") -> dict:
    """Dark on the left, light on the right."""
    f = bump_frame(name=name, channel=channel)
    n = f["pixels"]
    f["array"] = np.tile(np.linspace(0.0, 100e-12, n), (n, 1))
    return f


def grab(view) -> np.ndarray:
    img = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return buf[:, :img.width(), [2, 1, 0]].astype(int)


def near_cyan(img: np.ndarray) -> int:
    return int(((img[:, :, 0] < 40) & (img[:, :, 1] > 200) & (img[:, :, 2] > 230)).sum())


def ring_pixels(view, marker: dict, plain: np.ndarray, shown: np.ndarray) -> int:
    """How many pixels around the marker's ring differ from the same frame with no markers."""
    c = view.scan_to_widget(marker["x_nm"], marker["y_nm"])
    r = int(fvm.marker_radius(view._fpx())) + 6
    x, y = int(c.x()), int(c.y())
    a, b = shown[y - r:y + r, x - r:x + r], plain[y - r:y + r, x - r:x + r]
    return int((np.abs(a - b).sum(axis=2) > 30).sum())


def test_a_marker_of_another_frame_is_a_faded_ring_and_one_of_the_shown_frame_is_not(app):
    view = make_view(gradient_frame("sim001.sxm"))
    view.set_tool("marker")
    for fx in (0.2, 0.8):                                         # on the dark part and on the light part
        QTest.mouseClick(view, LEFT, NO_MOD, inside(view, fx, 0.5).toPoint())
    view.select_marker(None)                                      # (the selected marker is drawn in full on any frame)
    mine = grab(view)
    assert near_cyan(mine) > 100                                  # the shown frame's own markers: full strength
    markers = view.markers()
    same = gradient_frame("sim001.sxm", channel="current")        # another channel of the same frame: the same frame
    view.set_frame(same)
    assert near_cyan(grab(view)) > 100
    view.set_frame(gradient_frame("sim002.sxm"))
    view.select_marker(None)
    shown = grab(view)
    assert near_cyan(shown) == 0                                  # none at full strength
    view.clear_markers()
    plain = grab(view)
    view._markers = [dict(m) for m in markers]
    view.update()
    for m in markers:
        assert ring_pixels(view, m, plain, shown) > 15, m         # still seen, on dark and on light ground


def test_a_selected_marker_of_another_frame_keeps_its_full_look_and_a_new_marker_is_full(app):
    view = make_view(gradient_frame("sim001.sxm"))
    QTest.mouseClick(view, LEFT, NO_MOD, inside(view, 0.3, 0.5).toPoint())
    view.set_frame(gradient_frame("sim002.sxm"))
    assert view._on_other_frame(view.markers()[0], selected=False) and not view._on_other_frame(view.markers()[0], True)
    QTest.mouseClick(view, LEFT, NO_MOD, inside(view, 0.7, 0.5).toPoint())
    assert not view._on_other_frame(view.markers()[1], selected=False)
    assert [m["frame"] for m in view.markers()] == ["sim001.sxm", "sim002.sxm"]      # the table's rows and Report ← marker
