"""A drag with the Marker tool pans a zoomed frame; a click still marks.

Written for the round after screen trial 24 (2026-11-26): reading atom positions on a zoomed frame, the agent switched
between the Pan and Marker tools 16 times.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from tests.test_gui_zoom import app, bump_frame, inside, make_view  # noqa: E402, F401

LEFT = Qt.MouseButton.LeftButton
NO_MOD = Qt.KeyboardModifier.NoModifier


def zoomed_view(zoom: int = 4):
    view = make_view(bump_frame())
    view.set_zoom(zoom)
    return view


def record(view):
    log = []
    view.markers_changed.connect(lambda m: log.append(("markers_changed", len(m))))
    view.marker_added.connect(lambda m: log.append(("marker_added", m["col"], m["row"])))
    return log


def drag(view, a: QPointF, b: QPointF, steps: int = 6) -> None:
    QTest.mousePress(view, LEFT, NO_MOD, a.toPoint())
    for i in range(1, steps + 1):
        QTest.mouseMove(view, (a + (b - a) * (i / steps)).toPoint())
    QTest.mouseRelease(view, LEFT, NO_MOD, b.toPoint())


def test_a_click_in_the_marker_tool_on_a_zoomed_frame_marks_once_at_the_press_point_in_signal_order(app):
    view = zoomed_view()
    log = record(view)
    pos = inside(view, 0.5, 0.5).toPoint()
    want = view._point_at(QPointF(pos))
    QTest.mouseClick(view, LEFT, NO_MOD, pos)                    # press and release at one point, no motion (xdotool)
    assert len(view.markers()) == 1 and (view.markers()[0]["col"], view.markers()[0]["row"]) == (want["col"], want["row"])
    assert [e[0] for e in log] == ["markers_changed", "marker_added"]


def test_a_press_that_moves_a_pixel_or_two_is_still_a_click_at_the_press_point(app):
    view = zoomed_view()
    pos = inside(view, 0.4, 0.6).toPoint()
    want = view._point_at(QPointF(pos))
    before = view.view_rect()
    QTest.mousePress(view, LEFT, NO_MOD, pos)
    QTest.mouseMove(view, pos + QPoint(2, 1))
    QTest.mouseRelease(view, LEFT, NO_MOD, pos + QPoint(2, 1))
    assert len(view.markers()) == 1 and view.markers()[0]["col"] == want["col"] and view.view_rect() == before


def test_a_drag_in_the_marker_tool_on_a_zoomed_frame_pans_with_the_closed_hand_and_marks_nothing(app):
    view = zoomed_view()
    log = record(view)
    a, b = inside(view, 0.6, 0.6), inside(view, 0.4, 0.45)
    before = view.view_rect()
    QTest.mousePress(view, LEFT, NO_MOD, a.toPoint())
    assert view.markers() == []                                  # nothing at the press: click or drag is not known yet
    for i in range(1, 7):
        QTest.mouseMove(view, (a + (b - a) * (i / 6)).toPoint())
    assert view.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    QTest.mouseRelease(view, LEFT, NO_MOD, b.toPoint())
    after = view.view_rect()
    assert view.markers() == [] and log == []
    assert after[0] > before[0] and after[1] > before[1]         # the image followed the pointer up and left
    assert view.cursor().shape() == Qt.CursorShape.CrossCursor   # and the marker tool's cursor is back
    QTest.mouseClick(view, LEFT, NO_MOD, inside(view, 0.5, 0.5).toPoint())
    assert len(view.markers()) == 1                              # the next click marks


def test_the_pan_follows_the_pointer_from_where_the_button_went_down(app):
    pan = zoomed_view()
    pan.set_tool("pan")
    mark = zoomed_view()
    a, b = inside(pan, 0.6, 0.6), inside(pan, 0.3, 0.3)
    drag(pan, a, b)
    drag(mark, a, b)
    assert mark.view_rect() == pytest.approx(pan.view_rect())


def test_on_a_frame_that_is_not_zoomed_a_press_marks_at_once_as_before_and_the_other_tools_are_unchanged(app):
    view = make_view(bump_frame())
    pos = inside(view, 0.5, 0.5).toPoint()
    QTest.mousePress(view, LEFT, NO_MOD, pos)
    assert len(view.markers()) == 1                              # at the press, not the release
    QTest.mouseRelease(view, LEFT, NO_MOD, pos)
    before = view.view_rect()
    drag(view, inside(view, 0.3, 0.3), inside(view, 0.6, 0.6))
    assert view.view_rect() == before and len(view.markers()) == 2
    zoomed = zoomed_view()
    zoomed.set_tool("profile")
    got = []
    zoomed.profile_changed.connect(got.append)
    before = zoomed.view_rect()
    drag(zoomed, inside(zoomed, 0.3, 0.3), inside(zoomed, 0.6, 0.6))
    assert got and got[-1] and zoomed.view_rect() == before and zoomed.markers() == []
