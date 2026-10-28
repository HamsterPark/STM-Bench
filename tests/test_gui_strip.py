"""The strip of thumbnails beside the viewer, as a screen agent meets it: every frame of the episode can be reached
from it, newest first, a window of them at a time, with the frame on show marked.

Written for the round after screen trial 5 (step 42: "scrolling the wheel over it did nothing, so older frames
(sim005) are not reachable there"; the Data tab's list cost three more steps).
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from stmbench.gui import shots, style  # noqa: E402
from stmbench.gui.center import MAX_THUMBS, MIN_THUMBS, THUMB_PX  # noqa: E402
from tests.gui_helpers import app, fonts, open_window, run_button, scan  # noqa: E402, F401


def episode(win, n: int, first: int = 0) -> list[str]:
    """n more frames (small ones, each of another window so that their thumbnails differ), oldest first."""
    return [scan(win, 0.4 * i, -0.3 * i, 4.0, 16) for i in range(first, first + n)]


def shown(win) -> list[str]:
    """The names of the thumbnails on the strip, top to bottom."""
    return [b.name for b in win.center.thumbs if b.isVisible()]


def strip(win):
    return win.center.strip


def wheel_over(win, delta: int = 120) -> None:
    shots.wheel(win, win.center.thumbs[1], None, delta)


# ── reach ──
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_frame_is_reached_from_the_strip_by_wheel_notches_one_thumbnail_each(fonts, open_window, lang):
    win = open_window(lang)
    names = episode(win, 15)
    newest_first = names[::-1]
    n = len(shown(win))
    assert n == strip(win).capacity() and 5 <= n <= MAX_THUMBS                    # a window of the newest frames
    assert shown(win) == newest_first[:n] and win.center.thumbs[0].name == names[-1]
    seen = set(shown(win))
    for i in range(1, len(names) - n + 3):                                        # down, past the end (it stops there)
        wheel_over(win, -120)
        top = min(i, len(names) - n)                                              # one thumbnail a notch
        assert shown(win) == newest_first[top:top + n]
        seen |= set(shown(win))
    assert seen == set(names)                                                     # every frame was on the strip
    assert shown(win)[-1] == names[0]                                             # the oldest, at the bottom
    for _ in range(len(names) - n + 3):                                           # and back up
        wheel_over(win, +120)
    assert shown(win) == newest_first[:n]


def test_the_pages_buttons_move_a_window_at_a_time_and_say_how_many_frames_lie_beyond(open_window):
    win = open_window("en")
    names = episode(win, 15)
    newest_first = names[::-1]
    n = strip(win).capacity()
    newer, older = shots.widget(win, "strip.newer"), shots.widget(win, "strip.older")
    assert not newer.isEnabled() and older.isEnabled()
    assert older.text() == f"▼ {15 - n}" and newer.text() == "▲"
    run_button(win, "strip.older")                                                # a page: the next n frames
    assert shown(win) == newest_first[n:2 * n]
    assert newer.isEnabled() and newer.text() == f"▲ {n}"
    assert older.text() == (f"▼ {15 - 2 * n}" if 15 > 2 * n else "▼")
    for _ in range(4):
        run_button(win, "strip.older")
    assert shown(win) == newest_first[15 - n:] and not older.isEnabled() and older.text() == "▼"   # the oldest n
    run_button(win, "strip.newer")
    assert shown(win) == newest_first[15 - 2 * n:15 - n]
    for _ in range(4):
        run_button(win, "strip.newer")
    assert shown(win) == newest_first[:n] and not newer.isEnabled() and newer.text() == "▲"


def test_with_few_frames_the_paging_buttons_are_not_there_and_the_thumbnails_do_not_move_when_they_come(open_window):
    win = open_window("en")
    names = episode(win, 3)
    n = strip(win).capacity()
    assert len(names) < n
    newer, older = shots.widget(win, "strip.newer"), shots.widget(win, "strip.older")
    assert not newer.isVisible() and not older.isVisible()
    before = [b.mapTo(win, b.rect().topLeft()) for b in win.center.thumbs[:3]]
    episode(win, n + 2 - 3, first=3)                                              # more frames than the strip holds
    assert older.isVisible() and newer.isVisible()
    assert len(shown(win)) == n
    after = [b.mapTo(win, b.rect().topLeft()) for b in win.center.thumbs[:3]]
    assert before == after                                                        # the room for them was kept
    top = win.center.thumbs[0].name
    wheel_over(win, +120)                                                         # nothing newer to show: no move
    assert win.center.thumbs[0].name == top == win._data["frames"][-1]["frame"]


# ── the frame on show ──
def test_the_frame_on_show_is_marked_and_a_click_on_a_thumbnail_shows_that_frame(open_window):
    win = open_window("en")
    names = episode(win, 12)
    thumbs = win.center.thumbs
    assert win.viewed == ("frame", names[-1])
    assert thumbs[0].isChecked() and sum(b.isChecked() for b in thumbs) == 1
    third = thumbs[2]
    shots.click(win, third)
    shots.settle(win)
    assert win.viewed == ("frame", third.name) and third.isChecked() and sum(b.isChecked() for b in thumbs) == 1
    # the mark is drawn, in the window's accent colour, round the thumbnail (the toolkit's own is a darker edge)
    on = third.grab().toImage()
    third.setChecked(False)
    off = third.grab().toImage()
    third.setChecked(True)
    accent = style.COLORS["accent"].lstrip("#")
    ring = tuple(int(accent[i:i + 2], 16) for i in (0, 2, 4))
    for x, y in ((1, third.height() // 2), (third.width() - 2, third.height() // 2), (third.width() // 2, 1)):
        px = on.pixelColor(x, y)
        assert (px.red(), px.green(), px.blue()) == ring
    assert on != off
    # an older frame, reached by the wheel, shows when clicked, and stays marked where it is
    for _ in range(5):
        wheel_over(win, -120)
    old = [b for b in thumbs if b.isVisible()][3]
    shots.click(win, old)
    shots.settle(win)
    assert win.viewed == ("frame", old.name) and old.isChecked() and sum(b.isChecked() for b in thumbs) == 1


def test_a_frame_picked_elsewhere_is_brought_into_the_window_and_a_new_scan_brings_the_newest(open_window):
    win = open_window("en")
    names = episode(win, 14)
    n = len(shown(win))
    assert names[0] not in shown(win)
    win.request_frame(names[0])                                                   # as the Data tab's list does
    shots.settle(win)
    assert win.viewed == ("frame", names[0]) and names[0] in shown(win)
    assert [b.name for b in win.center.thumbs if b.isVisible() and b.isChecked()] == [names[0]]    # on show, marked
    assert len(shown(win)) == n
    fresh = scan(win, 0.0, 0.0, 4.0, 16)                                          # a new frame: the strip is at the top again
    assert shown(win)[0] == fresh and win.center.thumbs[0].isChecked()


def test_the_readers_place_on_the_strip_is_kept_when_a_spectrum_is_read_and_when_the_data_is_listed_again(open_window):
    win = open_window("en")
    names = episode(win, 13)
    for _ in range(4):
        wheel_over(win, -120)
    top = win.center.thumbs[0].name
    assert top == names[::-1][4]
    win.right.setCurrentWidget(win.spectroscopy)
    shots.fill(win, {"act.sts.v_start": "-0.2", "act.sts.v_end": "0.2", "act.sts.points": "16"})
    run_button(win, "run.sts")
    shots.settle(win)
    assert win.viewed[0] == "spectrum"
    assert win.center.thumbs[0].name == top and not any(b.isChecked() for b in win.center.thumbs)
    win._refresh_thumbs()
    assert win.center.thumbs[0].name == top


# ── what the strip is made of ──
def test_the_wheel_over_the_strip_is_taken_finer_wheels_add_up_and_ctrl_does_nothing(open_window):
    win = open_window("en")
    episode(win, 12)
    qwin = win.windowHandle()
    b = win.center.thumbs[1]
    pos = b.mapTo(win, b.rect().center())

    def turn(delta, mods=Qt.KeyboardModifier.NoModifier):
        ev = QWheelEvent(QPointF(pos), QPointF(qwin.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, delta),
                         Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(qwin, ev)
        QApplication.processEvents()
        return ev.isAccepted()

    first = win.center.thumbs[0].name
    turn(-120, Qt.KeyboardModifier.ControlModifier)
    assert win.center.thumbs[0].name == first
    for _ in range(3):
        turn(-30)                                                                 # three quarters of a notch: nothing yet
    assert win.center.thumbs[0].name == first
    turn(-30)
    assert win.center.thumbs[0].name != first                                     # the fourth makes one notch
    assert win.center.strip.first_index() == 1


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("height", [1080, 1016])
def test_the_strip_fits_its_column_whole_in_both_languages_and_both_window_heights(fonts, open_window, lang, height):
    win = open_window(lang, height=height)
    episode(win, 14)
    s = strip(win)
    n = len(shown(win))
    assert MIN_THUMBS <= n == s.capacity()
    inside = s.rect()
    gap = s.layout().spacing()
    assert gap == style.px(3)
    prev_bottom = -1
    buttons = (shots.widget(win, "strip.newer"), *[t for t in win.center.thumbs if t.isVisible()],
               shots.widget(win, "strip.older"))
    for b in buttons:
        r = b.geometry()
        assert inside.contains(r), (b.objectName(), r, inside)                    # whole, inside the strip
        assert r.top() > prev_bottom, (b.objectName(), "overlaps the one above")
        prev_bottom = r.bottom()
        shots.hit(win, b)
        if isinstance(b, QPushButton):
            assert b.width() >= style.px(56) and b.height() >= style.px(28)
            assert b.fontMetrics().horizontalAdvance(b.text()) + 2 * style.px(10) <= b.width()      # its text is not cut
        else:
            assert b.width() == style.px(THUMB_PX + 8)
    # one more thumbnail would not have fitted (unless the strip is at its largest)
    t = style.px(THUMB_PX + 8)
    nav = shots.widget(win, "strip.newer").height()
    assert n == MAX_THUMBS or 2 * nav + (n + 1) * t + (n + 2) * gap > s.height()
