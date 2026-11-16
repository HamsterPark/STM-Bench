"""The Report tab's form fits the left column in every font the console meets, before and after Finish.

Written for the delivery to OSWorld-Science, whose guest is Ubuntu 22.04 with DejaVu Sans: in screen trials 9 and 10
(Linux, DejaVu, 1920x1080) the form was 392 px (English) or 403 px (Chinese) wide in a column of 358 px, so the y_nm
field, the label of the frame field and the "+ shown file" button were cut, the right of Submit report and Finish
episode was clipped, and the text moved left after Finish. The column was sized by the other tabs: the form sits in a
scroll area, whose own minimum says nothing of the form's.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from stmbench.gui import shots  # noqa: E402
from tests.gui_helpers import app  # noqa: E402, F401
from tests.test_gui_app import fonts, open_window  # noqa: E402, F401


def clipped(win) -> list[str]:
    """Names of the form's visible widgets that reach past the scroll area's viewport."""
    rep, scroll = win.sidebar.report, win.sidebar.report_scroll
    edge = scroll.viewport().width()
    return [w.objectName() or type(w).__name__ for w in rep.findChildren(QWidget)
            if w.isVisible() and w.mapTo(rep, w.rect().bottomRight()).x() >= edge]


@pytest.mark.parametrize("scenario", ["P4_atom_positioning_cu111", "P1_barth1990_au111", "P3_corral_cu111"])
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_control_of_the_report_form_is_inside_the_left_column_before_and_after_finish(fonts, open_window, lang,
                                                                                            scenario):
    """P1 has the most claims (four, the tallest form), P3 three claims of which none takes a position (its position
    fields are greyed), P4 one. The evidence field has a row of its own, with its button on the label's line (2026-11-16)."""
    win = open_window(lang, scenario=scenario)
    win.resize(1920, 1080)
    win.sidebar.show_tab("report")
    shots.settle(win)
    QApplication.processEvents()
    bar = win.sidebar.report_scroll.horizontalScrollBar()
    rep = win.sidebar.report
    for step in ("before Finish", "after Finish"):
        assert rep.width() <= win.sidebar.report_scroll.viewport().width(), step      # the form is as wide as it needs
        assert clipped(win) == [], step
        assert bar.maximum() == 0, step                                              # nothing to scroll sideways
        if step == "before Finish":
            rep.set_finished()
            QApplication.processEvents()
    assert rep.finish_message.isVisible()
    assert rep.finish_message.mapTo(rep, rep.finish_message.rect().topLeft()).x() >= 0
    assert win.sidebar.width() >= win.sidebar.minimumSizeHint().width()
    assert win.center.width() >= win.center.minimumSizeHint().width()                # the image view keeps its room

