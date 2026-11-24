"""The Operate tab's Tip group, with its one line of "during move" fields, fits the right column in width and the tab in height,
in every font the console meets (the fit test of the Report form, tests/test_gui_report_fit.py, for this group).

Written for the delivery to OSWorld-Science, whose guest is Ubuntu 22.04 with DejaVu Sans. The line holds three text fields
(the range is a field too, checked against the table's choices): a row of range buttons beside two fields measured 784 px
(English) and 981 px (Chinese) in a column of 704 px, and a second line took 56 px of height that the tab does not have, so that
the controls at its foot were covered by the list of recent results. The Chinese window shows both languages in each label.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QGroupBox, QWidget  # noqa: E402

from stmbench.gui import layout, shots, style  # noqa: E402
from stmbench.gui.app import RIGHT_PX  # noqa: E402
from tests.gui_helpers import app, fonts, open_window, run_button  # noqa: E402, F401

SPARE_PX = 8


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_tip_group_and_its_during_move_line_fit_the_right_column_and_the_tab(fonts, open_window, lang):
    win = open_window(lang)
    win.resize(1920, 1080)
    shots.settle(win)
    QApplication.processEvents()
    panel = win.operate
    box = panel.findChild(QGroupBox, "group.tip")
    # width: the group is narrower than the column by a margin, so it never widens the column or reaches its edge
    need = box.minimumSizeHint().width()
    assert need <= style.px(RIGHT_PX) - style.px(SPARE_PX), (lang, need)
    edge = panel.viewport().width()
    clipped = [w.objectName() or type(w).__name__ for w in box.findChildren(QWidget)
               if w.isVisible() and w.mapTo(panel.widget(), w.rect().bottomRight()).x() >= edge]
    assert clipped == []
    # height: the tab still shows everything without scrolling
    assert not panel.verticalScrollBar().isVisible() and panel.verticalScrollBar().maximum() == 0
    for c in layout.during_cells():                                          # each can be clicked where it is
        shots.hit(win, shots.widget(win, f"during.{c.action}.{c.param}"))
    shots.hit(win, shots.widget(win, "act.wait.seconds"))                    # and so is the last control of the tab
