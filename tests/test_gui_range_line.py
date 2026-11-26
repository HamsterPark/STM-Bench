"""The range line beside the right tabs goes with its tab and its focus (screen trial 24)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from stmbench.gui import shots  # noqa: E402
from tests.gui_helpers import app, fonts, open_window, run_button  # noqa: E402, F401

P1, P4 = "P1_barth1990_au111", "P4_atom_positioning_cu111"

def starred(rep, key: str) -> bool:
    return rep.labels[key].text().endswith("*</span>")



def test_the_range_line_goes_with_its_tab(open_window):
    win = open_window("en", seed=3)
    shots.click(win, shots.widget(win, "act.move_tip.y_nm"))
    assert win.field_help.full_text().startswith("tip y")
    win.right.setCurrentWidget(win.spectroscopy)
    assert win.field_help.full_text() == ""
    win.right.setCurrentWidget(win.operate)
    assert win.field_help.full_text() == ""                                      # not brought back by coming back


def test_the_range_line_goes_when_focus_moves_to_another_widget_of_the_window_not_when_it_moves_to_another_field(open_window):
    win = open_window("en", seed=3)
    y, x = shots.widget(win, "act.move_tip.y_nm"), shots.widget(win, "act.move_tip.x_nm")
    shots.click(win, y)
    assert win.field_help.full_text().startswith("tip y")
    shots.click(win, shots.widget(win, "act.pulse.bias_v"))                      # another numeric field: its own range
    assert win.field_help.full_text().startswith("pulse")
    win.sidebar.report.submit.setFocus()                                          # another widget of the window
    assert win.field_help.full_text() == ""
    shots.click(win, x)
    assert win.field_help.full_text().startswith("tip x")
    win.clearFocus()                                                              # nothing takes the focus: the line stays
    assert win.field_help.full_text().startswith("tip x")
