"""The Report tab's list of cited files has its two lines for good: nothing below the evidence field moves while the reader types
in it (screen trial 23: Submit jumped by 28 px and two clicks landed on labels), a longer list ends in "+N more", and the
form still fits in every paper scenario (tests/test_gui_report_fit.py, test_gui_app.py)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from stmbench.gui import shots  # noqa: E402
from tests.gui_helpers import app, fonts, open_window  # noqa: E402, F401

NAMES = ["sim001.sxm", "20260101_000002.dat", "20260101_000003.dat", "sim004.sxm", "sim005.sxm", "sim006.sxm", "sim007.sxm"]


def below(win) -> dict[str, int]:
    rep = win.sidebar.report
    names = ("report.unit", "report.sigma", "report.note", "report.submit", "finish.begin", "report.evidence_full")
    return {n: shots.widget(win, n).mapTo(rep, shots.widget(win, n).rect().topLeft()).y() for n in names}


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_nothing_below_the_evidence_field_moves_while_it_is_typed_in(fonts, open_window, lang):
    win = open_window(lang, scenario="P1_barth1990_au111")
    win.sidebar.show_tab("report")
    full = shots.widget(win, "report.evidence_full")
    before, height = below(win), full.height()
    assert full.text() == "" and height > 0                          # its space is there before anything is typed
    ev = shots.widget(win, "report.evidence")
    for n in range(1, len(NAMES) + 1):                               # one file, two, ... until the list is longer than its lines
        ev.setText(", ".join(NAMES[:n]))
        assert below(win) == before and full.height() == height, n
    assert full.text().count("\n") <= 1                              # two lines at most
    assert full.text().rstrip().endswith("more")                     # the rest is told, not shown
    ev.setText("")
    assert below(win) == before and full.text() == ""


def test_a_list_of_two_files_fills_its_lines_in_full_and_a_short_one_shows_nothing(open_window):
    win = open_window("en", scenario="P1_barth1990_au111")
    win.sidebar.show_tab("report")
    full, ev = shots.widget(win, "report.evidence_full"), shots.widget(win, "report.evidence")
    ev.setText("2.dat")
    assert full.text() == ""                                         # the field shows it whole
    ev.setText("20260101_000002.dat, 20260101_000003.dat")
    assert "more" not in full.text() and "20260101_000002.dat" in full.text() and "20260101_000003.dat" in full.text()
