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


def _label_text_height(label) -> int:
    return label.sizeHint().height()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_reserved_label_draws_both_lines_whole_underscores_included(fonts, open_window, lang):
    """The second line of the list lost its underscores ("20260101 000002.dat"): the label was as tall as two line spacings and
    the underscore sits below the baseline. It is sized from what the style sheet draws, and a pixel check sees the underscore."""
    from PySide6.QtGui import QColor
    win = open_window(lang, scenario="P1_barth1990_au111")
    win.sidebar.show_tab("report")
    full, ev = shots.widget(win, "report.evidence_full"), shots.widget(win, "report.evidence")
    names = "20260101_000002.dat, 20260101_000003.dat, 20260101_000004.dat"
    ev.setText(names)
    assert full.text().count("\n") == 1
    assert full.heightForWidth(full.width()) <= full.height() and full.sizeHint().height() <= full.height()
    # the underscore of the second line is painted: a text of one line of capitals above a line of underscores alone
    full.setText("XXXX\n________")
    img = full.grab().toImage()
    bg = img.pixelColor(1, 1)
    rows = [y for y in range(img.height()) if any(img.pixelColor(x, y) != bg for x in range(1, min(img.width(), 200)))]
    line = full.fontMetrics().lineSpacing()
    assert any(y >= line for y in rows)                                 # something below the first line: the underscores
    assert max(rows) <= img.height() - 3                                 # with room under the underscores


def test_the_cut_list_keeps_the_separators_of_the_field(open_window):
    win = open_window("en", scenario="P1_barth1990_au111")
    win.sidebar.show_tab("report")
    full, ev = shots.widget(win, "report.evidence_full"), shots.widget(win, "report.evidence")
    ev.setText(", ".join(NAMES))
    first, second = full.text().split(chr(10))
    assert first.endswith(",") and ", " in first                           # the names read as they are typed: "a, b,"
    assert second.endswith("more") and ", … +" in second                  # and the rest is counted after a separator
    assert (first + " " + second).split(" +")[0].replace(",", "").split()[0] == NAMES[0]
