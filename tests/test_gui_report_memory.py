"""The Report form keeps its entries per result (screen trial 23: value, evidence, note and position carried over from one
result to the next, and a radius would have been submitted with the files of another result)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from stmbench.gui import shots  # noqa: E402
from tests.gui_helpers import app, fonts, open_window, run_button  # noqa: E402, F401

P1 = "P1_barth1990_au111"
A, B = "stripe_orientation_deg", "stripe_period_nm"        # A names a place (position fields), B a number about the surface


def fields(win) -> dict[str, str]:
    return {k: e.text() for k, e in win.sidebar.report.fields.items()}


def test_each_result_keeps_its_own_entries(open_window):
    win = open_window("en", scenario=P1)
    win.sidebar.show_tab("report")
    rep = win.sidebar.report
    run_button(win, f"report.claim.{A}")
    shots.fill(win, {"report.value": "30", "report.x_nm": "1", "report.y_nm": "2", "report.frame": "sim001.sxm",
                     "report.evidence": "sim001.sxm", "report.note": "a note"})
    a_held = fields(win)
    run_button(win, f"report.claim.{B}")                       # a result not yet opened: empty, except the unit
    unit_b = rep._claims[B]["unit"]
    assert {k: v for k, v in fields(win).items() if v} == ({"unit": unit_b} if unit_b else {})
    shots.fill(win, {"report.value": "4.5", "report.evidence": "2.dat"})
    b_held = fields(win)
    run_button(win, f"report.claim.{A}")                       # back: A's entries
    assert fields(win) == a_held
    run_button(win, "report.submit")                            # submit A: A's entries stay
    shots.settle(win)
    assert fields(win) == a_held
    run_button(win, f"report.claim.{B}")                       # B still holds B's
    assert fields(win) == b_held
    # a result's position fields stay greyed and empty for a result with no position
    assert not rep.fields["x_nm"].isEnabled() and rep.fields["x_nm"].text() == ""


def test_a_result_not_opened_before_shows_today_s_unit_prefill(open_window):
    win = open_window("en", scenario=P1)
    win.sidebar.show_tab("report")
    rep = win.sidebar.report
    run_button(win, f"report.claim.{B}")
    assert rep.fields["unit"].text() == rep._claims[B]["unit"]
    assert all(v == "" for k, v in fields(win).items() if k != "unit")
