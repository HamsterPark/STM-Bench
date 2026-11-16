"""Four usage gaps found by a sighted check of P3 through the screen interface (2026-11-16, p3check1): the console
worked, but nothing on the screen said how. Each is about how the software is read, none about the experiment.

1. The Report form greys out x_nm, y_nm and frame for a claim the rule card gives no position.
2. The evidence field has the form's width, and the whole list of files is shown under it when it does not fit.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from stmbench.gui import shots  # noqa: E402
from stmbench.gui.labels import one, tr  # noqa: E402
from tests.gui_helpers import app, run_button  # noqa: E402, F401
from tests.test_gui_app import fonts, open_left, open_window  # noqa: E402, F401

P3 = "P3_corral_cu111"
P1 = "P1_barth1990_au111"
POSITION = ("x_nm", "y_nm", "frame")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_claim_without_a_position_greys_the_position_fields_and_keeps_them_empty(open_window, lang):
    win = open_window(lang, scenario=P3)
    open_left(win, "report")
    rep = win.sidebar.report
    assert all(rep.fields[k].isEnabled() for k in POSITION)             # no claim picked yet: nothing is ruled out
    for cid in ("peak_lo_mev", "peak_hi_mev", "radius_nm"):
        run_button(win, f"report.claim.{cid}")
        assert not any(rep.fields[k].isEnabled() or rep.labels[k].isEnabled() for k in POSITION), cid
        assert all(rep.fields[k].text() == "" for k in POSITION), cid
    assert rep.fields["value"].isEnabled() and rep.fields["evidence"].isEnabled()
    middle = lambda w: w.grab().toImage().pixelColor(w.width() // 2, w.height() // 2).name()      # noqa: E731
    assert middle(rep.fields["x_nm"]) != middle(rep.fields["value"])                  # greyed to the eye too, not only inert
    assert middle(rep.fields["frame"]) != middle(rep.fields["evidence"])
    win.sidebar.report.fill_position(1.0, 2.0, "sim001.sxm")            # a marker sent to a claim that takes none
    assert all(rep.fields[k].text() == "" for k in POSITION)
    tone, text = win.recent.entries()[0]
    assert tone == "note" and text == tr("rep_no_position", lang)


def test_a_claim_with_a_position_enables_the_fields_again(open_window):
    win = open_window("en", scenario=P1)
    open_left(win, "report")
    rep = win.sidebar.report
    run_button(win, "report.claim.stripe_period_nm")
    assert not any(rep.fields[k].isEnabled() for k in POSITION)
    run_button(win, "report.claim.stripe_orientation_deg")
    assert all(rep.fields[k].isEnabled() and rep.labels[k].isEnabled() for k in POSITION)
    shots.fill(win, {"report.x_nm": "1"})
    assert rep.fields["x_nm"].text() == "1"


def test_p4_takes_its_position(open_window):
    win = open_window("en")                                              # one claim, picked at the start, at a place
    rep = win.sidebar.report
    assert all(rep.fields[k].isEnabled() for k in POSITION)
    rep.fill_position(1.0, 2.0, "sim001.sxm")
    assert rep.fields["x_nm"].text() == "1.000" and rep.fields["frame"].text() == "sim001.sxm"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_evidence_field_has_the_form_width_and_the_whole_list_is_shown_when_it_does_not_fit(fonts, open_window, lang):
    win = open_window(lang, scenario=P3)
    open_left(win, "report")
    shots.settle(win)
    rep = win.sidebar.report
    ev, full, button = rep.fields["evidence"], shots.widget(win, "report.evidence_full"), shots.widget(win, "report.viewed")
    assert ev.placeholderText() == one("rep_evidence_hint", lang) and not full.isVisible()
    label = rep.labels["evidence"]
    assert button.mapTo(rep, button.rect().topLeft()).y() < ev.mapTo(rep, ev.rect().topLeft()).y()      # the button is above
    assert ev.width() >= rep.width() - 2 * 12 - 4                                                      # the field is the form wide
    assert label.isVisible() and button.isVisible()
    ev.setText("sim001.sxm")
    assert not full.isVisible()                                          # it fits: nothing is repeated
    names = "sim001.sxm, 2.dat, sim004.sxm, sim005.sxm"
    ev.setText(names)
    shots.settle(win)
    assert ev.fontMetrics().horizontalAdvance(names) > ev.width() and full.isVisible() and full.text() == names
    assert full.mapTo(rep, full.rect().topLeft()).y() >= ev.mapTo(rep, ev.rect().bottomLeft()).y()   # under the field
    ev.setText("")
    assert not full.isVisible() and full.text() == ""
