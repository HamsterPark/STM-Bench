"""Four usage gaps found by a sighted check of P3 through the screen interface (2026-11-16, p3check1): the console
worked, but nothing on the screen said how. Each is about how the software is read, none about the experiment.

1. The Report form greys out x_nm, y_nm and frame for a claim the rule card gives no position.
2. The evidence field shows a list of files whole (it wraps) instead of the end of one narrow line.
3. A spectrum's sweep and the bias and setpoint it started from are on the viewer's title and the Data page.
4. The readout line's hint says that the wheel zooms about the pointer.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from stmbench.gui import fmt, shots  # noqa: E402
from stmbench.gui.labels import one, tr  # noqa: E402
from tests.gui_helpers import app, run_button  # noqa: E402, F401
from tests.test_gui_app import fonts, open_left, open_right, open_window  # noqa: E402, F401

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
    assert ev.placeholderText() == one("rep_evidence_hint", lang) and full.text() == ""
    label = rep.labels["evidence"]
    assert button.mapTo(rep, button.rect().topLeft()).y() < ev.mapTo(rep, ev.rect().topLeft()).y()      # the button is above
    assert ev.width() >= rep.width() - 2 * 12 - 4                                                      # the field is the form wide
    assert label.isVisible() and button.isVisible()
    ev.setText("sim001.sxm")
    assert full.text() == ""                                             # it fits: nothing is repeated (its lines stay reserved)
    names = "sim001.sxm, 2.dat, sim004.sxm, sim005.sxm"
    ev.setText(names)
    shots.settle(win)
    assert ev.fontMetrics().horizontalAdvance(names) > ev.width() and full.isVisible()   # its place is always there (screen trial 23)
    assert all(n in full.text().replace(chr(10), " ") for n in names.split(", ")) or "more" in full.text()
    assert full.mapTo(rep, full.rect().topLeft()).y() >= ev.mapTo(rep, ev.rect().bottomLeft()).y()   # under the field
    ev.setText("")
    assert full.text() == ""


def test_a_spectrum_says_its_sweep_and_where_the_junction_stood_when_it_began(open_window):
    win = open_window("en")
    start = win._status
    assert start is not None
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": "-0.2", "act.sts.v_end": "0.15", "act.sts.points": "16"})
    run_button(win, "run.sts")
    shots.settle(win)
    name = win.viewed[1]
    sweep = "sweep -0.200 to 0.150 V"
    junction = f"from {fmt.volts(start['bias_v'])} · {fmt.picoamps(start['setpoint_pa'])}"
    title = win.center.info.text()
    assert sweep in title and junction in title and f"{name}" in title and title.endswith("16 points")
    assert len(title.splitlines()) == 2
    open_left(win, "data")
    item = win.sidebar.data.spectra.item(0).text()
    assert sweep in item and junction in item and item.splitlines()[0] == name
    # a second sweep from another bias: the junction is read before each call, not once
    open_right(win, "operate")
    shots.fill(win, {"act.set_bias.bias_v": "0.3"})
    run_button(win, "run.set_bias")
    shots.settle(win)
    open_right(win, "spectroscopy")
    run_button(win, "run.sts")
    shots.settle(win)
    assert f"from {fmt.volts(0.3)}" in win.center.info.text()
    assert junction in win.sidebar.data.spectra.item(1).text()           # the first one, now second in the list, unchanged
    win.request_spectrum(name)                                           # shown again by a click: still told
    shots.settle(win)
    assert sweep in win.center.info.text() and junction in win.center.info.text()


def test_a_z_spectrum_says_its_sweep_too_and_a_trace_is_left_as_it_is(open_window):
    win = open_window("en")
    assert fmt.sweep_text({"sweep_nm": 0.5}) == "sweep 0.5 nm" and fmt.sweep_text(None) == ""
    assert fmt.junction_text({"bias_v": 0.05}) == "" and fmt.junction_text({}) == ""
    c = win.center
    trace = {"name": "move_0001.dat", "kind": "move", "t_s": 3.0, "tip_nm": [1.0, 2.0], "bias_v": 0.1,
             "setpoint_pa": 50.0, "feedback": True, "dist": [0.0, 1.0], "z": [0.0, 1.0], "i": [1.0, 1.0], "t": [0.0, 1.0],
             "from_nm": [0.0, 0.0], "to_nm": [1.0, 2.0], "speed_nm_s": 1.0}
    plain = c._spectrum_info(trace, 2)
    assert c._spectrum_info(trace, 2, {"v_start": 0.0, "v_end": 1.0, "bias_v": 9.0, "setpoint_pa": 9.0}) == plain


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_hint_says_the_wheel_zooms_about_the_pointer(lang):
    for key in ("hint_image", "hint_image_profile", "hint_image_pan"):
        assert ("scroll" in one(key, lang) or "wheel" in one(key, lang) or "滚轮" in one(key, lang)), key
    assert "zoom about the pointer" in one("hint_image", "en") and "滚轮以指针处为中心缩放" in one("hint_image", "zh")
