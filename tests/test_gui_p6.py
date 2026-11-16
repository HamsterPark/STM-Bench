"""Two things a screen agent doing P6 (a Kondo resonance a few meV wide in a +-50 mV sweep, reported in meV from
several cited spectra) needed from the console, found by looking at the 1280x720 pictures of a P6 episode.

1. The spectrum viewer read a +-50 mV sweep in volts (ticks -0.04 ... 0.04, a cursor that says ``V = -0.0365``), so a
   feature a few meV wide was read, and reported, through a factor of 1000. A narrow bias sweep now also offers its
   bias in mV and opens on it; a wide one is as it was.
2. The evidence field is one line a third of the form wide: with two spectra cited (26-character names) it shows the
   last 17 characters of the second, so the reader cannot see what it has cited. When the text does not fit the field,
   the files are listed below it, one a line.

The tests use P4's instrument: neither depends on the scenario.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui import center, shots  # noqa: E402
from tests.gui_helpers import app  # noqa: E402, F401
from tests.test_gui_app import fonts, open_right, open_window, run_button  # noqa: E402, F401
from tests.test_gui_report_fit import clipped  # noqa: E402


def sweep(win, lo: str, hi: str, points: str = "41") -> dict:
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": lo, "act.sts.v_end": hi, "act.sts.points": points})
    run_button(win, "run.sts")
    shots.settle(win)
    return win.center._spec


def test_with_millivolts_changes_only_a_narrow_bias_sweep():
    v = np.linspace(-0.05, 0.05, 41)
    narrow = {"name": "s.dat", "kind": "sts", "v": v, "i": v * 1e-9, "didv": (1.0 + v ** 2) * 1e-9}
    out = center.with_millivolts(narrow)
    assert np.array_equal(out["v_mv"], v) and "v_mv" not in narrow                     # the record itself is untouched
    assert center.axis_columns(out)[0] == ["v", "v_mv", "i", "didv"] and center.default_axes(out) == ("v_mv", "didv")
    wide = {**narrow, "v": np.linspace(-0.3, 0.3, 41)}
    assert center.with_millivolts(wide) is wide and center.default_axes(wide)[0] == "v"
    short = {**narrow, "v": v[:5], "i": v[:5], "didv": v[:5]}
    assert center.with_millivolts(short) is short
    trace = {"name": "m.dat", "kind": "move", "v": v, "dist": v, "t": v, "z_abs": v}
    assert center.with_millivolts(trace) is trace
    edge = {**narrow, "v": np.linspace(-0.125, 0.125, 41)}                              # exactly the limit: narrow
    assert "v_mv" in center.with_millivolts(edge)
    assert "v_mv" not in center.with_millivolts({**narrow, "v": np.linspace(-0.13, 0.13, 41)})


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_narrow_sweep_is_plotted_and_read_in_millivolts_and_a_wide_one_in_volts(open_window, lang):
    win = open_window(lang)
    rec = sweep(win, "-0.05", "0.05")
    plot = win.center.plot
    assert win.center.current_spectrum_axes() == ("v_mv", "didv")
    assert plot._x_label.endswith("V (mV)")
    assert np.allclose(plot._x, np.asarray(rec["v"]) * 1e3) and plot._x[0] == pytest.approx(-50.0)
    i = int(np.argmin(np.abs(plot._x - 12.5)))
    assert plot.point_position(i) is not None and "V (mV)" in [f[1] for f in plot._fields(i)][0]
    assert plot._x_decimals <= 1                                                         # 1.25 mV steps: tenths of a mV
    shots.widget(win, "view.spec_x.v")                                                  # the volts stay on offer
    run_button(win, "view.spec_x.v")
    assert win.center.current_spectrum_axes() == ("v", "didv") and plot._x_label.endswith("V")
    assert plot._x[-1] == pytest.approx(0.05)

    sweep(win, "-0.3", "0.3")
    assert win.center.current_spectrum_axes() == ("v", "didv")
    assert not win.center.spec_x.buttons()[1].objectName().endswith("v_mv")


def test_the_files_of_a_long_evidence_field_are_listed_under_it_and_nothing_is_added_while_it_fits(fonts, open_window):
    win = open_window("en")
    win.resize(1920, 1080)
    win.sidebar.show_tab("report")
    shots.settle(win)
    rep = win.sidebar.report
    field, listing = rep.fields["evidence"], rep.evidence_full
    QApplication.processEvents()
    assert not listing.isVisible()                                                       # empty: nothing there
    field.setText("Bias Spectroscopy00001.dat")
    QApplication.processEvents()
    assert not listing.isVisible() or listing.text() == field.text()                      # one name may or may not fit
    two = ["Bias Spectroscopy00001.dat", "Bias Spectroscopy00002.dat", "Bias Spectroscopy00003.dat"]
    field.setText(", ".join(two))
    QApplication.processEvents()
    assert listing.isVisible() and listing.text().split("\n") == two
    assert clipped(win) == [] and win.sidebar.report_scroll.horizontalScrollBar().maximum() == 0
    assert listing.mapTo(rep, listing.rect().bottomLeft()).y() <= rep.height()
    field.setText("sts:1, sts:2")                                                        # the short names fit again
    QApplication.processEvents()
    assert not listing.isVisible()
