"""The wheel on a spectrum's plot, and its Fit button, in the console window.

Written for the round after screen trial 23 (2026-11-25): the viewer's hint says the wheel zooms the image about the
pointer, the agent tried it on a spectrum, and the plot dropped the wheel. (The plot's own behaviour is in
``test_gui_plot_zoom.py``.)
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402

from stmbench.gui import shots  # noqa: E402
from stmbench.gui.labels import one  # noqa: E402
from tests.gui_helpers import app, fonts, open_window  # noqa: E402, F401
from tests.test_gui_app import make_trace  # noqa: E402


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_wheel_zooms_a_spectrum_about_the_pointer_and_fit_and_new_data_go_back(fonts, open_window, lang):
    win = open_window(lang)
    c = win.center
    c.show_spectrum(make_trace("move_0001.dat", n=60))
    shots.settle(win)
    plot = c.plot
    assert c.readout.full_text() == one("hint_plot", lang)
    assert ("wheel" in one("hint_plot", "en")) and ("滚轮" in one("hint_plot", "zh"))        # the hint says so
    fit = shots.widget(win, "view.fit_plot")
    assert fit.text() == one("zoom_fit", lang) and fit.isVisible()
    full = plot.x_range()
    where = plot.plot_rect().center().toPoint()
    shots.wheel(win, plot, where)
    assert plot.is_zoomed() and plot.x_range()[1] - plot.x_range()[0] == pytest.approx(0.5 * (full[1] - full[0]))
    shots.wheel(win, plot, where, -120)
    assert not plot.is_zoomed()
    shots.wheel(win, plot, where)
    shots.wheel(win, plot, where)
    shots.click(win, fit)
    assert not plot.is_zoomed() and plot.x_range() == pytest.approx(full)
    shots.wheel(win, plot, where)
    c.show_spectrum(make_trace("move_0001.dat", n=60))                                # new data
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)               # the old axis buttons are gone
    assert not plot.is_zoomed()
    shots.wheel(win, plot, where)
    shots.click(win, shots.widget(win, "view.spec_y.i"))                              # another column: also new data
    assert not plot.is_zoomed()
    assert fit.isVisible() and c.toolbars.rect().contains(fit.mapTo(c.toolbars, fit.rect().bottomRight()))      # it fits
    assert fit.fontMetrics().horizontalAdvance(fit.text()) <= fit.width()
