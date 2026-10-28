"""Fixtures and helpers for the tests that drive the console window as a screen agent does (clicks, wheel notches
and keystrokes at widget positions) and read the viewer: ``test_gui_markers.py``, ``test_gui_strip.py``.

(``test_gui_app.py`` has its own copies of ``app`` and ``open_window``; these are the viewer tests' own, so that the
two files do not depend on each other.)
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"                 # never a visible window

from PySide6.QtCore import QCoreApplication, QEvent, QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui import shots, style  # noqa: E402
from stmbench.gui.app import MainWindow  # noqa: E402
from stmbench.gui.jobs import Outcome  # noqa: E402
from stmbench.gui.session import GuiSession  # noqa: E402
from stmbench.gui.worker import Job  # noqa: E402

P4 = "P4_atom_positioning_cu111"
SMALL = {"act.scan.center_x_nm": "0", "act.scan.center_y_nm": "0", "act.scan.size_nm": "8", "act.scan.pixels": "32"}


@pytest.fixture(scope="module")
def app():
    """The application with the console's look, taken off again afterwards (the viewer widgets' tests pin pixel
    sizes that depend on the default font)."""
    a = QApplication.instance() or QApplication(["test"])
    before = (a.font(), a.palette(), a.styleSheet(), a.style().objectName())
    style.apply(a, 1.0)
    yield a
    style.SCALE = 1.0
    a.setStyleSheet(before[2])
    a.setStyle(before[3])
    a.setPalette(before[1])
    a.setFont(before[0])


@pytest.fixture(params=["default", "dejavu"])
def fonts(request, app):
    """The window's text in the fonts it was tuned with, and again in DejaVu Sans / Sans Mono (what a Linux desktop
    without anything else falls back to: wider, another line height). Requested before ``open_window``, so that the
    window is made in these fonts; skipped where matplotlib's DejaVu files are not there."""
    if request.param == "default":
        yield "default"
        return
    mpl = pytest.importorskip("matplotlib")
    from pathlib import Path

    from PySide6.QtGui import QFontDatabase
    folder = Path(mpl.get_data_path()) / "fonts" / "ttf"
    names = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")
    if not all((folder / n).is_file() for n in names):
        pytest.skip("no DejaVu fonts to test with")
    ids = [QFontDatabase.addApplicationFont(str(folder / n)) for n in names]
    sans, mono = list(style.SANS), list(style.MONO)
    style.SANS[:] = ["DejaVu Sans", *sans]
    style.MONO[:] = ["DejaVu Sans Mono", *mono]
    style.apply(app, 1.0)
    try:
        yield "dejavu"
    finally:
        style.SANS[:] = sans
        style.MONO[:] = mono
        style.apply(app, 1.0)
        for i in ids:
            QFontDatabase.removeApplicationFont(i)


@pytest.fixture
def open_window(app, tmp_path):
    """A factory of windows on a full-screen 1920x1080 window, shut down afterwards (their threads must end)."""
    made: list[MainWindow] = []

    def make(lang: str = "en", seed: int = 0, name: str = "run", scenario: str = P4, height: int = 1080) -> MainWindow:
        sess = GuiSession(scenario, seed, run_dir=tmp_path / name, export_dir=tmp_path / name / "export", lang=lang)
        win = MainWindow(sess, lang=lang)
        win.worker.guard_s = 0.0
        win.resize(1920, height)
        win.show()
        made.append(win)
        shots.settle(win)
        return win

    yield make
    for w in made:
        w.shutdown(finish=False)
        w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def run_button(win: MainWindow, name: str) -> None:
    shots.click(win, shots.widget(win, name))


def scan(win: MainWindow, cx: float = 0.0, cy: float = 0.0, size: float = 8.0, pixels: int = 32) -> str:
    """Click Scan with these fields and wait for the frame; its name."""
    shots.fill(win, {"act.scan.center_x_nm": f"{cx}", "act.scan.center_y_nm": f"{cy}", "act.scan.size_nm": f"{size}",
                     "act.scan.pixels": f"{pixels}"})
    run_button(win, "run.scan")
    shots.settle(win)
    return win.viewed[1]


def pixel_point(win: MainWindow, col: float, row: float, dx: float = 0.0, dy: float = 0.0) -> QPoint:
    """The window position (in the frame view's coordinates) of pixel ``(col, row)``'s centre, shifted by a
    fraction ``(dx, dy)`` of a pixel."""
    fv = win.center.frame_view
    return fv.pixel_to_widget(col + dx, row + dy).toPoint()


def mark(win: MainWindow, col: float, row: float, dx: float = 0.0, dy: float = 0.0) -> None:
    """A click of the Marker tool on pixel ``(col, row)``."""
    shots.click(win, win.center.frame_view, pixel_point(win, col, row, dx, dy))


def hold_scan(win: MainWindow) -> "Hold":
    return Hold(win)


class Hold:
    """A scan that is "running" for as long as the test wants: a job that holds the worker until it is opened."""

    def __init__(self, win: MainWindow):
        self.win = win
        self.event = threading.Event()
        args = {"center_x_nm": 0, "center_y_nm": 0, "size_nm": 40, "pixels": 1024}

        def hold(_session):
            self.event.wait(60)
            return Outcome("act", "scan", dict(args), ok=True, reply={"held": True})

        win._submit(Job("act", "scan", dict(args), hold))
        end = time.monotonic() + 10
        while win.status.result.property("tone") != "busy" and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()

    def open(self) -> None:
        self.event.set()
        shots.settle(self.win)


def noise_free_atom_frame(n: int = 48, atoms=((0.3, 0.4), (0.7, 0.6))) -> np.ndarray:
    """A z array in metres: a flat terrace and a few Gaussian atoms (for tests that need something to look at)."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    z = np.zeros((n, n))
    for fx, fy in atoms:
        z += 60e-12 * np.exp(-((xx - fx * n) ** 2 + (yy - fy * n) ** 2) / (2 * (n / 24) ** 2))
    return z


__all__ = ["P4", "SMALL", "app", "fonts", "open_window", "run_button", "scan", "pixel_point", "mark", "Hold",
           "hold_scan", "noise_free_atom_frame"]
