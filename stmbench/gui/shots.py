"""Screenshots of the window, rendered offscreen: ``python -m stmbench.gui.shots --out <dir>``.

Why. The window is built for an agent that sees the screen downscaled to about 1280x720, so the only
honest review is to look at it that size. This script runs the window offscreen (never a visible
window) at the size it has on a 1920x1080 screen (full screen, as the sealed mode opens: the picture an agent
is shown; ``--maximized`` for the dev mode's window under GNOME's top bar), drives it with the same clicks and
keystrokes an agent's would be, and saves each state twice: as the full screen and as the 1280x720 picture a model
is shown. The states, for P4 seed 0: (a) just started, (b) after a 30 nm, 256 px scan with three
markers and a profile, (c) an STS spectrum, (d) the Report tab filled in, (e) a refusal, (f-h) the Data,
Log and Manual tabs, (i) a 1024 px scan running with two calls queued behind it.

The helpers (:func:`click`, :func:`type_into`, :func:`drag`) are shared with the tests: they send
events to the deepest widget under a window position, so a control that is hidden or covered fails
instead of being clicked behind the back of the layout.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"                  # never a visible window: set before Qt loads

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from . import style  # noqa: E402
from .app import MainWindow  # noqa: E402
from .session import GuiSession  # noqa: E402

SCREEN = (1920, 1080)
WINDOW = (1920, 1053)                                          # a maximized window under GNOME's top bar (dev mode)
FULLSCREEN = SCREEN                                            # the sealed mode's window is the screen
DOWNSCALED = (1280, 720)
TOP_BAR_PX = SCREEN[1] - WINDOW[1]


# ── driving the window the way a screen agent does ──
def hit(win: QWidget, w: QWidget, local: QPoint | None = None) -> tuple[QWidget, QPoint]:
    """The widget that a click at ``local`` (in ``w``'s coordinates; default its centre) reaches, and the
    position in that widget's own coordinates. Fails if it is not ``w`` or one of its children: a control
    that is hidden, clipped or covered cannot be clicked."""
    pos = w.mapTo(win, w.rect().center() if local is None else local)
    target = win.childAt(pos)
    node = target
    while node is not None and node is not w:
        node = node.parentWidget()
    if node is None:
        got = (target.objectName() or target.__class__.__name__) if target else "nothing"
        meant = w.objectName() or w.__class__.__name__
        raise AssertionError(f"a click at {pos.x()},{pos.y()} meant for {meant} reaches {got}")
    return target, target.mapFrom(win, pos)


def click(win: QWidget, w: QWidget, local: QPoint | None = None) -> None:
    target, at = hit(win, w, local)
    QTest.mouseClick(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)


def type_into(win: QWidget, w: QWidget, text: str) -> None:
    """Click a field (its content is selected) and type: the text replaces what was there."""
    click(win, w)
    QTest.keyClicks(w, text)


def drag(win: QWidget, w: QWidget, a: QPoint, b: QPoint, steps: int = 6) -> None:
    """Press at ``a``, move to ``b`` and release there (positions in ``w``'s own coordinates)."""
    target, la = hit(win, w, a)
    mod = Qt.KeyboardModifier.NoModifier
    QTest.mousePress(target, Qt.MouseButton.LeftButton, mod, la)
    for i in range(1, steps + 1):
        QTest.mouseMove(target, target.mapFrom(win, w.mapTo(win, a + (b - a) * (i / steps))))
    QTest.mouseRelease(target, Qt.MouseButton.LeftButton, mod, target.mapFrom(win, w.mapTo(win, b)))


def widget(win: QWidget, name: str) -> QWidget:
    w = win.findChild(QWidget, name)
    if w is None:
        raise KeyError(name)
    return w


def fill(win: QWidget, values: dict[str, str]) -> None:
    for name, text in values.items():
        type_into(win, widget(win, name), text)


def settle(win: MainWindow) -> None:
    if not win.wait_idle(300):
        raise TimeoutError("the instrument did not finish")


# ── pictures ──
def screen_image(win: QWidget) -> QImage:
    """The window on a 1920x1080 screen, under a top bar, as a screenshot of it would be."""
    shot = win.grab().toImage()
    img = QImage(SCREEN[0], SCREEN[1], QImage.Format.Format_RGB32)
    img.fill(QColor("#2e3436"))
    p = QPainter(img)
    p.drawImage(0, max(0, SCREEN[1] - shot.height()), shot)           # a full-screen window is the whole picture
    p.end()
    return img


def save(win: QWidget, out: Path, name: str) -> tuple[Path, Path]:
    img = screen_image(win)
    full, small = out / f"{name}.png", out / f"{name}_1280x720.png"
    img.save(str(full))
    img.scaled(DOWNSCALED[0], DOWNSCALED[1], Qt.AspectRatioMode.IgnoreAspectRatio,
               Qt.TransformationMode.SmoothTransformation).save(str(small))
    return full, small


def brightest_spots(array: np.ndarray, n: int = 3, apart: int = 24, edge: int = 20) -> list[tuple[int, int]]:
    """Pixel positions (col, row) of the ``n`` most point-like bumps of a topograph, away from steps:
    where a reader would put markers. Only for choosing where this script clicks."""
    from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
    a = array - np.median(array, axis=1, keepdims=True)
    bump = gaussian_filter(a, 1.5) - gaussian_filter(a, 5.0)
    steep = gaussian_gradient_magnitude(a, 3.0)
    bump = np.where(steep < np.percentile(steep, 60), bump, -np.inf)
    ny, nx = a.shape
    out: list[tuple[int, int]] = []
    for k in np.argsort(bump, axis=None)[::-1]:
        r, c = divmod(int(k), nx)
        if not np.isfinite(bump[r, c]):
            break
        inside = edge <= c < nx - edge and edge <= r < ny - edge
        if inside and all((c - c2) ** 2 + (r - r2) ** 2 >= apart ** 2 for c2, r2 in out):
            out.append((c, r))
            if len(out) == n:
                break
    return out


# ── the states ──
def render_states(out: Path, lang: str = "zh", scale: float = 1.0, scenario: str = "P4_atom_positioning_cu111",
                  seed: int = 0, work: Path | None = None, fullscreen: bool = True) -> list[Path]:
    app = QApplication.instance() or QApplication(["stmbench-shots"])
    style.apply(app, scale)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="stmbench_shots_"))
    sess = GuiSession(scenario, seed, run_dir=work / "run", export_dir=work / "run" / "export", lang=lang)
    win = MainWindow(sess, lang=lang)
    win.resize(*(FULLSCREEN if fullscreen else WINDOW))
    win.show()
    settle(win)
    saved: list[Path] = []

    def snap(name: str) -> None:
        app.processEvents()
        saved.extend(save(win, out, name))

    snap(f"a_started_{lang}")

    # (b) a 30 nm, 256 px scan, three markers and a profile
    fill(win, {"act.scan.center_x_nm": "0", "act.scan.center_y_nm": "0", "act.scan.size_nm": "30",
               "act.scan.pixels": "256"})
    click(win, widget(win, "run.scan"))
    settle(win)
    fv = win.center.frame_view
    name = win.viewed[1]
    spots = brightest_spots(win.cached_frame(name)["array"])
    for c, r in spots:                                           # markers: the Marker tool is the default
        pt = fv.pixel_to_widget(c, r)
        click(win, fv, pt.toPoint())
    click(win, widget(win, "view.tool.profile"))
    if len(spots) >= 2:
        a, b = (fv.pixel_to_widget(*spots[0]).toPoint(), fv.pixel_to_widget(*spots[1]).toPoint())
        drag(win, fv, a, b)
    click(win, widget(win, "view.tool.marker"))
    snap(f"b_scan_markers_profile_{lang}")

    # (e) a refusal
    fill(win, {"act.move_tip.x_nm": "1000000", "act.move_tip.y_nm": "0"})
    click(win, widget(win, "run.move_tip"))
    settle(win)
    snap(f"e_refusal_{lang}")

    # (c) an STS spectrum, from the Spectroscopy tab
    click(win, win.right.tabBar(), win.right.tabBar().tabRect(1).center())
    fill(win, {"act.sts.v_start": "-0.6", "act.sts.v_end": "0.6", "act.sts.points": "200"})
    click(win, widget(win, "run.sts"))
    settle(win)
    snap(f"c_spectrum_{lang}")
    click(win, win.right.tabBar(), win.right.tabBar().tabRect(0).center())

    # (d) the Report tab, filled in
    bar = win.sidebar.tabBar()
    click(win, bar, bar.tabRect(win.sidebar.KEYS.index("report")).center())
    click(win, widget(win, "report.claim.target_site"))
    fill(win, {"report.value": "1.234", "report.x_nm": "1.234", "report.y_nm": "-0.512", "report.frame": name})
    click(win, widget(win, "report.viewed"))
    snap(f"d_report_{lang}")

    # the other left-hand tabs, after the calls above
    for key, label in (("data", "f_data"), ("log", "g_log"), ("manual", "h_manual")):
        click(win, bar, bar.tabRect(win.sidebar.KEYS.index(key)).center())
        snap(f"{label}_{lang}")

    # (i) a scan that takes a second or more, and two calls clicked behind it: the strip says what runs, the
    # viewer says a scan is running, the log shows the two as queued, the Cancel queued button is there
    click(win, bar, bar.tabRect(win.sidebar.KEYS.index("log")).center())
    fill(win, {"act.scan.center_x_nm": "0", "act.scan.center_y_nm": "0", "act.scan.size_nm": "40",
               "act.scan.pixels": "1024", "act.set_bias.bias_v": "0.1", "act.set_setpoint.current_pa": "100"})
    for name in ("run.scan", "run.set_bias", "run.set_setpoint"):
        click(win, widget(win, name))
    app.processEvents()
    snap(f"i_busy_{lang}")
    settle(win)
    win.shutdown(finish=False)
    win.deleteLater()
    return saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stmbench.gui.shots", description="Render the desktop console offscreen in "
                                 "scripted states (P4, seed 0) and save each as a 1920x1080 screen and as the "
                                 "1280x720 picture a model is shown.")
    ap.add_argument("--out", required=True, help="directory for the PNGs")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"])
    ap.add_argument("--ui-scale", type=float, default=1.0)
    ap.add_argument("--maximized", action="store_true",
                    help="the dev mode's window (1920x1053 under GNOME's top bar) instead of the full-screen one")
    ap.add_argument("--work", default=None, help="a new directory for the run and its export (default: a temporary one)")
    a = ap.parse_args(argv)
    for p in render_states(Path(a.out), a.lang, a.ui_scale, work=Path(a.work) if a.work else None,
                           fullscreen=not a.maximized):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
