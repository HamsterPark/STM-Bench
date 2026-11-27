"""The window and ``--data-dir`` (stmbench.gui.datacopy): the copies appear after the actions that saved something, nothing is
written without the option, and the Manual says it only with the option and still fits two screens."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from stmbench.gui.session import GuiSession  # noqa: E402
from tests.gui_helpers import app  # noqa: E402, F401
from tests.test_gui_datacopy import P3, parse  # noqa: E402


def make_window(tmp_path, data_dir, lang="en", scenario=P3):
    from stmbench.gui import shots
    from stmbench.gui.app import MainWindow
    sess = GuiSession(scenario, 3, run_dir=tmp_path / "run", export_dir=tmp_path / "run" / "export", lang=lang)
    win = MainWindow(sess, lang=lang, data_dir=data_dir)
    win.resize(*shots.WINDOW)
    win.show()
    shots.settle(win)
    return win


def test_the_window_writes_the_copies_after_the_actions_and_writes_nothing_without_the_option(app, tmp_path):
    from stmbench.gui import shots
    from tests.test_gui_app import SMALL, run_button
    data = tmp_path / "copies"
    win = make_window(tmp_path / "a", data)
    try:
        assert not data.exists() or list(data.iterdir()) == []             # nothing before the first save
        shots.fill(win, SMALL)
        run_button(win, "run.scan")
        shots.settle(win)
        win._copier.wait()
        names = sorted(p.name for p in data.iterdir())
        assert names == ["sim001_current_backward.csv", "sim001_current_forward.csv", "sim001_z_backward.csv", "sim001_z_forward.csv"]
        head, rest = parse(data / "sim001_z_forward.csv")
        assert head["source"] == "sim001.sxm" and len(rest) == 32          # a frame whole, once its scan reply arrived
    finally:
        win.shutdown(finish=False)
        win.deleteLater()
    win = make_window(tmp_path / "b", None)
    try:
        assert win._copier is None
        shots.fill(win, SMALL)
        run_button(win, "run.scan")
        shots.settle(win)
        assert list((tmp_path / "b").rglob("*.csv")) == []
    finally:
        win.shutdown(finish=False)
        win.deleteLater()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_manual_says_it_only_with_the_option_and_still_fits_two_screens(app, tmp_path, lang):
    from PySide6.QtWidgets import QApplication
    from stmbench.gui import shots
    from stmbench.gui.labels import DATA_NOTE, with_console_notes
    from tests.test_gui_app import manual_lines, open_left, run_button
    assert str(Path("/x")) not in with_console_notes("m", lang)
    win = make_window(tmp_path, "/home/user/stm_data", lang)
    try:
        text = shots.widget(win, "manual.text").toPlainText()
        assert DATA_NOTE[0 if lang == "en" else 1].format(dir="/home/user/stm_data") in text
        open_left(win, "manual")
        run_button(win, "manual.wide")
        for _ in range(3):
            QApplication.processEvents()
        total, page = manual_lines(win)
        assert total <= 2 * page, (total, page)
    finally:
        win.shutdown(finish=False)
        win.deleteLater()
    plain = make_window(tmp_path / "p", None, lang)
    try:
        assert "CSV" not in shots.widget(plain, "manual.text").toPlainText()
    finally:
        plain.shutdown(finish=False)
        plain.deleteLater()
