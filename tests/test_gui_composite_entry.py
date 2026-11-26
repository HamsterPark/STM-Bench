"""One entry for a move with during-move values (screen trial 24)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from stmbench.gui import shots  # noqa: E402
from tests.gui_helpers import app, fonts, open_window, run_button  # noqa: E402, F401

P1, P4 = "P1_barth1990_au111", "P4_atom_positioning_cu111"

def starred(rep, key: str) -> bool:
    return rep.labels[key].text().endswith("*</span>")



def drag(win, **extra):
    fields = {"act.move_tip.x_nm": "2", "act.move_tip.y_nm": "1", "during.set_current_range.full_scale_na": "100",
              "during.set_setpoint.current_pa": "150", **extra}
    shots.fill(win, fields)
    run_button(win, "run.move_tip")
    shots.settle(win)


def test_a_move_with_during_move_values_is_one_result_the_moves_own_reply_then_what_was_put_back(open_window):
    win = open_window("en", seed=3)
    before = win.session.status()
    drag(win)
    entries = win.recent.entries()
    assert len(entries) == 1 and entries[0][0] == "ok", entries                 # not one row per call
    text = entries[0][1]
    assert text.startswith("move_tip") and ".dat" in text                      # its own reply first, the trace name visible
    assert " · set back: " in text and "current range" in text and "setpoint" in text
    assert str(int(before["current_full_scale_na"])) in text.split("set back:")[1]
    assert win.status.result.full_text().startswith("✓ move_tip") and "set back" in win.status.result.full_text()
    assert [e["label"] for e in win.sidebar.log._entries] == ["set_current_range", "set_setpoint", "move_tip", "set_setpoint",
                                                              "set_current_range"]          # the Log keeps every call


def test_the_set_back_words_in_chinese(open_window):
    win = open_window("zh", seed=3)
    drag(win)
    t = win.recent.entries()[0][1]
    assert "已改回: " in t and "量程 10 nA" in t and "设定点 200 pA" in t


def test_a_composite_move_with_a_refused_call_is_a_refusal_and_keeps_the_hold(open_window):
    win = open_window("en", seed=3)
    drag(win, **{"act.move_tip.x_nm": "100000"})                               # the instrument refuses the move
    entries = win.recent.entries()
    assert entries[0][0] == "bad" and "piezo range" in entries[0][1] and entries[0][1].startswith("refused")
    assert all(t != "ok" for t, _ in entries)                                  # nothing is told as done
    assert win.worker.held and win._refusal.call.startswith("move_tip")


def test_a_plain_move_tip_and_reposition_are_one_entry_as_before(open_window):
    win = open_window("en", seed=3)
    shots.fill(win, {"act.move_tip.x_nm": "2", "act.move_tip.y_nm": "1"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    run_button(win, "tip.reposition")
    shots.settle(win)
    entries = win.recent.entries()
    assert len(entries) == 2 and all(t == "ok" and x.startswith("move_tip") and "set back" not in x for t, x in entries)


