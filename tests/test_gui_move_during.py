"""The Tip group's optional "during the move" values (stmbench.gui.layout.During, jobs.run_move_during).

Written for POST_DELIVERY items 3 and 5 of the P3 delivery: in the blind screen trials every atom drag cost five or six
form operations (raise the preamp range, raise the setpoint, sometimes change the bias, move, then put them back);
trial 13 spent 168 actions of which move_tip 55, set_setpoint 37, set_current_range 34, and trial 21 212 of which
move_tip 65, set_setpoint 48, set_current_range 42. The console runs those calls for the operator, in a fixed order
and with no advice on values; this file pins what it runs, that it is the same as clicking them one by one, and what
it does when a call is refused.

The job-level cases use a stand-in session (no Qt, no instrument); the window-level ones run the real instrument.
"""
from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from stmbench.gui import jobs, layout, shots  # noqa: E402
from stmbench.instrument import ACTIONS, ActionError  # noqa: E402
from stmbench.instrument.core import validate  # noqa: E402
from tests.gui_helpers import app, fonts, open_window, run_button  # noqa: E402, F401

MOVE = {"x_nm": 2.0, "y_nm": 1.0, "speed_nm_s": 50.0}
STATE = {"bias_v": 0.1, "setpoint_pa": 20.0, "current_full_scale_na": 1.0}
ORDER = ["set_current_range", "set_bias", "set_setpoint"]


class Stand:
    """The session as the job sees it: records the calls, keeps a state like the instrument's, refuses on demand."""

    def __init__(self, refuse: tuple[str, int] | None = None):
        self.state = dict(STATE)
        self.calls: list[tuple[str, dict]] = []
        self.reads = 0
        self.refuse = refuse                              # (action, nth call of it that is refused), 1-based
        self.seen: dict[str, int] = {}

    def status(self):
        self.reads += 1
        return dict(self.state, instrument_time_s=0.0)

    def act(self, name, args=None):
        args = dict(args or {})
        self.seen[name] = self.seen.get(name, 0) + 1
        if self.refuse == (name, self.seen[name]):
            raise ActionError(f"{name}: refused by the test")
        self.calls.append((name, args))
        key = {"set_current_range": "current_full_scale_na", "set_bias": "bias_v", "set_setpoint": "setpoint_pa"}.get(name)
        if key:
            self.state[key] = next(iter(args.values()))
        return {"instrument_time_s": 0.0}


DURING = {"set_current_range": {"full_scale_na": 10.0}, "set_bias": {"bias_v": 0.5}, "set_setpoint": {"current_pa": 300.0}}


def test_the_calls_are_the_manual_sequence_in_the_manual_order_and_each_value_goes_back_to_what_was_read():
    s = Stand()
    out = jobs.run_move_during(s, MOVE, DURING)
    assert out.ok and [o.label for o in out.steps] == ORDER + ["move_tip"] + ["set_setpoint", "set_bias", "set_current_range"]
    assert s.calls == [("set_current_range", {"full_scale_na": 10.0}), ("set_bias", {"bias_v": 0.5}),
                       ("set_setpoint", {"current_pa": 300.0}), ("move_tip", MOVE),
                       ("set_setpoint", {"current_pa": 20.0}), ("set_bias", {"bias_v": 0.1}),
                       ("set_current_range", {"full_scale_na": 1.0})]
    assert s.state == STATE


def test_only_what_is_set_runs_and_only_what_it_changed_is_put_back():
    s = Stand()
    out = jobs.run_move_during(s, MOVE, {"set_setpoint": {"current_pa": 300.0}, "set_bias": {"bias_v": 0.1}})
    assert out.ok
    assert [n for n, _ in s.calls] == ["set_bias", "set_setpoint", "move_tip", "set_setpoint"]   # the bias was already 0.1
    assert s.state == STATE


def test_a_value_that_is_refused_at_the_check_runs_nothing_and_says_which():
    s = Stand()
    out = jobs.run_move_during(s, MOVE, {"set_current_range": {"full_scale_na": 10.0}, "set_bias": {"bias_v": 99.0}})
    assert not out.ok and out.refused and s.calls == [] and s.reads == 0
    assert [(o.label, o.ok) for o in out.steps] == [("set_bias", False)] and "outside [-10.0, 10.0]" in out.text
    out = jobs.run_move_during(s, MOVE, {"set_setpoint": {"current_pa": "abc"}})
    assert not out.ok and s.calls == [] and "must be a number" in out.text
    out = jobs.run_move_during(s, {**MOVE, "speed_nm_s": 0.0}, {"set_bias": {"bias_v": 0.5}})    # the move's own field
    assert not out.ok and s.calls == [] and out.steps[0].label == "move_tip"
    out = jobs.run_move_during(s, MOVE, {"set_current_range": {"full_scale_na": 7.0}})            # not one of the choices
    assert not out.ok and s.calls == [] and "must be one of" in out.text


@pytest.mark.parametrize("refused", [("set_bias", 1), ("set_setpoint", 1), ("move_tip", 1)])
def test_a_call_refused_at_run_time_puts_back_what_was_applied_in_reverse_order(refused):
    s = Stand(refuse=refused)
    out = jobs.run_move_during(s, MOVE, DURING)
    assert not out.ok and out.refused and s.state == STATE
    ran = [n for n, _ in s.calls]
    applied = ORDER[:ORDER.index(refused[0])] if refused[0] in ORDER else ORDER
    assert ran[len(applied):] == list(reversed(applied))
    assert [(o.label, o.ok) for o in out.steps][:len(applied)] == [(n, True) for n in applied]
    assert out.steps[len(applied)].label == refused[0] and not out.steps[len(applied)].ok
    assert out.text.startswith(refused[0])


def test_a_putting_back_that_is_refused_stops_there_and_is_the_refusal_that_is_shown():
    s = Stand(refuse=("set_bias", 2))                    # the second set_bias is the putting back
    out = jobs.run_move_during(s, MOVE, DURING)
    assert not out.ok and out.steps[-1].label == "set_bias" and not out.steps[-1].ok
    assert out.text.startswith("set_bias") and [o.label for o in out.steps][-2:] == ["set_setpoint", "set_bias"]
    assert "set_current_range" not in [n for n, _ in s.calls[4:]]          # the last putting back did not run
    assert s.state["current_full_scale_na"] == 10.0 and s.state["setpoint_pa"] == 20.0


def test_the_state_is_read_when_the_job_runs_not_when_it_was_built():
    s = Stand()
    job = lambda: jobs.run_move_during(s, MOVE, {"set_bias": {"bias_v": 0.5}})      # noqa: E731
    s.state["bias_v"] = -0.3                                                         # an earlier click changed it meanwhile
    job()
    assert s.calls[-1] == ("set_bias", {"bias_v": -0.3})


# ── the arrangement ──
def test_the_during_cells_borrow_a_setters_parameter_and_do_not_count_as_controls_of_it():
    cells = layout.during_cells()
    assert [(c.action, c.param) for c in cells] == [("set_current_range", "full_scale_na"), ("set_setpoint", "current_pa")]
    assert [c.en for c in cells] == ["move range", "move setpoint"]
    assert [c.zh for c in cells] == ["移动时量程", "移动时设定点"]
    assert [c.unit for c in cells] == ["nA", "pA"]
    tip = next(g for g in layout.OPERATE if g.key == "tip")
    assert tip.lines[1] == tuple(cells)      # directly under the move_tip line
    assert tip.lines[0][-1].action == "move_tip"
    for c in cells:
        assert any(p.name == c.param for p in ACTIONS[c.action].params)
    assert layout.group_of_action()["set_bias"] == "junction"


# ── the window ──
P4 = "P4_atom_positioning_cu111"


def acts(win):
    """The instrument's record of what ran (the reads are left out: they spend nothing and draw nothing)."""
    return [(r["name"], r.get("params", r.get("args"))) for r in win.session._inst.log if r["kind"] in ("act", "refused")]


def fill_move(win, **during):
    shots.fill(win, {"act.move_tip.x_nm": "2", "act.move_tip.y_nm": "1", "act.move_tip.speed_nm_s": "50",
                     **{f"during.{k}": v for k, v in during.items()}})


def test_all_three_on_keep_is_exactly_the_move_tip_of_today(open_window):
    win = open_window("en", seed=3)
    fill_move(win)
    assert win.operate.collect_during() == {}
    run_button(win, "run.move_tip")
    shots.settle(win)
    assert [n for n, _ in acts(win)] == ["move_tip"]
    assert [e["label"] for e in win.sidebar.log._entries] == ["move_tip"]


def test_the_composite_and_the_same_calls_clicked_one_by_one_end_alike(open_window):
    a = open_window("en", seed=3, name="a")
    b = open_window("en", seed=3, name="b")
    before = a.session.status()
    # a: the composite
    fill_move(a, **{"set_setpoint.current_pa": "150", "set_current_range.full_scale_na": "100"})
    run_button(a, "run.move_tip")
    shots.settle(a)
    # b: the same calls, clicked
    shots.fill(b, {"act.set_setpoint.current_pa": "150"})
    run_button(b, "run.set_current_range.100")
    run_button(b, "run.set_setpoint")
    fill_move(b)
    run_button(b, "run.move_tip")
    shots.fill(b, {"act.set_setpoint.current_pa": str(before["setpoint_pa"])})
    run_button(b, "run.set_setpoint")
    run_button(b, f"run.set_current_range.{before['current_full_scale_na']:g}")
    shots.settle(b)
    assert [n for n, _ in acts(a)] == ["set_current_range", "set_setpoint", "move_tip", "set_setpoint", "set_current_range"]
    assert acts(a) == acts(b)                                           # the same names and arguments
    ia, ib = a.session._inst, b.session._inst
    assert ia.used_s == ib.used_s                                       # the same virtual clock
    assert ia.digest() == ib.digest()                                   # the same world state and measurements
    rec = [json.loads(line) for line in (a.session._export.dir / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["name"] for r in rec if r["kind"] == "act"][2] == "move_tip"      # in the export as if clicked
    assert [e["label"] for e in a.sidebar.log._entries] == ["set_current_range", "set_setpoint", "move_tip",
                                                            "set_setpoint", "set_current_range"]


def test_a_bad_during_value_runs_nothing_and_holds_like_a_bad_field(open_window):
    win = open_window("en", seed=3)
    n = len(acts(win))
    fill_move(win, **{"set_setpoint.current_pa": "1e9", "set_current_range.full_scale_na": "100"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    assert len(acts(win)) == n                                          # nothing ran, not even the range
    assert win.worker.held and win._refusal.call.startswith("set_setpoint") and "outside" in win._refusal.words
    assert [e["label"] for e in win.sidebar.log._entries] == ["set_setpoint"]


def test_a_move_the_instrument_refuses_puts_the_values_back_and_lists_every_call(open_window):
    win = open_window("en", seed=3)
    before = win.session.status()
    shots.fill(win, {"act.move_tip.x_nm": "100000", "act.move_tip.y_nm": "0", "during.set_setpoint.current_pa": "150",
                     "during.set_current_range.full_scale_na": "100"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    assert [n for n, _ in acts(win)] == ["set_current_range", "set_setpoint", "move_tip", "set_setpoint", "set_current_range"]
    after = win.session.status()
    assert (after["setpoint_pa"], after["current_full_scale_na"]) == (before["setpoint_pa"], before["current_full_scale_na"])
    assert win._refusal.call.startswith("move_tip") and "piezo range" in win._refusal.words
    assert [e["label"] for e in win.sidebar.log._entries] == ["set_current_range", "set_setpoint", "move_tip", "set_setpoint",
                                                              "set_current_range"]
    assert win.worker.held


def test_the_range_field_runs_nothing_when_typed_in_and_goes_back_to_keep_when_blank(open_window):
    win = open_window("en", seed=3)
    n = len(acts(win))
    shots.fill(win, {"during.set_current_range.full_scale_na": "100"})
    assert win.operate.collect_during() == {"set_current_range": {"full_scale_na": 100}}
    shots.fill(win, {"during.set_current_range.full_scale_na": " "})
    assert win.operate.collect_during() == {}
    shots.settle(win)
    assert len(acts(win)) == n and win.sidebar.log._entries == []


def test_enter_in_a_during_field_moves_the_tip_like_enter_in_a_move_field(open_window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    win = open_window("en", seed=3)
    fill_move(win, **{"set_setpoint.current_pa": "100"})
    QTest.keyClick(shots.widget(win, "during.set_setpoint.current_pa"), Qt.Key.Key_Return)
    shots.settle(win)
    assert [n for n, _ in acts(win)] == ["set_setpoint", "move_tip", "set_setpoint"]


# ── form memory (POST_DELIVERY item 5) ──
def snapshot(win):
    from PySide6.QtWidgets import QLineEdit
    from stmbench.gui.controls import Segmented
    out = {}
    for panel in (win.operate, win.spectroscopy):
        for key, w in panel.controls.items():
            out[key] = w.value() if isinstance(w, Segmented) else w.text()
    return out


def test_no_field_of_the_operate_or_spectroscopy_tabs_loses_its_value(open_window):
    """After a run, a refusal, its acknowledgement, a tab switch and a page change every text field holds what was
    typed. (The window's language is fixed when it is opened: there is no switch to test.) The during-move fields are
    among them, so that a series of drags needs them typed once."""
    win = open_window("en", seed=3)
    typed = {"act.move_tip.x_nm": "2", "act.move_tip.y_nm": "1", "act.move_tip.speed_nm_s": "50",
             "during.set_setpoint.current_pa": "150",
             "act.scan.size_nm": "8", "act.scan.pixels": "32"}
    typed["during.set_current_range.full_scale_na"] = "100"
    shots.fill(win, typed)
    win.spectroscopy.set_field("sts", "v_start", "-1")                  # another tab: set as a marker click does
    win.spectroscopy.set_field("sts", "v_end", "1")
    want = snapshot(win)
    for step in range(3):
        run_button(win, "run.move_tip")                                  # a run
        shots.settle(win)
        assert snapshot(win) == want, f"after the run {step}"
    shots.fill(win, {"during.set_setpoint.current_pa": "1e9"})
    want = snapshot(win)
    run_button(win, "run.move_tip")                                      # a refusal
    shots.settle(win)
    assert win.worker.held and snapshot(win) == want
    run_button(win, "refusal.ack")
    shots.settle(win)
    assert snapshot(win) == want                                         # an acknowledgement
    for tab in ("report", "log", "data", "task", "manual"):
        win.sidebar.show_tab(tab)
    win.right.setCurrentWidget(win.spectroscopy)
    win.right.setCurrentWidget(win.operate)
    assert snapshot(win) == want                                         # a tab switch, a page change
    assert win.operate.collect_during()["set_setpoint"] == {"current_pa": 1e9}


def test_every_during_label_is_unique_and_shown_in_both_languages(open_window):
    for lang in ("en", "zh"):
        win = open_window(lang, seed=3, name=f"l{lang}")
        shown = [shots.widget(win, f"lbl.during.{c.action}.{c.param}").text() for c in layout.during_cells()]
        assert len(set(shown)) == 2 and all(c.en in t or c.zh in t for c, t in zip(layout.during_cells(), shown))


def test_a_range_that_is_not_one_of_the_tables_choices_is_refused_before_anything_runs(open_window):
    win = open_window("en", seed=3)
    n = len(acts(win))
    fill_move(win, **{"set_current_range.full_scale_na": "7", "set_setpoint.current_pa": "150"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    assert len(acts(win)) == n and win.worker.held
    assert win._refusal.call.startswith("set_current_range") and "must be one of" in win._refusal.words


def test_the_range_field_help_lists_the_allowed_values(open_window):
    win = open_window("en", seed=3)
    help_ = win.operate.field_help("during.set_current_range", "full_scale_na")
    assert help_.endswith("one of 10000, 1000, 100, 10, 1, 0.1") or all(v in help_ for v in ("0.1", "1", "10", "100", "1000", "10000"))
    assert help_.startswith("move range (nA)")


# ── the order when the range is lowered (screen trial 22) ──
def test_a_lowered_range_runs_the_setpoint_first_and_puts_the_range_back_first():
    s = Stand()                                           # range 1 nA, setpoint 20 pA
    out = jobs.run_move_during(s, MOVE, {"set_current_range": {"full_scale_na": 0.1}, "set_setpoint": {"current_pa": 50.0}})
    assert out.ok and [n for n, _ in s.calls] == ["set_setpoint", "set_current_range", "move_tip", "set_current_range",
                                                  "set_setpoint"]
    assert s.state == STATE
    s = Stand()                                           # a raised range keeps the order
    jobs.run_move_during(s, MOVE, {"set_current_range": {"full_scale_na": 10.0}, "set_setpoint": {"current_pa": 50.0}})
    assert [n for n, _ in s.calls] == ["set_current_range", "set_setpoint", "move_tip", "set_setpoint", "set_current_range"]


def test_a_lowered_range_refused_midway_puts_back_in_the_reverse_of_the_order_it_ran():
    s = Stand(refuse=("set_current_range", 1))
    out = jobs.run_move_during(s, MOVE, {"set_current_range": {"full_scale_na": 0.1}, "set_setpoint": {"current_pa": 50.0}})
    assert not out.ok and [n for n, _ in s.calls] == ["set_setpoint", "set_setpoint"] and s.state == STATE


def _crashes(sess) -> int:
    return sum(1 for e in sess._inst.world.events if e["kind"] == "crash")


def _real_session(tmp_path, name):
    from stmbench.gui.session import GuiSession
    sess = GuiSession(P4, 3, run_dir=tmp_path / name, export_dir=tmp_path / name / "export")
    assert sess._inst.world.physics.crash_over_range                  # the P4 world: a setpoint over the range crashes the tip
    sess.act("set_setpoint", {"current_pa": 5000.0})                  # 5 nA on the 10 nA range
    return sess


def test_lowering_the_range_in_the_old_order_crashes_the_tip_and_in_the_new_one_does_not(tmp_path):
    during = {"set_current_range": {"full_scale_na": 1.0}, "set_setpoint": {"current_pa": 500.0}}
    new = _real_session(tmp_path, "new")
    n = _crashes(new)
    out = jobs.run_move_during(new, MOVE, during)
    assert out.ok and _crashes(new) == n                              # no setpoint_over_range, the tip is as it was
    st = new.status()
    assert (st["current_full_scale_na"], st["setpoint_pa"]) == (10.0, 5000.0)
    old = _real_session(tmp_path, "old")
    n = _crashes(old)
    old.act("set_current_range", {"full_scale_na": 1.0})              # the order the console used to run
    assert _crashes(old) == n + 1                                     # 5 nA setpoint on a 1 nA range: the tip is crashed
