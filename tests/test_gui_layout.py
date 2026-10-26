"""The panel arrangement (stmbench.gui.layout) against the action table (no Qt): every action, every
parameter, nothing else - so a change to ``ACTIONS`` that the window does not show fails here."""
from __future__ import annotations

from collections import Counter

from stmbench.gui import layout
from stmbench.gui.layout import Choices, F, OnOff, Run
from stmbench.instrument import ACTIONS

#: the actions a P4 episode uses: the Operate tab shows all of them
P4_ACTIONS = {"scan", "set_bias", "set_setpoint", "feedback", "set_current_range", "move_tip", "drift_comp",
              "pulse", "poke", "withdraw", "approach", "coarse_move", "wait"}


def test_the_arrangement_covers_the_action_table_exactly():
    want = Counter((name, p.name) for name, spec in ACTIONS.items() for p in spec.params)
    assert Counter(layout.covered_params()) == want              # each parameter once, nothing extra
    assert Counter(layout.runners()) == Counter(list(ACTIONS))        # every action has exactly one way to run


def test_each_kind_of_cell_fits_the_parameter_it_controls():
    for _, c in layout.cells():
        spec = ACTIONS[c.action]
        params = {p.name: p for p in spec.params}
        if isinstance(c, F):
            assert c.param in params and c.param != "on", (c.action, c.param)
        elif isinstance(c, OnOff):
            assert params["on"].kind == "boolean" and params["on"].required, c.action
        elif isinstance(c, Choices):
            assert params[c.param].choices and len(spec.params) == 1, c.action
        for text in (c.en, c.zh):
            assert text.strip(), c


def test_the_operate_tab_holds_every_action_a_p4_episode_uses_and_spectroscopy_the_rest_of_the_signal_chain():
    def actions(groups):
        return {c.action for g in groups for line in g.lines for c in line}
    assert actions(layout.OPERATE) == P4_ACTIONS
    assert actions(layout.SPECTROSCOPY) == {"sts", "zspec", "lockin", "pll"}
    assert actions(layout.OPERATE) | actions(layout.SPECTROSCOPY) == set(ACTIONS)


def test_a_group_has_one_reply_line_for_all_its_actions():
    owner = layout.group_of_action()
    assert set(owner) == set(ACTIONS)
    keys = [g.key for _, groups in layout.TABS for g in groups]
    assert len(keys) == len(set(keys)) and set(owner.values()) == set(keys)
