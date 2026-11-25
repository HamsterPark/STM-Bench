"""The panel arrangement (stmbench.gui.layout) against the action table (no Qt): every action, every
parameter, nothing else - so a change to ``ACTIONS`` that the window does not show fails here."""
from __future__ import annotations

from collections import Counter

from stmbench.gui import layout
from stmbench.gui.layout import Choices, During, F, OnOff, Reposition
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
        elif isinstance(c, Reposition):      # console-only: the call Move tip makes with nothing around it
            assert c.action == "move_tip", c
        elif isinstance(c, During):          # console-only: it borrows a setter's one parameter, and is none of its own
            assert c.action in ("set_current_range", "set_bias", "set_setpoint") and len(spec.params) == 1, c.action
            assert c.param in params and params[c.param].kind in ("number", "integer") and params[c.param].required, c
        for text in (c.en, c.zh):
            assert text.strip(), c


def test_the_operate_tab_holds_every_action_a_p4_episode_uses_and_spectroscopy_the_rest_of_the_signal_chain():
    def actions(groups):
        return {c.action for g in groups for line in g.lines for c in line}
    assert actions(layout.OPERATE) == P4_ACTIONS
    assert actions(layout.SPECTROSCOPY) == {"sts", "zspec", "lockin", "pll"}
    assert actions(layout.OPERATE) | actions(layout.SPECTROSCOPY) == set(ACTIONS)


def test_no_two_controls_carry_the_same_label_but_the_same_setting_of_the_two_sweeps():
    """A reader of a screenshot tells controls apart by their labels: "bias (V)" over the junction's field and
    over the poke's is one misread away from a wrong call. The same setting of the STS and of the Z
    spectrum (points, settle, integration, sweep back) may be named alike: each sits in its own box, named
    for its sweep, beside the button that runs it."""
    from stmbench.gui.labels import both
    twins = {("sts", "points"), ("zspec", "points"), ("sts", "settle_ms"), ("zspec", "settle_ms"),
             ("sts", "integration_ms"), ("zspec", "integration_ms"), ("sts", "backward"), ("zspec", "backward")}
    for lang in ("en", "zh"):
        shown: dict[str, list[tuple[str, str]]] = {}
        for _, c in layout.cells():
            if isinstance(c, (F, Choices, During)):
                text = f"{both(c.en, c.zh, lang)} ({c.unit})" if c.unit else both(c.en, c.zh, lang)
                shown.setdefault(text, []).append((c.action, c.param) if not isinstance(c, During)
                                                  else ("during." + c.action, c.param))
            elif isinstance(c, OnOff):
                shown.setdefault(both(c.en, c.zh, lang), []).append((c.action, "on"))
        for text, owners in shown.items():
            assert len(owners) == 1 or (set(owners) <= twins and {a for a, _ in owners} == {"sts", "zspec"}), (lang, text, owners)


def test_a_group_owns_each_of_its_actions():
    owner = layout.group_of_action()
    assert set(owner) == set(ACTIONS)
    keys = [g.key for _, groups in layout.TABS for g in groups]
    assert len(keys) == len(set(keys)) and set(owner.values()) == set(keys)


def test_a_during_cell_is_not_a_control_of_the_parameter_it_borrows():
    """The arrangement still covers the table exactly (each parameter once): the during-move cells name the setter parameter
    they borrow, and count in neither ``covered_params`` nor ``runners`` nor the group that owns the setter's action."""
    borrowed = [(c.action, c.param) for c in layout.during_cells()]
    assert borrowed and all(b in layout.covered_params() for b in borrowed)
    assert Counter(layout.covered_params())[borrowed[0]] == 1
    assert all(layout.group_of_action()[a] == "junction" for a, _ in borrowed)
