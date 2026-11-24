"""The Operate and Spectroscopy panels: every action's parameters, each in its own labelled field.

The arrangement comes from :mod:`stmbench.gui.layout`; what a field is (number, choice, boolean), its
default, whether it is required and what a choice may be come from :data:`stmbench.instrument.ACTIONS`
when the panel is built. A click puts the job in the worker's queue at once with the values the fields
hold at that moment (buttons are never disabled while the instrument is busy). The window never
checks a value: an empty field sends nothing and a field that is not a number is sent as typed, so the
instrument refuses it in its own words.

Object names are the window's address book for tests and for anyone who scripts the screen:
``act.<action>.<param>`` for every parameter's control, ``during.<setter>.<param>`` for the Tip group's optional
during-move values (text fields too: blank leaves the setting alone), ``run.<action>`` for a run button,
``run.<action>.on`` / ``.off`` for an On / Off pair, ``run.<action>.<value>`` for one of a row of
value buttons. What an action answered is not shown here: every result goes to the list of recent
results under the tabs (:mod:`stmbench.gui.recent`), where several calls of one step stay side by side.

A numeric field's range (``Param.lo`` and ``hi``, the minimum and maximum an MCP agent's input schema carries) is
told for the field the reader entered last, on the line beside the tabs (:meth:`ActionPanel.field_help`, shown by the
window): never written by hand, in the field's own unit, and read without hovering. The panel still checks nothing:
a value outside it is sent and refused by the instrument in its own words.
"""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from ..instrument import ACTIONS
from . import fmt, style
from .controls import Segmented, SelectAllLineEdit, strong
from .labels import both, one, tr
from .layout import Cell, Choices, During, F, Group, OnOff, Run


def token(value: Any) -> str:
    """A choice as it appears in an object name: ``10000``, ``0.1``, ``X+``, ``down``."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def placeholder(p) -> str:
    if p.required:
        return "required"
    if p.default is not None:
        return f"default {fmt.num(p.default)}"
    return "optional"


def layout_during_order(cells) -> list[During]:
    """The during-move cells in the order their setters run (range, bias, setpoint)."""
    order = ["set_current_range", "set_bias", "set_setpoint"]
    return sorted(cells, key=lambda c: order.index(c.action))


def _param(action: str, name: str):
    return next(p for p in ACTIONS[action].params if p.name == name)


class ActionPanel(QScrollArea):
    """The groups of one tab. ``submit(action, args)`` is called for every click; ``field_entered(action, param)`` is
    emitted when the reader clicks into a text field."""

    field_entered = Signal(str, str)

    def __init__(self, groups: tuple[Group, ...], submit: Callable[[str, dict], None], lang: str = "zh",
                 parent: QWidget | None = None, submit_move: Callable[[dict, dict], None] | None = None):
        super().__init__(parent)
        self.groups = groups
        self.lang = lang
        self._submit = submit
        self._submit_move = submit_move                  # (move_tip's arguments, the during-move calls): see run_move_during
        self._during: dict[tuple[str, str], During] = {}
        self._cells: dict[tuple[str, str], F] = {}
        self.controls: dict[tuple[str, str], QWidget] = {}
        self.run_buttons: dict[str, QPushButton] = {}
        self._group_of: dict[str, str] = {}
        self._onoff: set[str] = set()
        self._runs: set[str] = set()
        self._ranges: dict[str, str] = {}                # actions with one button per value: the parameter
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("panel_body")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(style.px(2), style.px(2), style.px(2), style.px(2))
        lay.setSpacing(style.px(3))
        for g in groups:
            lay.addWidget(self._group(g))
        lay.addStretch(1)
        self.setWidget(body)

    def set_group_visible(self, key: str, visible: bool) -> None:
        """Show or hide one group (a rig without the part it controls, e.g. the qPlus sensor's PLL)."""
        self.findChild(QGroupBox, f"group.{key}").setVisible(visible)

    # ── building ──
    def _group(self, g: Group) -> QGroupBox:
        box = QGroupBox(both(g.en, g.zh, self.lang))
        box.setObjectName(f"group.{g.key}")
        v = QVBoxLayout(box)
        v.setContentsMargins(style.px(6), 0, style.px(6), style.px(1))
        v.setSpacing(style.px(2))
        for line in g.lines:
            h = QHBoxLayout()
            h.setSpacing(style.px(8))
            for cell in line:
                if not isinstance(cell, During):         # it borrows a setter's parameter; the setter's own group owns the action
                    self._group_of[cell.action] = g.key
                widget, stretch = self._cell(cell)
                h.addWidget(widget, stretch, Qt.AlignmentFlag.AlignBottom)
            v.addLayout(h)
        return box

    def _cell(self, c: Cell) -> tuple[QWidget, int]:
        if isinstance(c, F):
            return self._field_cell(c), 1
        if isinstance(c, Run):
            self._runs.add(c.action)
            return self._run_cell(c), 0
        if isinstance(c, OnOff):
            self._onoff.add(c.action)
            return self._onoff_cell(c), 0
        if isinstance(c, During):
            return self._during_cell(c), 1
        return self._choices_cell(c), 0

    def _label(self, name: str, en: str, zh: str, unit: str = "", required: bool = False) -> QLabel:
        text = both(en, zh, self.lang)
        html = text.replace("&", "&amp;").replace("<", "&lt;")
        if unit:
            html += f" <span style='color:{style.COLORS['muted']}'>({unit})</span>"
        if required:
            html += f" <span style='color:{style.COLORS['bad']}'>*</span>"
        lab = QLabel(html)
        lab.setObjectName(name)
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setProperty("role", "label")
        return lab

    def _field_cell(self, c: F) -> QWidget:
        p = _param(c.action, c.param)
        self._cells[(c.action, c.param)] = c
        cell = QWidget()
        v = QVBoxLayout(cell)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._label(f"lbl.{c.action}.{c.param}", c.en, c.zh, c.unit, p.required))
        name = f"act.{c.action}.{c.param}"
        if p.kind == "boolean" or p.choices is not None:
            if p.kind == "boolean":
                opts = [(True, tr("yes", self.lang), f"{name}.yes"), (False, tr("no", self.lang), f"{name}.no")]
            else:
                opts = [(ch, str(ch), f"{name}.{token(ch)}") for ch in p.choices]
            w: QWidget = Segmented(opts, p.default)
        else:
            w = SelectAllLineEdit()
            w.setFont(style.mono_font(18))
            ph = placeholder(p)
            w.setPlaceholderText(ph)
            w.setMinimumWidth(max(style.px(84), QFontMetrics(w.font()).horizontalAdvance(ph) + style.px(24)))
            w.returnPressed.connect(lambda a=c.action: self._enter(a))
            w.entered.connect(lambda a=c.action, k=c.param: self.field_entered.emit(a, k))
        w.setObjectName(name)
        self.controls[(c.action, c.param)] = w
        v.addWidget(w)
        return cell

    def _during_cell(self, c: During) -> QWidget:
        """A during-move value: a text field, blank to keep the setting. Touching it runs nothing; the value is read when
        Move tip is clicked, and checked against the setter's table entry (the range against its choices) before anything runs."""
        self._during[(c.action, c.param)] = c
        cell = QWidget()
        v = QVBoxLayout(cell)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        name = f"during.{c.action}.{c.param}"
        v.addWidget(self._label(f"lbl.{name}", c.en, c.zh, c.unit))
        w = SelectAllLineEdit()
        w.setFont(style.mono_font(18))
        ph = one("keep", self.lang)
        w.setPlaceholderText(ph)
        w.setMinimumWidth(max(style.px(84), QFontMetrics(w.font()).horizontalAdvance(ph) + style.px(24)))
        w.returnPressed.connect(lambda: self._run("move_tip"))
        w.entered.connect(lambda a=c.action, k=c.param: self.field_entered.emit(f"during.{a}", k))
        w.setObjectName(name)
        self.controls[(f"during.{c.action}", c.param)] = w
        v.addWidget(w)
        return cell

    def _button(self, name: str, text: str, role: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setObjectName(name)
        b.setAutoDefault(False)
        if role:
            strong(b, role)
        return b

    def _run_cell(self, c: Run) -> QWidget:
        b = self._button(f"run.{c.action}", both(c.en, c.zh, self.lang), "primary")
        b.clicked.connect(lambda _=False, a=c.action: self._run(a))
        self.run_buttons[c.action] = b
        return b

    def _onoff_cell(self, c: OnOff) -> QWidget:
        cell = QWidget()
        v = QVBoxLayout(cell)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._label(f"lbl.{c.action}.on", c.en, c.zh))
        # checkable: the selected one is the state the instrument reports (see show_state), not a pending choice
        seg = Segmented([(True, tr("on", self.lang), f"run.{c.action}.on"),
                         (False, tr("off", self.lang), f"run.{c.action}.off")])
        seg.setObjectName(f"act.{c.action}.on")
        for value in (True, False):
            b = seg.button(value)
            b.clicked.connect(lambda _=False, a=c.action, on=value: self._run(a, on=on))
            self.run_buttons[f"{c.action}.{'on' if value else 'off'}"] = b
        self.controls[(c.action, "on")] = seg
        v.addWidget(seg)
        return cell

    def _choices_cell(self, c: Choices) -> QWidget:
        p = _param(c.action, c.param)
        cell = QWidget()
        v = QVBoxLayout(cell)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._label(f"lbl.{c.action}.{c.param}", c.en, c.zh, c.unit))
        seg = Segmented([(ch, token(ch), f"run.{c.action}.{token(ch)}") for ch in p.choices])
        self._ranges[c.action] = c.param
        seg.setObjectName(f"act.{c.action}.{c.param}")
        for ch in p.choices:
            b = seg.button(ch)
            b.clicked.connect(lambda _=False, a=c.action, k=c.param, val=ch: self._submit(a, {k: val}))
            self.run_buttons[f"{c.action}.{token(ch)}"] = b
        self.controls[(c.action, c.param)] = seg
        v.addWidget(seg)
        return cell

    # ── running ──
    def collect(self, action: str) -> dict:
        """The fields of ``action`` as the instrument should get them: empty fields send nothing."""
        args: dict[str, Any] = {}
        for p in ACTIONS[action].params:
            w = self.controls.get((action, p.name))
            if w is None or p.name == "on":
                continue
            if isinstance(w, Segmented):
                if w.value() is not None:
                    args[p.name] = w.value()
                continue
            text = w.text().strip()
            if text == "":
                continue
            args[p.name] = fmt.parse_number(text) if p.kind in ("number", "integer") else text
        return args

    def collect_during(self) -> dict[str, dict]:
        """The during-move values that are set, by the setter that applies them, in the order they run. A blank field
        and a range left on "keep" are not set; a field that is not a number is returned as typed (the check before the
        move refuses it in the table's words)."""
        out: dict[str, dict] = {}
        for c in layout_during_order(self._during.values()):
            text = self.controls[(f"during.{c.action}", c.param)].text().strip()
            if text:
                out[c.action] = {c.param: fmt.parse_number(text)}
        return out

    def _run(self, action: str, on: bool | None = None) -> None:
        args = self.collect(action)
        if on is not None:
            args = {"on": on, **args}
        during = self.collect_during() if action == "move_tip" else {}
        if during and self._submit_move is not None:
            self._submit_move(args, during)
            return
        self._submit(action, args)

    def _enter(self, action: str) -> None:
        """Enter in a field runs its group's main action (the On button where there is an On / Off pair)."""
        if action in self._onoff:
            self._run(action, on=True)
        elif action in self._runs:
            self._run(action)

    def field_help(self, action: str, param: str) -> str:
        """One line about a text field, for the reader who just clicked it: its label as the panel shows it, with
        its unit, and the range the action table gives it, ``setpoint (pA): range 0.1 … 100000``; the label alone
        for a field with no range. The numbers are the table's, in the unit the label states."""
        if action.startswith("during."):                 # a during-move value: the range of the setter it borrows
            action = action[len("during."):]
            c = self._during.get((action, param))
        else:
            c = self._cells.get((action, param))
        if c is None:
            return ""
        p = _param(action, param)
        label = both(c.en, c.zh, self.lang) + (f" ({c.unit})" if c.unit else "")
        rng = fmt.range_text(p.lo, p.hi)
        if p.choices is not None:                        # a choice is told as its allowed values
            rng = f"{one('field_one_of', self.lang)} " + ", ".join(fmt.num(ch) for ch in p.choices)
        return f"{label}: {one('field_range', self.lang)} {rng}" if rng else label

    # ── what the window tells the panel ──
    def has_action(self, action: str) -> bool:
        return action in self._group_of

    def show_state(self, state: dict) -> None:
        """Mark what the instrument reports as the selected button of each On / Off pair and of the range
        row (``state`` is :func:`stmbench.gui.fmt.instrument_state`): the pairs and the row look like
        selectors, so they must show the state. No click is made."""
        for action, value in state.items():
            if action in self._onoff:
                self.controls[(action, "on")].show_value(value)
            elif action in self._ranges:
                self.controls[(action, self._ranges[action])].show_value(value)

    def set_field(self, action: str, param: str, text: str) -> bool:
        w = self.controls.get((action, param))
        if w is None or isinstance(w, Segmented):
            return False
        w.setText(text)
        return True
