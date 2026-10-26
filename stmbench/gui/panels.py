"""The Operate and Spectroscopy panels: every action's parameters, each in its own labelled field.

The arrangement comes from :mod:`stmbench.gui.layout`; what a field is (number, choice, boolean), its
default, whether it is required and what a choice may be come from :data:`stmbench.instrument.ACTIONS`
when the panel is built. A click puts the job in the worker's queue at once with the values the fields
hold at that moment (buttons are never disabled while the instrument is busy). The window never
checks a value: an empty field sends nothing and a field that is not a number is sent as typed, so the
instrument refuses it in its own words.

Object names are the window's address book for tests and for anyone who scripts the screen:
``act.<action>.<param>`` for every parameter's control, ``run.<action>`` for a run button,
``run.<action>.on`` / ``.off`` for an On / Off pair, ``run.<action>.<value>`` for one of a row of
value buttons, ``reply.<group>`` for a group's reply line.
"""
from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from ..instrument import ACTIONS
from . import fmt, style
from .controls import ElidedLabel, Segmented, SelectAllLineEdit, polish
from .labels import both, tr
from .layout import Cell, Choices, F, Group, OnOff, Run


def token(value: Any) -> str:
    """A choice as it appears in an object name: ``10000``, ``0.1``, ``X+``, ``down``."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def placeholder(p) -> str:
    if p.required:
        return "required"
    if p.default is not None:
        return f"default {fmt.num(p.default)}"
    return "optional"


def _param(action: str, name: str):
    return next(p for p in ACTIONS[action].params if p.name == name)


class ActionPanel(QScrollArea):
    """The groups of one tab. ``submit(action, args)`` is called for every click."""

    def __init__(self, groups: tuple[Group, ...], submit: Callable[[str, dict], None], lang: str = "zh",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.groups = groups
        self.lang = lang
        self._submit = submit
        self.controls: dict[tuple[str, str], QWidget] = {}
        self.run_buttons: dict[str, QPushButton] = {}
        self.reply_labels: dict[str, ElidedLabel] = {}
        self._group_of: dict[str, str] = {}
        self._onoff: set[str] = set()
        self._runs: set[str] = set()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("panel_body")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(style.px(4), style.px(4), style.px(4), style.px(4))
        lay.setSpacing(style.px(4))
        for g in groups:
            lay.addWidget(self._group(g))
        lay.addStretch(1)
        self.setWidget(body)

    # ── building ──
    def _group(self, g: Group) -> QGroupBox:
        box = QGroupBox(both(g.en, g.zh, self.lang))
        box.setObjectName(f"group.{g.key}")
        v = QVBoxLayout(box)
        v.setContentsMargins(style.px(6), style.px(2), style.px(6), style.px(2))
        v.setSpacing(style.px(4))
        for line in g.lines:
            h = QHBoxLayout()
            h.setSpacing(style.px(8))
            for cell in line:
                self._group_of[cell.action] = g.key
                widget, stretch = self._cell(cell)
                h.addWidget(widget, stretch, Qt.AlignmentFlag.AlignBottom)
            v.addLayout(h)
        reply = ElidedLabel()
        reply.setObjectName(f"reply.{g.key}")
        reply.setProperty("role", "label")
        reply.setMinimumHeight(style.px(22))
        self.reply_labels[g.key] = reply
        v.addWidget(reply)
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
        cell = QWidget()
        v = QVBoxLayout(cell)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(style.px(2))
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
        w.setObjectName(name)
        self.controls[(c.action, c.param)] = w
        v.addWidget(w)
        return cell

    def _button(self, name: str, text: str, role: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setObjectName(name)
        b.setAutoDefault(False)
        if role:
            b.setProperty("role", role)
        return b

    def _run_cell(self, c: Run) -> QWidget:
        b = self._button(f"run.{c.action}", both(c.en, c.zh, self.lang), "primary")
        b.setMinimumWidth(style.px(92))
        b.clicked.connect(lambda _=False, a=c.action: self._run(a))
        self.run_buttons[c.action] = b
        return b

    def _onoff_cell(self, c: OnOff) -> QWidget:
        cell = QWidget()
        v = QVBoxLayout(cell)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(style.px(2))
        v.addWidget(self._label(f"lbl.{c.action}.on", c.en, c.zh))
        seg = Segmented([(True, tr("on", self.lang), f"run.{c.action}.on"),
                         (False, tr("off", self.lang), f"run.{c.action}.off")], checkable=False)
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
        v.setSpacing(style.px(2))
        v.addWidget(self._label(f"lbl.{c.action}.{c.param}", c.en, c.zh, c.unit))
        seg = Segmented([(ch, token(ch), f"run.{c.action}.{token(ch)}") for ch in p.choices], checkable=False)
        seg.setObjectName(f"act.{c.action}.{c.param}")
        for ch in p.choices:
            b = seg.button(ch)
            b.setMinimumWidth(style.px(54))
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

    def _run(self, action: str, on: bool | None = None) -> None:
        args = self.collect(action)
        if on is not None:
            args = {"on": on, **args}
        self._submit(action, args)

    def _enter(self, action: str) -> None:
        """Enter in a field runs its group's main action (the On button where there is an On / Off pair)."""
        if action in self._onoff:
            self._run(action, on=True)
        elif action in self._runs:
            self._run(action)

    # ── what the window tells the panel ──
    def has_action(self, action: str) -> bool:
        return action in self._group_of

    def set_field(self, action: str, param: str, text: str) -> bool:
        w = self.controls.get((action, param))
        if w is None or isinstance(w, Segmented):
            return False
        w.setText(text)
        return True

    def set_reply(self, action: str, ok: bool, text: str) -> None:
        lab = self.reply_labels.get(self._group_of.get(action, ""))
        if lab is None:
            return
        lab.set_full_text(("✓ " if ok else "✗ ") + text)
        lab.setProperty("tone", "ok" if ok else "bad")
        polish(lab)
