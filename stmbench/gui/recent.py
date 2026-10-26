"""The last few results, newest first, always on screen under the action tabs.

Why. One agent step often runs several actions, and the screenshot comes after the last of them: a
refused Set bias followed by an accepted Set setpoint left only the success on screen. Every call's
result lands here, in the instrument's words, a tick in green or a cross in red, and the five most
recent stay visible whichever tab is open, so a refusal is never pushed out of sight by the successes
after it. A note (a click on an empty viewer, queued jobs dropped) is a result too, in grey.

Rows wrap rather than elide: a long refusal is the one sentence the reader needs whole. The list has the
height of six lines, so that a wrapped row does not hide the oldest; in the rare case of more, it scrolls
inside its box and nothing else in the window moves.
"""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QFrame, QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget

from . import style
from .controls import polish
from .labels import tr

N_RECENT = 5
#: the box shows this many lines of text without scrolling: five rows and one line of wrapping
LINES_SHOWN = 6
LINE_PX = 21
MARKS = {"ok": "✓ ", "bad": "✗ ", "note": "• "}


class RecentList(QGroupBox):
    """The last :data:`N_RECENT` results. Rows are ``recent.0`` (the newest) to ``recent.4``."""

    def __init__(self, lang: str, parent: QWidget | None = None):
        super().__init__(tr("recent_title", lang), parent)
        self.setObjectName("recent.box")
        self.lang = lang
        self._items: deque[tuple[str, str]] = deque(maxlen=N_RECENT)          # oldest first
        v = QVBoxLayout(self)
        v.setContentsMargins(style.px(6), 0, style.px(2), style.px(2))
        v.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("recent.scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFixedHeight(style.px(LINES_SHOWN * LINE_PX + 4))
        body = QWidget()
        body.setObjectName("recent.body")
        pal = body.palette()                                     # white like the boxes above, whatever the viewport does
        pal.setColor(QPalette.ColorRole.Window, QColor(style.COLORS["panel"]))
        body.setPalette(pal)
        body.setAutoFillBackground(True)
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(style.px(1))
        self.empty = QLabel(tr("no_call", lang))                  # until the first call: the box is not just a frame
        self.empty.setObjectName("recent.empty")
        self.empty.setProperty("role", "recent")
        self.empty.setProperty("tone", "note")
        col.addWidget(self.empty)
        self.rows: list[QLabel] = []
        for i in range(N_RECENT):
            lab = QLabel()
            lab.setObjectName(f"recent.{i}")
            lab.setWordWrap(True)
            lab.setProperty("role", "recent")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            lab.hide()
            col.addWidget(lab)
            self.rows.append(lab)
        col.addStretch(1)
        self.scroll.setWidget(body)
        v.addWidget(self.scroll)
        self._render()

    def push(self, tone: str, text: str) -> None:
        """Add a result: ``tone`` is ``ok``, ``bad`` or ``note``. The sixth oldest drops off."""
        self._items.append((tone if tone in MARKS else "note", str(text)))
        self._render()

    def entries(self) -> list[tuple[str, str]]:
        """``(tone, text)`` of what is listed, newest first (the text without its mark)."""
        return list(reversed(self._items))

    def texts(self) -> list[str]:
        """What the rows say, newest first, with their marks: for tests and logs."""
        return [MARKS[t] + x for t, x in self.entries()]

    def _render(self) -> None:
        shown = self.entries()
        self.empty.setVisible(not shown)
        for i, lab in enumerate(self.rows):
            if i < len(shown):
                tone, text = shown[i]
                lab.setText(MARKS[tone] + text)
                lab.setProperty("tone", tone)
                polish(lab)
                lab.show()
            else:
                lab.setText("")
                lab.hide()
        self.scroll.verticalScrollBar().setValue(0)             # the newest is always the first thing in view
