"""Small controls the panels are made of.

Each is shaped for a reader that clicks coordinates it saw in a screenshot: a choice is a row of
buttons with the selected one filled (never a drop-down that hides its options), a text field selects
its whole content when it is clicked (typing replaces it), and a long line is cut with an ellipsis
rather than growing its column.
"""
from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSizePolicy,
                               QTabBar, QTabWidget, QTextBrowser, QToolButton, QWidget)

from . import style


def strong(widget: QWidget, role: str = "") -> None:
    """Bold text for a button or a label, set on its font rather than by the style sheet: a style sheet's
    weight is not in the widget's size hint, and bold text in a button sized for plain text is clipped."""
    f = widget.font()
    f.setBold(True)
    widget.setFont(f)
    if role:
        widget.setProperty("role", role)


def polish(widget: QWidget) -> None:
    """Re-read the style sheet after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


#: no button of a row of choices is narrower than this: "Z" was 22 px wide in the 1280x720 picture and was
#: missed for its neighbour
SEG_MIN_PX = 56


def _same(a: Any, b: Any) -> bool:
    """Equal as the instrument's readings are: a number to its rounding, a flag to a flag."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or (isinstance(a, bool) and isinstance(b, bool) and a == b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


class Segmented(QWidget):
    """A row of buttons, one selected (or none yet): the visible state is the value.

    ``options`` is ``(value, text, object name)``. ``changed`` carries the new value."""

    changed = Signal(object)

    def __init__(self, options: Sequence[tuple[Any, str, str]], value: Any = None, parent: QWidget | None = None,
                 *, checkable: bool = True):
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[int, tuple[Any, QPushButton]] = {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        n = len(options)
        for i, (val, text, name) in enumerate(options):
            b = QPushButton(text)
            b.setObjectName(name)
            b.setCheckable(checkable)
            b.setProperty("seg", "only" if n == 1 else "first" if i == 0 else "last" if i == n - 1 else "mid")
            b.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            b.setMinimumWidth(style.px(SEG_MIN_PX))
            self._group.addButton(b, i)
            self._buttons[i] = (val, b)
            lay.addWidget(b)
            if value is not None and val == value:
                b.setChecked(True)
        lay.addStretch(1)
        self._group.idClicked.connect(lambda i: self.changed.emit(self._buttons[i][0]))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def value(self) -> Any:
        i = self._group.checkedId()
        return self._buttons[i][0] if i in self._buttons else None

    def set_value(self, value: Any) -> None:
        for val, b in self._buttons.values():
            if val == value:
                b.setChecked(True)
                return

    def show_value(self, value: Any) -> None:
        """Show ``value`` as the selected one without a click: the state the instrument reports, not a
        choice made here. A value no button has (or None) leaves none selected."""
        match = next((b for val, b in self._buttons.values() if _same(val, value)), None)
        self._group.setExclusive(False)
        for _, b in self._buttons.values():
            b.setChecked(b is match)
        self._group.setExclusive(True)

    def button(self, value: Any) -> QPushButton | None:
        return next((b for val, b in self._buttons.values() if val == value), None)

    def buttons(self) -> list[QPushButton]:
        return [b for _, b in self._buttons.values()]

    def set_enabled_values(self, allowed: Sequence[Any]) -> None:
        for val, b in self._buttons.values():
            b.setEnabled(val in allowed)


class SelectAllLineEdit(QLineEdit):
    """A field whose whole content is selected by every click, not only the one that gives it the focus: the
    next keystrokes replace it, whether the field was focused before, had a selection already, had its caret
    somewhere, or the pointer wobbled between press and release. No context menu, no dropped text, no dragged
    selection: the only gestures are clicking and typing (the keyboard still moves the caret). ``entered`` is
    emitted when the reader clicks into the field or tabs to it: the window describes the field last entered."""

    entered = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setAcceptDrops(False)
        self.setDragEnabled(False)            # a press on selected text would wait for a drag and, released, drop the selection

    def focusInEvent(self, ev) -> None:
        super().focusInEvent(ev)
        self.entered.emit()

    def mousePressEvent(self, ev) -> None:
        super().mousePressEvent(ev)
        self.selectAll()
        self.entered.emit()

    def mouseMoveEvent(self, ev) -> None:
        ev.accept()                           # a pointer that moves with the button down must not drag a partial selection

    def mouseReleaseEvent(self, ev) -> None:
        super().mouseReleaseEvent(ev)
        self.selectAll()                      # whatever the press and the moves in between did

    def mouseDoubleClickEvent(self, ev) -> None:
        super().mouseDoubleClickEvent(ev)
        self.selectAll()


#: what one notch of the wheel moves a text view (lines, as the view wraps them): the platform's own setting would make the
#: amount depend on the machine, and a reader of screenshots counts on the same step every time
LINES_PER_NOTCH = 3


class Notches:
    """Wheel turns as whole notches: a notch is 120 of a wheel event's angle delta, and what is left of a smaller turn
    (a finer wheel, a touchpad) waits for the next event. Positive: the wheel turned away from the reader (up)."""

    def __init__(self) -> None:
        self._rest = 0

    def take(self, ev) -> int:
        self._rest += ev.angleDelta().y()
        n = int(self._rest / 120)                                        # towards zero
        self._rest -= 120 * n
        return n


class QuietText(QPlainTextEdit):
    """Read-only text that scrolls with the wheel and does nothing else: no context menu, no zoom. A notch moves it
    :data:`LINES_PER_NOTCH` lines (visual lines, wrapped as shown), whatever the machine's own setting."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._notches = Notches()

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:         # Ctrl + wheel would zoom the text
            ev.ignore()
            return
        n = self._notches.take(ev)
        if n:
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() - n * LINES_PER_NOTCH)
        ev.accept()


class QuietBrowser(QTextBrowser):
    """The same for rich text (the log)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setOpenLinks(False)

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            ev.ignore()
            return
        super().wheelEvent(ev)


class _FixedTabBar(QTabBar):
    """Tabs change by click only: a wheel turned over the bar must not switch the page under the reader."""

    def wheelEvent(self, ev) -> None:
        ev.ignore()


class Tabs(QTabWidget):
    """A tab widget whose tab bar ignores the wheel."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setTabBar(_FixedTabBar())


class ElidedLabel(QLabel):
    """One line that ends in an ellipsis where it would not fit; the whole text stays in :meth:`full_text`."""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
        self.set_full_text(text)

    def full_text(self) -> str:
        return self._full

    def set_full_text(self, text: str) -> None:
        self._full = str(text)
        self._elide()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._elide()

    def _elide(self) -> None:
        w = max(0, self.width() - 2)
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, w) if w else self._full
        if shown != self.text():
            self.setText(shown)


class ThumbButton(QToolButton):
    """A frame's thumbnail with its short name; a click shows the frame."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setCheckable(True)
        self.setAutoRaise(False)
        self.name = ""

    def set_thumb(self, name: str, pixmap: QPixmap | None, size: int) -> None:
        self.name = name
        self.setIconSize(QSize(size, size))
        self.setFixedSize(size + style.px(8), size + style.px(8))
        if pixmap is not None:
            self.setIcon(QIcon(pixmap))
        self.setVisible(bool(name))
