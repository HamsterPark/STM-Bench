"""One look on every machine: Fusion, a fixed light palette and pixel font sizes.

Why fixed. The window is built and screenshotted on Windows (tests, review) and runs on Linux in the
virtual machine; an agent that learns where things are from one screenshot must find them in the
next. So no platform theme, no point sizes, no dark mode: the same palette and the same pixel sizes
(:func:`px`, scaled by ``--ui-scale``) everywhere. Text is large on purpose: the agent often sees the
screenshot downscaled to about 1280x720.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QStandardPaths
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

#: multiplies every font and size (``--ui-scale``); 1.0 is tuned for a 1920x1080 screen
SCALE = 1.0

COLORS = {
    "bg": "#eef0f4", "panel": "#ffffff", "ink": "#16181d", "muted": "#4d5563", "line": "#b8c0cc",
    "line_strong": "#7f8a9b", "accent": "#1f5fbf", "accent_dark": "#164a96", "accent_soft": "#dbe7fb",
    "ok": "#13692c", "bad": "#b42318", "warn": "#8a5300", "field": "#ffffff", "button": "#f2f4f8",
    "button_hover": "#e3eafa", "button_down": "#cddcf6", "head": "#e4e8ef", "stripe": "#f5f7fa",
}

#: preferred families, first found wins; Qt falls back per glyph for what a family lacks (Chinese, ✓ ✗ ←)
SANS = ["Segoe UI", "Ubuntu", "DejaVu Sans", "Noto Sans", "Liberation Sans", "Arial", "Microsoft YaHei UI",
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Droid Sans Fallback", "Segoe UI Symbol"]
MONO = ["Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", "Courier New",
        "Microsoft YaHei UI", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Droid Sans Fallback"]

#: font files registered when Qt knows no family at all (the offscreen platform on Windows), by file name
_FALLBACK_FILES = ("segoeui.ttf", "segoeuib.ttf", "seguisym.ttf", "consola.ttf", "consolab.ttf", "msyh.ttc",
                   "msyhbd.ttc", "arial.ttf", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf",
                   "DejaVuSansMono-Bold.ttf")


def px(n: float) -> int:
    """A size in pixels at the current ui scale."""
    return max(1, int(round(n * SCALE)))


def ensure_fonts() -> None:
    """Give Qt some fonts when it has none: a headless platform that reads no system fonts draws every
    glyph as a box. Registers the usual faces from the system font directories; a no-op wherever Qt
    already knows families (the Linux desktop)."""
    if QFontDatabase.families():
        return
    wanted = {n.lower() for n in _FALLBACK_FILES}
    for folder in QStandardPaths.standardLocations(QStandardPaths.StandardLocation.FontsLocation):
        for root, dirs, files in os.walk(folder):
            if root[len(folder):].count(os.sep) >= 3:            # /usr/share/fonts/truetype/dejavu, no deeper
                dirs.clear()
            for name in files:
                if name.lower() in wanted:
                    QFontDatabase.addApplicationFont(os.path.join(root, name))


def _first_family(candidates: list[str], default: str) -> str:
    have = set(QFontDatabase.families())
    return next((f for f in candidates if f in have), default)


def sans_family() -> str:
    return _first_family(SANS, "sans-serif")


def mono_family() -> str:
    return _first_family(MONO, "monospace")


def sans_font(pixel: float, bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(SANS)
    f.setPixelSize(px(pixel))
    f.setBold(bold)
    return f


def mono_font(pixel: float, bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(MONO)
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setFixedPitch(True)
    f.setPixelSize(px(pixel))
    f.setBold(bold)
    return f


def light_palette() -> QPalette:
    c = {k: QColor(v) for k, v in COLORS.items()}
    p = QPalette()
    for role, color in ((QPalette.ColorRole.Window, c["bg"]), (QPalette.ColorRole.WindowText, c["ink"]),
                        (QPalette.ColorRole.Base, c["field"]), (QPalette.ColorRole.AlternateBase, c["stripe"]),
                        (QPalette.ColorRole.Text, c["ink"]), (QPalette.ColorRole.Button, c["button"]),
                        (QPalette.ColorRole.ButtonText, c["ink"]), (QPalette.ColorRole.Highlight, c["accent"]),
                        (QPalette.ColorRole.HighlightedText, QColor("#ffffff")),
                        (QPalette.ColorRole.ToolTipBase, c["panel"]), (QPalette.ColorRole.ToolTipText, c["ink"]),
                        (QPalette.ColorRole.PlaceholderText, QColor("#6b7380")), (QPalette.ColorRole.Link, c["accent"])):
        p.setColor(role, color)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#8b93a0"))
    return p


def stylesheet() -> str:
    """The one style sheet. Two things in it are deliberate: weights of buttons and labels are not here
    (a style sheet's bold is not in a widget's size hint, so bold text is clipped; see ``controls.strong``),
    and the marker table's font is (the font set on it in code was lost when an ancestor was re-parented)."""
    c = COLORS
    return f"""
QWidget {{ color: {c['ink']}; }}
QMainWindow, QWidget#root {{ background: {c['bg']}; }}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {c['muted']}; }}
QLabel[role="label"] {{ color: {c['ink']}; font-size: {px(17)}px; }}
QLabel[tone="ok"] {{ color: {c['ok']}; }}
QLabel[tone="bad"] {{ color: {c['bad']}; }}
QLabel[role="pill"] {{ border-radius: {px(6)}px; padding: 1px {px(10)}px; background: {c['accent_soft']}; color: {c['accent_dark']}; }}
QLabel[role="pill"][tone="busy"] {{ background: #ffe9c2; color: {c['warn']}; }}
QLabel[role="pill"][tone="done"] {{ background: #d9f0df; color: {c['ok']}; }}
QFrame#strip {{ background: {c['panel']}; border: 1px solid {c['line']}; border-radius: {px(8)}px; }}
QFrame[role="panel"] {{ background: {c['panel']}; border: 1px solid {c['line']}; border-radius: {px(8)}px; }}
QGroupBox {{ background: {c['panel']}; border: 1px solid {c['line_strong']}; border-radius: {px(8)}px;
            margin-top: {px(12)}px; padding: {px(14)}px {px(2)}px {px(2)}px {px(2)}px; font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: {px(12)}px; padding: 0 {px(6)}px; color: {c['accent_dark']}; }}
QPushButton {{ background: {c['button']}; border: 1px solid {c['line_strong']}; border-radius: {px(6)}px;
              padding: 0 {px(12)}px; min-height: {px(32)}px; font-size: {px(17)}px; }}
QPushButton:hover {{ background: {c['button_hover']}; border-color: {c['accent']}; }}
QPushButton:pressed {{ background: {c['button_down']}; }}
QPushButton:disabled {{ color: #8b93a0; background: #f4f5f7; border-color: {c['line']}; }}
QPushButton[role="primary"] {{ background: {c['accent']}; color: #ffffff; border-color: {c['accent_dark']}; }}
QPushButton[role="primary"]:hover {{ background: #2a6fd6; }}
QPushButton[role="primary"]:pressed {{ background: {c['accent_dark']}; }}
QPushButton[role="danger"] {{ background: #fdecea; color: {c['bad']}; border-color: #e0a29c; }}
QPushButton[role="danger"]:hover {{ background: #fbd9d5; }}
QPushButton[seg] {{ border-radius: 0; padding: 0 {px(10)}px; margin: 0; }}
QPushButton[seg="first"] {{ border-top-left-radius: {px(6)}px; border-bottom-left-radius: {px(6)}px; }}
QPushButton[seg="last"] {{ border-top-right-radius: {px(6)}px; border-bottom-right-radius: {px(6)}px; }}
QPushButton[seg="only"] {{ border-radius: {px(6)}px; }}
QPushButton[seg]:checked {{ background: {c['accent']}; color: #ffffff; border-color: {c['accent_dark']}; }}
QPushButton[seg]:checked:disabled {{ background: #9db7e3; color: #ffffff; }}
QLineEdit {{ background: {c['field']}; border: 1px solid {c['line_strong']}; border-radius: {px(5)}px;
            padding: 0 {px(6)}px; min-height: {px(30)}px; selection-background-color: {c['accent']}; selection-color: #ffffff; }}
QLineEdit:focus {{ border: 2px solid {c['accent']}; padding: 0 {px(5)}px; }}
QRadioButton {{ spacing: {px(8)}px; min-height: {px(32)}px; }}
QRadioButton::indicator {{ width: {px(18)}px; height: {px(18)}px; }}
QTabWidget::pane {{ border: 1px solid {c['line']}; background: {c['panel']}; border-radius: {px(6)}px; top: -1px; }}
QTabBar::tab {{ background: {c['head']}; border: 1px solid {c['line']}; border-bottom: none; padding: {px(5)}px {px(6)}px;
               min-height: {px(26)}px; font-weight: bold; font-size: {px(17)}px; border-top-left-radius: {px(6)}px; border-top-right-radius: {px(6)}px; }}
QTabBar::tab:selected {{ background: {c['panel']}; color: {c['accent_dark']}; border-top: {px(3)}px solid {c['accent']}; }}
QTabBar::tab:!selected:hover {{ background: {c['button_hover']}; }}
QTableWidget {{ font-family: "{mono_family()}"; font-size: {px(17)}px; background: {c['field']}; alternate-background-color: {c['stripe']}; gridline-color: {c['line']};
               border: 1px solid {c['line']}; selection-background-color: {c['accent_soft']}; selection-color: {c['ink']}; }}
QTableWidget::item {{ padding: 0 {px(3)}px; }}
QHeaderView::section {{ background: {c['head']}; border: none; border-right: 1px solid {c['line']}; border-bottom: 1px solid {c['line']};
                       padding: {px(1)}px {px(5)}px; font-weight: bold; font-size: {px(15)}px; }}
QListWidget {{ background: {c['field']}; alternate-background-color: {c['stripe']}; border: 1px solid {c['line']}; border-radius: {px(5)}px; }}
QListWidget::item {{ padding: {px(4)}px {px(6)}px; border-bottom: 1px solid {c['line']}; }}
QListWidget::item:selected {{ background: {c['accent_soft']}; color: {c['ink']}; border-left: {px(4)}px solid {c['accent']}; }}
QPlainTextEdit, QTextBrowser {{ background: {c['field']}; border: 1px solid {c['line']}; border-radius: {px(5)}px; padding: {px(4)}px; }}
QProgressBar {{ border: 1px solid {c['line_strong']}; border-radius: {px(5)}px; background: #ffffff; max-height: {px(14)}px; min-height: {px(14)}px; text-align: center; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: {px(4)}px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ width: {px(16)}px; background: {c['stripe']}; }}
QScrollBar::handle:vertical {{ background: #a9b3c3; border-radius: {px(5)}px; min-height: {px(30)}px; margin: 2px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


def apply(app: QApplication, scale: float = 1.0) -> None:
    """Install the look on the application (once, before any window)."""
    global SCALE
    SCALE = max(0.5, float(scale))
    ensure_fonts()
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    app.setFont(sans_font(18))
    app.setStyleSheet(stylesheet())
