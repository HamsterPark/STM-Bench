"""The desktop console: the benchmark instrument in a native window (``stmbench.cli gui``), for an agent
that works through screenshots, mouse and keyboard.

The window has no network listener of any kind; its screen is the only way to operate it. It drives the
same :class:`~stmbench.instrument.Instrument` an MCP agent gets, through :class:`GuiSession`, and writes the
export directory as the episode happens. Importing this package needs no Qt: the application
(:mod:`stmbench.gui.app`, installed with ``pip install "stm-bench[gui]"``) is imported when it is run.
"""
from .session import GuiSession, SpecError, make_private_dir, read_spec

__all__ = ["GuiSession", "SpecError", "make_private_dir", "read_spec"]
