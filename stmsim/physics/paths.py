"""Compatibility shim — the helper lives in :mod:`stmsim.paths`.

Kept for compatibility with :mod:`stmsim.physics.world`, which imports
``default_session_dir`` from here. It does not read the environment or name a machine path.
"""
from __future__ import annotations

from pathlib import Path

from stmsim.paths import sessions_dir


def default_session_dir() -> Path:
    """Where a :class:`~stmsim.physics.world.World` saves frames when no ``session_dir`` is given."""
    return sessions_dir("default")


__all__ = ["default_session_dir"]
