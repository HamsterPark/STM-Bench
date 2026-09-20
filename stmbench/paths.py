"""Data-root helper for ``stmbench`` — the same one ``stmsim`` uses.

``stmbench`` already depends on ``stmsim``; the benchmark and the simulator must agree
on where run ledgers, sessions and calibration tables live, so this module re-exports
:mod:`stmsim.paths` rather than defining a second root. Import from here inside
``stmbench`` so a future split of the two packages has one seam to cut::

    from stmbench.paths import data_root, runs_dir, calib_dir
"""
from __future__ import annotations

from stmsim.paths import (  # noqa: F401 — re-exported on purpose
    CORPUS_ENV_VAR,
    ENV_VAR,
    LEGACY_CORPUS_ENV_VAR,
    MAST_ENV_VAR,
    RAW_ENV_VAR,
    calib_dir,
    corpus_index_dir,
    data_available,
    data_path,
    data_root,
    describe,
    fidelity_dir,
    index_dir,
    mast_root,
    raw_mirror_root,
    runs_dir,
    sessions_dir,
    trackA_dir,
)

__all__ = [
    "ENV_VAR", "CORPUS_ENV_VAR", "RAW_ENV_VAR", "MAST_ENV_VAR", "LEGACY_CORPUS_ENV_VAR",
    "data_root", "data_path", "runs_dir", "calib_dir", "index_dir", "trackA_dir", "fidelity_dir",
    "sessions_dir", "corpus_index_dir", "raw_mirror_root", "mast_root",
    "data_available", "describe",
]
