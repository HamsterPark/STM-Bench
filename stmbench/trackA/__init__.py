"""Track A — offline decisions with operator ground truth (DESIGN.md §5.1).

Manifests (T1 continue/stop, T2 where-next, T3 stay/relocate, T4 working-point prior)
are built by :mod:`stmbench.trackA.manifests` from the historical ``.sxm`` index and
declared column-by-column in :mod:`stmbench.trackA.schema`.

Data roots come from :mod:`stmbench.paths` — nothing machine-specific is baked into code:

* ``STM_BENCH_DATA`` — benchmark data root; manifests go to ``<root>/trackA/``
  (:func:`manifests_dir`).
* ``STM_BENCH_CORPUS_INDEX`` — directory holding ``sxm_index_full.parquet``,
  ``sxm_sequence.parquet`` and ``autosave_history_by_month.csv``
  (:func:`index_dir`; the old name ``STM_BENCH_INDEX`` still works with a warning).
"""
from __future__ import annotations

from pathlib import Path

from stmbench.paths import corpus_index_dir, data_root, trackA_dir  # noqa: F401 — data_root re-exported


def index_dir() -> Path:
    """The corpus index directory the manifests are built from (``stmbench.paths.corpus_index_dir``)."""
    return corpus_index_dir()


def manifests_dir() -> Path:
    """Where the Track A manifest parquet files live: ``<data_root>/trackA``."""
    return trackA_dir()


__all__ = ["data_root", "index_dir", "manifests_dir"]
