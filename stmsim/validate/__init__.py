"""Fidelity validation — does the simulator look like the instrument to MAST's own eyes?

docs/DESIGN.md §4.6. Nothing in this package changes the simulator; it renders frames, runs
MAST's detectors on them exactly as a real frame would be judged, and compares the resulting
distributions with the real corpus index. A miss is a finding for the tip / renderer /
surface owner, written into the report, never patched over here.

* :mod:`stmsim.validate.fidelity` — item 1: detector-distribution alignment (KS distances
  on the indexer's statistics, the plan's bands on MAST's detectors).
  ``python -m stmsim.validate.fidelity --out $STM_BENCH_DATA/fidelity/report.json --n 12``

Items 2–6 (sequence dynamics, poke curves, spectroscopy, real-vs-sim classifier, skill
end-to-end) live elsewhere or are still to be written; see the design document.
"""
from __future__ import annotations

from .fidelity import (  # noqa: F401
    ATOMIC, BANDS, POINTS, VERIFY, FidelitySet, ImagingPoint, compare, evaluate_frame, index_stats,
    ks_distance, make_world, real_distributions, resolve_index, scan_frame,
)

__all__ = [
    "ATOMIC", "BANDS", "POINTS", "VERIFY", "FidelitySet", "ImagingPoint", "compare", "evaluate_frame",
    "index_stats", "ks_distance", "make_world", "real_distributions", "resolve_index", "scan_frame",
]
