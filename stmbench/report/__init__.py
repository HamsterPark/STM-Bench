"""Episode ledgers → tables (family × mode × model), diag correctness, fidelity appendix.

* :mod:`stmbench.report.summarize` — walk run directories, one row per ``episode.json``,
  render the table (``--ref-model`` adds the paired Δ with its seed-bootstrap CI).
* :mod:`stmbench.report.stats` — Wilson CI, paired Δ vs a reference model, the
  LLM-sampling vs sim variance split (docs/DESIGN.md §5.2 metrics).

Data roots come from :mod:`stmbench.paths` (``STM_BENCH_DATA``); nothing here spells a
machine path.
"""
