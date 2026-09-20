"""Test configuration.

MAST is the oracle for the wire codec (its patched nanonis_spm client decodes what we
encode), so ``MASTv2`` must be importable for the contract tests. Set ``MAST_ROOT`` or
install MAST in the active environment. Everything that needs it carries ``requires_mast``
and is skipped (not failed) when MAST is absent, so ``pytest tests`` is green on a checkout
without MAST — CI runs ``pytest tests -m "not requires_mast"`` to deselect them outright.

Markers (registered in ``pyproject.toml``)::

    from tests.conftest import requires_mast, requires_data

    pytestmark = requires_mast          # whole module
    @requires_data                      # one test that reads STM_BENCH_DATA

* ``requires_mast`` — MAST + nanonis_spm importable (``mast_available()``).
* ``requires_data`` — the data root ``stmsim.paths.data_root()`` exists (``STM_BENCH_DATA``,
  or ``~/stm_bench``). Absent ⇒ skip. Tests that only *write* scratch output
  should use ``tmp_path`` instead and not carry this marker.

Both are plain named marks; the skip is applied at collection time below, so
``-m "not requires_mast"`` / ``-m "not requires_data"`` select on the same names.
"""
from __future__ import annotations

import sys

import pytest

from stmsim.paths import mast_root

MAST_ROOT = mast_root()
MASTV2 = MAST_ROOT / "MASTv2" if MAST_ROOT is not None else None
if MASTV2 is not None and MASTV2.exists() and str(MASTV2) not in sys.path:
    sys.path.insert(0, str(MASTV2))


def mast_available() -> bool:
    try:
        import nanonis_spm  # noqa: F401
        import mast.core.nanonis_patch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def data_available() -> bool:
    try:
        from stmsim.paths import data_available as _avail
        return _avail()
    except Exception:  # noqa: BLE001
        return False


requires_mast = pytest.mark.requires_mast
requires_data = pytest.mark.requires_data
# ``slow`` — a test whose wall time is set by a MAST budget parameter, not by the
# simulator (e.g. ForgeAuTip's minimum ``time_budget_h`` = 0.1 h = 6 min of wall clock:
# MAST measures its budgets with ``time.time()``, the sim's ``time_scale`` cannot
# shorten it). Never skipped automatically — deselect explicitly with ``-m "not slow"``.
# Registered here (not in pyproject) so the marker lives next to the other two.
slow = pytest.mark.slow


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: wall time set by a MAST budget parameter (minutes); deselect with -m \"not slow\"")
    config.addinivalue_line(
        "markers",
        "isolated: must run as the only RuntimeHost of its process (MAST keeps process-level "
        "singletons a second host inherits); run with a separate `pytest -m isolated` invocation")

_SKIP_REASONS = {
    "requires_mast": "MAST + nanonis_spm not importable (set MAST_ROOT or put MASTv2 on PYTHONPATH)",
    "requires_data": "STM_BENCH_DATA root not present (set STM_BENCH_DATA to the data directory)",
}


def pytest_collection_modifyitems(config, items):
    avail = {"requires_mast": mast_available(), "requires_data": data_available()}
    for item in items:
        for name, ok in avail.items():
            if not ok and item.get_closest_marker(name) is not None:
                item.add_marker(pytest.mark.skip(reason=_SKIP_REASONS[name]))
