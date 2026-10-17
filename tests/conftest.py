"""Test configuration.

Marker (registered in ``pyproject.toml``)::

    from tests.conftest import requires_data

    @requires_data                      # one test that reads STM_BENCH_DATA

* ``requires_data`` — the data root ``stmsim.paths.data_root()`` exists (``STM_BENCH_DATA``,
  or ``~/stm_bench``). Absent ⇒ skip. Tests that only *write* scratch output
  should use ``tmp_path`` instead and not carry this marker.

The skip is applied at collection time below, so ``-m "not requires_data"`` selects on the
same name.
"""
from __future__ import annotations

import pytest


def data_available() -> bool:
    try:
        from stmsim.paths import data_available as _avail
        return _avail()
    except Exception:  # noqa: BLE001
        return False


requires_data = pytest.mark.requires_data

_SKIP_REASON = "STM_BENCH_DATA root not present (set STM_BENCH_DATA to the data directory)"


def pytest_collection_modifyitems(config, items):
    if data_available():
        return
    for item in items:
        if item.get_closest_marker("requires_data") is not None:
            item.add_marker(pytest.mark.skip(reason=_SKIP_REASON))
