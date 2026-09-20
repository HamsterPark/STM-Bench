"""Where STM-Bench finds the things it does not ship — the ONE place a machine path may live.

Everything outside the repo is reached through this module. Package code never spells a
drive letter or a home directory; it asks one of the functions below, so a fresh checkout
works on any machine by setting environment variables (``tests/test_no_machine_paths.py``
excludes this file by name and fails on a literal anywhere else under ``stmsim/`` /
``stmbench/``).

Four roots, four variables (blank or whitespace = unset; ``~`` is expanded; surrounding
quotes are stripped so a pasted PowerShell value works):

=========================  =======================  ====================================
function                   variable                 default
=========================  =======================  ====================================
:func:`data_root`          ``STM_BENCH_DATA``       ``~/stm_bench`` on every platform
:func:`corpus_index_dir`   ``STM_BENCH_CORPUS_INDEX``  ``<data_root>/corpus_index``
:func:`raw_mirror_root`    ``STM_BENCH_RAW``        ``<data_root>/raw``
:func:`mast_root`          ``MAST_ROOT``            the checkout that owns the importable
                                                    ``mast`` package (``None`` if neither)
=========================  =======================  ====================================

Named sub-directories of the data root (no creation here — writers call
``mkdir(parents=True, exist_ok=True)`` on what they get): :func:`runs_dir`,
:func:`calib_dir`, :func:`index_dir`, :func:`trackA_dir`, :func:`fidelity_dir`,
:func:`sessions_dir`; anything else via :func:`data_path`.

``STM_BENCH_INDEX`` was the old name for the corpus index variable; it is still honoured
when ``STM_BENCH_CORPUS_INDEX`` is unset, with a :class:`DeprecationWarning`.

``stmbench.paths`` re-exports this module so the benchmark and the simulator cannot
disagree about where run ledgers, sessions and calibration tables live.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

ENV_VAR = "STM_BENCH_DATA"
CORPUS_ENV_VAR = "STM_BENCH_CORPUS_INDEX"
RAW_ENV_VAR = "STM_BENCH_RAW"
MAST_ENV_VAR = "MAST_ROOT"
LEGACY_CORPUS_ENV_VAR = "STM_BENCH_INDEX"        # deprecated alias of CORPUS_ENV_VAR


def _from_env(var: str) -> Path | None:
    """The variable as a path, or ``None`` when unset / blank."""
    v = os.environ.get(var, "")
    v = v.strip().strip('"').strip("'").strip()
    return Path(v).expanduser() if v else None


# ── the data root and its sub-directories ────────────────────────────────────

def data_root() -> Path:
    """Root directory for everything STM-Bench writes or reads outside the repo."""
    p = _from_env(ENV_VAR)
    if p is not None:
        return p
    return Path.home() / "stm_bench"


def data_path(*parts: str) -> Path:
    """``data_root() / parts...`` (no creation)."""
    return data_root().joinpath(*parts)


def runs_dir() -> Path:
    """Episode run directories (``stmbench.cli run --out`` default)."""
    return data_path("runs")


def calib_dir() -> Path:
    """Fitted calibration tables: ``thresholds.json``, ``creep.json``, ``working_points.json``…"""
    return data_path("calib")


def index_dir() -> Path:
    """Indices STM-Bench builds itself (e.g. ``dat_index.parquet``) — not the corpus index."""
    return data_path("index")


def trackA_dir() -> Path:
    """Track A manifests (``T1..T4.parquet``, ``meta.json``) and their rendered prompts."""
    return data_path("trackA")


def fidelity_dir() -> Path:
    """``stmsim.validate.fidelity`` report and the sim sessions it renders."""
    return data_path("fidelity")


def sessions_dir(name: str = "default") -> Path:
    """Simulator session directory (where ``Scan_Save`` writes ``.sxm``) for a named session."""
    return data_path("sessions", name)


# ── external inputs (individually configurable) ──────────────────────────────

def corpus_index_dir() -> Path:
    """Read-only index of the real-instrument corpus (private; not shipped).

    ``STM_BENCH_CORPUS_INDEX`` if set; else the deprecated ``STM_BENCH_INDEX`` (warns);
    else ``data_root()/corpus_index`` so the error a user
    sees names a directory under the root they configured.
    """
    p = _from_env(CORPUS_ENV_VAR)
    if p is not None:
        return p
    legacy = _from_env(LEGACY_CORPUS_ENV_VAR)
    if legacy is not None:
        warnings.warn(f"{LEGACY_CORPUS_ENV_VAR} is deprecated; set {CORPUS_ENV_VAR} instead",
                      DeprecationWarning, stacklevel=2)
        return legacy
    return data_path("corpus_index")


def raw_mirror_root() -> Path:
    """Root of the raw instrument-file mirror (``.dat`` / ``.sxm`` tree; private).

    ``STM_BENCH_RAW`` if set; else ``data_root()/raw``.
    """
    p = _from_env(RAW_ENV_VAR)
    if p is not None:
        return p
    return data_path("raw")


def mast_root() -> Path | None:
    """The MAST checkout (the directory that contains ``MASTv2/``).

    ``MAST_ROOT`` if set; else derived from the importable ``mast`` package
    (``<root>/MASTv2/mast/__init__.py``); else ``None`` — callers that need it say so.
    """
    p = _from_env(MAST_ENV_VAR)
    if p is not None:
        return p
    try:
        import mast  # noqa: F401  (MAST is optional; only MAST-backed features need it)
    except Exception:  # noqa: BLE001
        return None
    init = getattr(mast, "__file__", None)
    if not init:
        return None
    root = Path(init).resolve().parents[2]            # mast/__init__.py → MASTv2 → root
    return root if (root / "MASTv2" / "mast").is_dir() else None


# ── introspection ────────────────────────────────────────────────────────────

def data_available() -> bool:
    """True when :func:`data_root` exists on this machine (tests use it to skip)."""
    try:
        return data_root().is_dir()
    except OSError:
        return False


def describe() -> str:
    """One line for logs / ledgers: which root is in use and where it came from."""
    src = "env" if _from_env(ENV_VAR) is not None else "home-default"
    return f"{ENV_VAR}={data_root()} ({src}; exists={data_available()})"


__all__ = [
    "ENV_VAR", "CORPUS_ENV_VAR", "RAW_ENV_VAR", "MAST_ENV_VAR", "LEGACY_CORPUS_ENV_VAR",
    "data_root", "data_path", "runs_dir", "calib_dir", "index_dir", "trackA_dir", "fidelity_dir",
    "sessions_dir", "corpus_index_dir", "raw_mirror_root", "mast_root",
    "data_available", "describe",
]
