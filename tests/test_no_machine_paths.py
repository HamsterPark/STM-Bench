"""Public-release guard: package source must not spell a machine-specific path.

The rule (README "Data layout"): everything outside the repo is reached through
``stmsim.paths`` (re-exported by ``stmbench.paths``). A literal drive-rooted path such as
``D:/Users/researcher/…`` in a default argument, a ``Path(...)`` fallback or a usage docstring
means a fresh checkout on another machine silently writes into a relative directory
called ``E:`` (POSIX) or fails (Windows without that drive). A second helper that reads
``STM_BENCH_*`` / ``MAST_ROOT`` itself is the same defect one step removed: two readers
of one variable drift apart (blank-means-unset, ``~`` expansion, the POSIX default).

Scope: every ``.py`` / ``.yaml`` / ``.json`` / ``.toml`` under ``stmsim/`` and
``stmbench/``, including the throw-away ``probe_*.py`` scripts. Excluded: the one
sanctioned helper ``stmsim/paths.py`` and comment-only lines (``# …``). ``docs/`` and
``tests/`` are out of scope.

This test is STRICT: an offender fails the suite with the file:line list.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from tests.conftest import requires_mast

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("stmsim", "stmbench")
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml"}
# the canonical helper only — a second `paths.py` elsewhere in the tree is an offender
EXCLUDE_PATHS = {"stmsim/paths.py"}
# any drive-letter path (for example D:/Users/researcher, C:/Users/researcher, or F:/data)
# not preceded by an identifier character
PATTERN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]")
# a direct read of one of the helper's variables: os.environ.get("STM_BENCH_…") / os.environ["MAST_ROOT"]
ENV_READ = re.compile(r"""(?:environ(?:\.get\(|\[)|getenv\()\s*["'](?:STM_BENCH_\w+|MAST_ROOT)["']""")


def _excluded(path: Path, root: Path) -> bool:
    if path.relative_to(root).as_posix() in EXCLUDE_PATHS:
        return True
    return "__pycache__" in path.parts


def scan(root: Path = ROOT, pattern: re.Pattern = PATTERN) -> list[tuple[str, int, str]]:
    """(relative path, 1-based line, stripped line) for every offending non-comment line."""
    out: list[tuple[str, int, str]] = []
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix not in SUFFIXES or _excluded(p, root):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if pattern.search(line):
                    out.append((p.relative_to(root).as_posix(), i, s[:140]))
    return out


def _report(offenders: list[tuple[str, int, str]], what: str) -> str:
    files = sorted({f for f, _, _ in offenders})
    lines = [f"{len(offenders)} {what} in {len(files)} file(s); "
             f"route them through stmsim.paths / stmbench.paths:"]
    lines += [f"  {f}:{n}: {s}" for f, n, s in offenders]
    lines.append("files needing the helper: " + ", ".join(files))
    return "\n".join(lines)


# ── the scanner ──────────────────────────────────────────────────────────────

def test_no_machine_paths_in_package_source():
    offenders = scan()
    if offenders:
        pytest.fail(_report(offenders, "machine-path literal(s)"))


def test_no_parallel_env_readers_in_package_source():
    """Only ``stmsim/paths.py`` may read ``STM_BENCH_*`` / ``MAST_ROOT`` from the environment."""
    offenders = scan(pattern=ENV_READ)
    if offenders:
        pytest.fail(_report(offenders, "direct STM_BENCH_*/MAST_ROOT environment read(s)"))


def test_scanner_finds_a_planted_literal(tmp_path):
    """The guard must actually see a path when one is there (a scanner that never fires
    would make the green above vacuous)."""
    pkg = tmp_path / "stmsim"
    pkg.mkdir()
    (pkg / "bad.py").write_text('X = "E:/stm_bench/x"\n# E:/comment is fine\nY = 1\n', encoding="utf-8")
    (pkg / "paths.py").write_text('D = "E:/stm_bench"\n', encoding="utf-8")       # sanctioned
    (pkg / "sub").mkdir()
    (pkg / "sub" / "paths.py").write_text('D = "E:/stm_bench"\n', encoding="utf-8")  # a second helper: offender
    (pkg / "probe_x.py").write_text('D = "D:/Users/researcher/x"\n', encoding="utf-8")     # probes are NOT exempt
    (pkg / "drive_c.py").write_text('C = "C:/Users/researcher/x"\n', encoding="utf-8")
    (pkg / "drive_y.py").write_text('F = "F:/data/x"\n', encoding="utf-8")
    (pkg / "ok.py").write_text('SIZE_E = "SIZE:/"\nname = "TYPE:/x"\n', encoding="utf-8")
    hits = scan(tmp_path)
    assert hits == [("stmsim/bad.py", 1, 'X = "E:/stm_bench/x"'),
                    ("stmsim/drive_c.py", 1, 'C = "C:/Users/researcher/x"'),
                    ("stmsim/drive_y.py", 1, 'F = "F:/data/x"'),
                    ("stmsim/probe_x.py", 1, 'D = "D:/Users/researcher/x"'),
                    ("stmsim/sub/paths.py", 1, 'D = "E:/stm_bench"')]


def test_scanner_finds_a_planted_env_read(tmp_path):
    pkg = tmp_path / "stmbench"
    pkg.mkdir()
    (pkg / "own.py").write_text(
        'r = os.environ.get("STM_BENCH_DATA", "x")\n'
        "m = os.environ['MAST_ROOT']\n"
        'p = os.environ.get("MAST_NANONIS_PORT_MAIN")\n'      # not ours: fine
        'q = "STM_BENCH_DATA"\n',                             # a mention, not a read: fine
        encoding="utf-8")
    hits = scan(tmp_path, pattern=ENV_READ)
    assert [(f, n) for f, n, _ in hits] == [("stmbench/own.py", 1), ("stmbench/own.py", 2)]


# ── the helper ───────────────────────────────────────────────────────────────

def test_paths_helper_honours_env(monkeypatch, tmp_path):
    from stmsim import paths as sp
    from stmbench import paths as bp

    monkeypatch.setenv(sp.ENV_VAR, str(tmp_path))
    assert sp.data_root() == tmp_path
    assert bp.data_root() == tmp_path              # stmbench re-exports the same root
    assert sp.runs_dir() == tmp_path / "runs"
    assert sp.calib_dir() == tmp_path / "calib"
    assert sp.index_dir() == tmp_path / "index"
    assert sp.trackA_dir() == tmp_path / "trackA"
    assert sp.fidelity_dir() == tmp_path / "fidelity"
    assert sp.sessions_dir() == tmp_path / "sessions" / "default"
    assert sp.sessions_dir("x") == tmp_path / "sessions" / "x"
    assert sp.data_path("a", "b") == tmp_path / "a" / "b"
    assert sp.data_available() is True
    assert "env" in sp.describe()

    monkeypatch.setenv(sp.ENV_VAR, "   ")           # blank counts as unset
    assert sp.data_root() != tmp_path
    monkeypatch.setenv(sp.ENV_VAR, f'"{tmp_path}"')  # a pasted, quoted value still works
    assert sp.data_root() == tmp_path

    monkeypatch.setenv(sp.CORPUS_ENV_VAR, str(tmp_path / "idx"))
    assert sp.corpus_index_dir() == tmp_path / "idx"
    monkeypatch.setenv(sp.RAW_ENV_VAR, str(tmp_path / "raw"))
    assert sp.raw_mirror_root() == tmp_path / "raw"
    monkeypatch.setenv(sp.MAST_ENV_VAR, str(tmp_path / "MAST"))
    assert sp.mast_root() == tmp_path / "MAST"


def test_paths_helper_expands_home(monkeypatch, tmp_path):
    from stmsim import paths as sp

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv(sp.ENV_VAR, "~/bench")
    assert sp.data_root() == Path.home() / "bench"


@pytest.mark.parametrize("home", [PurePosixPath("/home/researcher"),
                                 PureWindowsPath("C:/Users/researcher")],
                         ids=["posix", "windows"])
def test_paths_helper_defaults_follow_home(monkeypatch, home):
    """Both path conventions place the defaults under the user's home directory."""
    from stmsim import paths as sp

    for v in (sp.ENV_VAR, sp.CORPUS_ENV_VAR, sp.RAW_ENV_VAR, sp.LEGACY_CORPUS_ENV_VAR):
        monkeypatch.delenv(v, raising=False)
    # Avoid changing os.name: pathlib itself chooses concrete path classes from it.
    monkeypatch.setattr(sp.Path, "home", classmethod(lambda cls: home))
    root = home / "stm_bench"
    assert sp.data_root() == root
    assert sp.corpus_index_dir() == root / "corpus_index"
    assert sp.raw_mirror_root() == root / "raw"
    for p in (sp.data_root(), sp.corpus_index_dir(), sp.raw_mirror_root()):
        assert p.is_absolute()


def test_paths_helper_default_description(monkeypatch):
    from stmsim import paths as sp

    monkeypatch.delenv(sp.ENV_VAR, raising=False)
    assert "home-default" in sp.describe()


def test_legacy_corpus_index_name_warns(monkeypatch, tmp_path):
    from stmsim import paths as sp

    monkeypatch.delenv(sp.CORPUS_ENV_VAR, raising=False)
    monkeypatch.setenv(sp.LEGACY_CORPUS_ENV_VAR, str(tmp_path / "old"))
    with pytest.warns(DeprecationWarning, match="STM_BENCH_INDEX is deprecated"):
        assert sp.corpus_index_dir() == tmp_path / "old"
    # the new name wins silently when both are set
    monkeypatch.setenv(sp.CORPUS_ENV_VAR, str(tmp_path / "new"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sp.corpus_index_dir() == tmp_path / "new"


def test_mast_root_unset_and_unimportable(monkeypatch):
    import builtins
    from stmsim import paths as sp

    monkeypatch.delenv(sp.MAST_ENV_VAR, raising=False)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "mast" or name.startswith("mast."):
            raise ImportError("no mast here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert sp.mast_root() is None


@requires_mast
def test_mast_root_from_importable_package(monkeypatch):
    import mast
    from stmsim import paths as sp

    monkeypatch.delenv(sp.MAST_ENV_VAR, raising=False)
    root = sp.mast_root()
    assert root is not None
    assert (root / "MASTv2" / "mast" / "__init__.py").resolve() == Path(mast.__file__).resolve()


def test_trackA_and_physics_shims_delegate(monkeypatch, tmp_path):
    """The names other modules still import resolve to the one helper."""
    from stmsim import paths as sp
    from stmbench import trackA
    from stmbench.trackA import render
    from stmsim.physics import paths as physics_paths

    monkeypatch.setenv(sp.ENV_VAR, str(tmp_path))
    monkeypatch.setenv(sp.CORPUS_ENV_VAR, str(tmp_path / "idx"))
    assert trackA.data_root() == tmp_path
    assert render.data_root() == tmp_path
    assert trackA.manifests_dir() == tmp_path / "trackA"
    assert trackA.index_dir() == tmp_path / "idx"
    assert physics_paths.default_session_dir() == tmp_path / "sessions" / "default"


def test_argparse_defaults_follow_the_env(monkeypatch, tmp_path):
    """Script defaults are computed from the helper at parse time, not frozen literals."""
    from stmsim import paths as sp
    from stmsim.calibrate import fit_creep, index_dat, working_points
    from stmbench.trackB import derive_thresholds
    from stmbench.harness import forge_trial, host_trial, skill_probe

    monkeypatch.setenv(sp.ENV_VAR, str(tmp_path))
    monkeypatch.setenv(sp.CORPUS_ENV_VAR, str(tmp_path / "idx"))
    monkeypatch.setenv(sp.RAW_ENV_VAR, str(tmp_path / "raw"))

    def defaults(mod):
        """Every ``add_argument(..., default=...)`` the module's ``main`` registers."""
        import argparse
        seen: dict[str, object] = {}
        orig = argparse.ArgumentParser.add_argument

        def spy(self, *names, **kw):
            seen[names[0]] = kw.get("default")
            return orig(self, *names, **kw)

        def stop(self, argv=None):
            raise SystemExit(0)                      # defaults are registered; never run the script

        monkeypatch.setattr(argparse.ArgumentParser, "add_argument", spy)
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
        with pytest.raises(SystemExit):
            mod.main([])
        return seen

    assert Path(defaults(fit_creep)["--out"]) == tmp_path / "calib" / "creep.json"
    assert Path(defaults(fit_creep)["--csv"]) == tmp_path / "idx" / "glance_drift.csv"
    assert Path(defaults(index_dat)["--root"]) == tmp_path / "raw"
    assert Path(defaults(index_dat)["--out"]) == tmp_path / "index" / "dat_index.parquet"
    assert Path(defaults(working_points)["--index"]) == tmp_path / "idx" / "sxm_index_full.parquet"
    assert Path(defaults(derive_thresholds)["--out"]) == tmp_path / "calib" / "thresholds.json"
    assert Path(defaults(derive_thresholds)["--index"]) == tmp_path / "idx" / "sxm_index_full.parquet"
    assert Path(defaults(forge_trial)["--root"]) == tmp_path / "forge_root"
    assert Path(defaults(host_trial)["--root"]) == tmp_path / "host_root"
    assert Path(defaults(skill_probe)["--root"]) == tmp_path / "probe_root"


# ── the vendor's name stays out of this repository ─────────────────────────
#: names that are MAST's (its client package, its patch module, its env vars, its tool) or
#: the .sxm file format's own header token; everything else that says "nanonis" is a leak
_VENDOR_ALLOWED = ("nanonis_spm", "nanonis_files", "nanonis_patch", "nanonis_manual",
                   "nanonis_script", "NanonisClass_upstream", "NanonisConfig", "MAST_NANONIS_PORT_",
                   "NANONIS_VERSION", "nanonis_spm_format_strings_reference", "Nanonis(",
                   "Nanonis.send", "Nanonis.quickSend", "import Nanonis",
                   # the ledger / budget key's former name, read for compatibility only
                   "nanonis_cmds",
                   # the rig profile's provenance: which real controller the numbers were measured on
                   "spm_controller_v5e")
_VENDOR_EXCLUDE: set[str] = set()
_VENDOR = re.compile(r"nanonis", re.IGNORECASE)


def test_the_vendor_name_appears_only_as_mast_api_or_file_format_tokens():
    """The simulator speaks a controller's wire protocol; it does not carry that
    controller's brand (2026-09-11 decision). MAST's own identifiers and the .sxm header
    token are the only places the word may appear in package source and docs."""
    offenders = []
    for d in SCAN_DIRS + ("docs",):
        for p in (ROOT / d).rglob("*"):
            if p.suffix not in SUFFIXES | {".md", ".html"} or "__pycache__" in p.parts:
                continue
            if p.relative_to(ROOT).as_posix() in _VENDOR_EXCLUDE:
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if not _VENDOR.search(line):
                    continue
                stripped = line
                for tok in _VENDOR_ALLOWED:
                    stripped = stripped.replace(tok, "")
                if _VENDOR.search(stripped):
                    offenders.append((str(p.relative_to(ROOT)), n, line.strip()[:100]))
    for p in (ROOT / "README.md", ROOT / "pyproject.toml"):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line
            for tok in _VENDOR_ALLOWED:
                stripped = stripped.replace(tok, "")
            if _VENDOR.search(stripped):
                offenders.append((p.name, n, line.strip()[:100]))
    assert not offenders, _report(offenders, "vendor name")
