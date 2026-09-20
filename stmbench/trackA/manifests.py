"""Build the Track A manifests (T1–T4) from the historical ``.sxm`` index.

    python -m stmbench.trackA.manifests build --limit 3000 --since 2024-05
    python -m stmbench.trackA.manifests build            # everything

Inputs (``$STM_BENCH_CORPUS_INDEX``, see :mod:`stmbench.trackA`):

* ``sxm_index_full.parquet`` — one row per ``.sxm`` (header fields, acquired rows, …);
* ``sxm_sequence.parquet`` — instrument frames in time order with the
  ``early``/``aborted`` flags. Its neighbour columns (``dt_next``, ``move_next_frac``) are
  NOT used: a double-saved frame (§3.6) gives its twin ``dt_next = 0``, so every neighbour
  relation is recomputed here after de-duplication;
* ``autosave_history_by_month.csv`` — restart-save regime per instrument-month.

Outputs ``<$STM_BENCH_DATA>/trackA/T{1..4}.parquet`` + ``meta.json``. Columns are declared
in :mod:`stmbench.trackA.schema`; every manifest is validated before it is written.

Frame references are **relative**: ``rel_path`` and every ``context_paths_json`` entry are
paths under :func:`stmsim.paths.raw_mirror_root` (``$STM_BENCH_RAW`` / ``--raw-root``), and
consumers resolve them through the same function (``stmbench.trackA.render.resolve_path``),
so a manifest built on one machine reads on another. The absolute ``path`` column stays as
bookkeeping (role ``meta``). Frames whose index path does not lie under the raw root cannot
be resolved and are dropped (counted in ``meta.json`` → ``frames.n_outside_raw_root``); if
none lies under it the build refuses — the root is misconfigured, not the data.

Two things a reader of the index would get wrong, verified on the raw headers 2026-08-28:
the index column ``setpoint`` is the Z-controller *on* flag and ``z_ctrl_on`` is the
controller *name* (the ``:Z-CONTROLLER:`` table is shifted one column). The real current
setpoint is only in the raw header → :func:`read_setpoint_a`, opt-in via ``--read-headers``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stmbench.paths import raw_mirror_root

from . import index_dir as default_index_dir
from . import manifests_dir as default_out_dir
from . import schema as S

INDEX_FILE = "sxm_index_full.parquet"
SEQUENCE_FILE = "sxm_sequence.parquet"
AUTOSAVE_FILE = "autosave_history_by_month.csv"
INDEX_COLUMNS = ["path", "nx", "ny", "range_x_m", "range_y_m", "off_x_m", "off_y_m", "angle_deg", "bias_v",
                 "acq_frac", "acq_rows", "t_fwd_s", "comment", "rec_temp", "z_ctrl_on", "setpoint"]
SEQUENCE_COLUMNS = ["path", "group", "t", "early", "aborted"]
COPY_DIR_MARKER = "副本"   # Windows "- copy" folders inside the mirror


# ── loading ──────────────────────────────────────────────────────────────────
def load_inputs(index_dir: Path, *, instruments: list[str] | None = None, since: str | None = None,
                limit: int = 0, raw_root: str | os.PathLike | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, str]:
    """Read the three inputs; ``limit`` keeps the first N sequence rows per instrument
    (after ``since``, ``YYYY-MM``) so a build can be rehearsed on a subsample. The fourth
    value is the raw mirror root ``rel_path`` is made against: ``raw_root`` if given, else
    :func:`stmsim.paths.raw_mirror_root` — a configured root, never derived from the rows
    loaded, so ``rel_path`` is the same in a rehearsal and in the full build."""
    index_dir = Path(index_dir)
    seq = pd.read_parquet(index_dir / SEQUENCE_FILE, columns=SEQUENCE_COLUMNS)
    if instruments:
        seq = seq[seq["group"].isin(instruments)]
    if since:
        seq = seq[pd.to_datetime(seq["t"]) >= pd.Timestamp(since + "-01")]
    if limit and limit > 0:
        seq = seq.sort_values(["group", "t"]).groupby("group", sort=False).head(limit)
    seq = seq.reset_index(drop=True)
    ix = pd.read_parquet(index_dir / INDEX_FILE, columns=INDEX_COLUMNS)
    ix = ix[ix["path"].isin(seq["path"])].reset_index(drop=True)
    auto_path = index_dir / AUTOSAVE_FILE
    auto = pd.read_csv(auto_path) if auto_path.exists() else None
    return ix, seq, auto, _root_str(raw_root)


def autosave_regimes(auto: pd.DataFrame | None) -> dict[tuple[str, str], str]:
    """``{(instrument, 'YYYY-MM'): regime}`` from ``autosave_history_by_month.csv``
    (groups there are ``corpus:Rig_B`` — the prefix is dropped)."""
    if auto is None or not len(auto):
        return {}
    out: dict[tuple[str, str], str] = {}
    for g, ym, a in zip(auto["group"], auto["ym"], auto["autosave"]):
        inst = str(g).split(":")[-1]
        a = str(a).strip().lower()
        out[(inst, str(ym))] = a if a in S.AUTOSAVE_REGIMES else "unknown"
    return out


# ── material from COMMENT / file name ────────────────────────────────────────
_SUBSTRATES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Cu\s*\(?\s*111\s*\)?", re.I), "Cu(111)"),
    (re.compile(r"Cu\s*\(?\s*100\s*\)?", re.I), "Cu(100)"),
    (re.compile(r"Cu\s*\(?\s*110\s*\)?", re.I), "Cu(110)"),
    (re.compile(r"Ag\s*i?\s*\(?\s*111\s*\)?", re.I), "Ag(111)"),
    (re.compile(r"Ag\s*\(?\s*100\s*\)?", re.I), "Ag(100)"),
    (re.compile(r"Ag\s*\(?\s*110\s*\)?", re.I), "Ag(110)"),
    (re.compile(r"Au\s*\(?\s*111\s*\)?", re.I), "Au(111)"),
    (re.compile(r"Au\s*\(?\s*100\s*\)?", re.I), "Au(100)"),
    (re.compile(r"Si\s*\(?\s*111\s*\)?", re.I), "Si(111)"),
    (re.compile(r"Si\s*\(?\s*100\s*\)?", re.I), "Si(100)"),
    (re.compile(r"HOPG", re.I), "HOPG"),
    (re.compile(r"graphene", re.I), "graphene"),
    (re.compile(r"Bi2Se3", re.I), "Bi2Se3"),
    (re.compile(r"WSe2", re.I), "WSe2"),
    (re.compile(r"MoS2", re.I), "MoS2"),
    (re.compile(r"HgCrSe", re.I), "HgCrSe"),
    (re.compile(r"Ta2Pd3Te5", re.I), "Ta2Pd3Te5"),
    # bare element after a slash ("BCN/Cu#2") — face unknown
    (re.compile(r"/\s*Cu(?![a-z])"), "Cu"),
    (re.compile(r"/\s*Ag(?![a-z])"), "Ag"),
    (re.compile(r"/\s*Au(?![a-z])"), "Au"),
]
_METALS = ("Cu", "Ag", "Au")
_ADSORBATE = re.compile(r"(?:^|[\s,;])([A-Za-z][A-Za-z0-9\-]{0,11})\s*/\s*(?=Cu|Ag|Au|HOPG|Si\s*\()")
_ADSORBATE_SPACE = re.compile(r"^\s*(B)\s+(?=Ag|Cu|Au)")


def parse_material(comment: str, filename: str = "") -> tuple[str, str, str]:
    """``(material, adsorbate, genre)`` from the COMMENT (first) or the file name."""
    comment = (comment or "").strip()
    stem = re.sub(r"\d+\.sxm$", "", os.path.basename(filename or ""), flags=re.I)
    material = ""
    for text in (comment, stem):
        if not text:
            continue
        for pat, name in _SUBSTRATES:
            if pat.search(text):
                material = name
                break
        if material:
            break
    if not material:
        return "unknown", "", "unknown"
    adsorbate = ""
    m = _ADSORBATE.search(comment) or _ADSORBATE_SPACE.search(comment)
    if m:
        adsorbate = m.group(1)
    is_metal = material.startswith(_METALS)
    if is_metal:
        genre = "molecule_on_metal" if adsorbate else "clean_metal"
    elif material.startswith("Si"):
        genre = "semiconductor"
    elif material in ("HOPG", "graphene"):
        genre = "layered"
    else:
        genre = "other"
    return material, adsorbate, genre


def _materials(df: pd.DataFrame) -> pd.DataFrame:
    comments = df["comment"].fillna("").astype(str)
    names = df["path"].astype(str).map(os.path.basename)
    cache: dict[tuple[str, str], tuple[str, str, str]] = {}
    rows = []
    for c, n in zip(comments, names):
        k = (c, n)
        if k not in cache:
            cache[k] = parse_material(c, n)
        rows.append(cache[k])
    return pd.DataFrame(rows, columns=["material", "adsorbate", "genre"], index=df.index)


# ── frames table ─────────────────────────────────────────────────────────────
def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/")


def _windows_drive_path(p: str) -> bool:
    """Whether a normalised path carries a Windows drive prefix on this host or another."""
    return bool(re.match(r"^[A-Za-z]:/", p))


def common_root(paths) -> str:
    norm = [_norm_path(p) for p in paths]
    if not norm:
        return ""
    try:
        root = os.path.commonpath(norm)
    except ValueError:
        return ""
    return _norm_path(root)


def _root_str(raw_root: str | os.PathLike | None) -> str:
    """The raw mirror root as a '/'-separated string without a trailing slash."""
    return _norm_path(str(raw_root if raw_root else raw_mirror_root())).rstrip("/")


def _under_root(path: str, root: str) -> bool:
    """``path`` lies under ``root`` under the path style it was saved with.

    Saved drive-letter paths retain Windows' case-insensitive comparison even when a
    manifest is built or read on POSIX. Native POSIX paths retain case sensitivity.
    """
    p, r = path, root + "/"
    if os.name == "nt" or (_windows_drive_path(p) and _windows_drive_path(root)):
        p, r = p.lower(), r.lower()
    return p.startswith(r)


def rel_to_root(path: str, root: str) -> str:
    """``path`` relative to ``root`` ('/' separators); caller checks :func:`_under_root` first."""
    return _norm_path(path)[len(root) + 1:]


def prepare_frames(index_df: pd.DataFrame, seq_df: pd.DataFrame, autosave_df: pd.DataFrame | None = None,
                   *, raw_root: str | os.PathLike | None = None) -> tuple[pd.DataFrame, dict]:
    """Merge, de-duplicate (§3.6), and derive everything the four builders share: calendar
    keys, autosave regime, material, neighbour geometry (recomputed within instrument-day),
    abort kind and same-position runs. Returns ``(frames, info)``.

    ``raw_root`` (default :func:`stmsim.paths.raw_mirror_root`) is stripped to make
    ``rel_path``. Frames not under it are dropped and counted (``info["n_outside_raw_root"]``);
    if no frame is under it a :class:`ValueError` names the root and an example path.
    """
    seq = seq_df[[c for c in SEQUENCE_COLUMNS if c in seq_df.columns]].copy()
    seq = seq.rename(columns={"group": "instrument"})
    df = seq.merge(index_df, on="path", how="inner")
    info: dict = {"n_sequence": int(len(seq_df)), "n_merged": int(len(df))}
    df["t"] = pd.to_datetime(df["t"])
    df = df[df["t"].notna() & (df["t"].dt.year >= 2000)]                     # REC_DATE=1904 ⇒ missing
    # relative frame references: everything a consumer opens is under the raw mirror root
    root = _root_str(raw_root)
    norm = df["path"].astype(str).map(_norm_path)
    under = norm.map(lambda p: _under_root(p, root))
    info["raw_root"] = root
    info["index_common_root"] = common_root(norm)
    info["n_outside_raw_root"] = int((~under).sum())
    if len(df) and not under.any():
        raise ValueError(f"no frame path lies under the raw mirror root {root!r} (e.g. {norm.iloc[0]!r}); "
                         f"set STM_BENCH_RAW / --raw-root to the root the index was built from")
    df = df[under]
    # A Windows path is native on its source host, but embedded backslashes are literal
    # filename characters to ``pathlib`` on POSIX. Canonicalise this bookkeeping field
    # when preparing a manifest there so saved Windows paths remain readable in CI.
    if os.name != "nt":
        df["path"] = norm[under]
    df["rel_path"] = norm[under].map(lambda p: rel_to_root(p, root))
    for c in ("range_x_m", "range_y_m", "off_x_m", "off_y_m", "acq_frac", "bias_v", "t_fwd_s", "angle_deg"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["range_x_m"] > 0) & df["off_x_m"].notna() & df["off_y_m"].notna()]
    df["range_y_m"] = df["range_y_m"].where(df["range_y_m"] > 0, df["range_x_m"])
    df["angle_deg"] = df["angle_deg"].fillna(0.0)
    df["acq_frac"] = df["acq_frac"].fillna(1.0)
    if "acq_rows" not in df.columns:
        df["acq_rows"] = np.round(df["acq_frac"] * df["ny"])
    df["acq_rows"] = pd.to_numeric(df["acq_rows"], errors="coerce").fillna(0).astype(int)
    df["nx"] = pd.to_numeric(df["nx"], errors="coerce").fillna(0).astype(int)
    df["ny"] = pd.to_numeric(df["ny"], errors="coerce").fillna(0).astype(int)
    if "aborted" not in df.columns:
        df["aborted"] = df["acq_frac"] < 1.0
    df["aborted"] = df["aborted"].fillna(False).astype(bool)
    # the index column named "setpoint" is the Z-controller ON flag ('1'/'0'; see module doc)
    flag = df["setpoint"].astype(str).str.strip() if "setpoint" in df.columns else pd.Series("", index=df.index)
    df["z_feedback"] = np.select([flag == "1", flag == "0"], ["on", "off"], default="unknown")
    info["n_valid"] = int(len(df))

    # §3.6 de-duplication: (instrument, start time, rows, size, position); prefer the file
    # that is not in a "- 副本" copy folder, then the lexically first path.
    r = S.DEDUP_DECIMALS_M
    df["_copy"] = df["path"].astype(str).str.contains(COPY_DIR_MARKER, regex=False)
    df["_k_size"] = df["range_x_m"].round(r)
    df["_k_x"] = df["off_x_m"].round(r)
    df["_k_y"] = df["off_y_m"].round(r)
    key = ["instrument", "t", "ny", "acq_rows", "_k_size", "_k_x", "_k_y"]
    df = df.sort_values(["instrument", "t", "_copy", "path"], kind="stable")
    dup = df.duplicated(key, keep="first")
    info["n_duplicates_dropped"] = int(dup.sum())
    df = df[~dup].drop(columns=["_copy", "_k_size", "_k_x", "_k_y"])
    df = df.sort_values(["instrument", "t", "path"], kind="stable").reset_index(drop=True)

    # calendar / grouping / split / regime
    df["date"] = df["t"].dt.strftime("%Y-%m-%d")
    df["ym"] = df["t"].dt.strftime("%Y-%m")
    df["year"] = df["t"].dt.year.astype(int)
    df["group_key"] = df["instrument"].astype(str) + ":" + df["date"]
    df["split"] = df["group_key"].map(S.split_of)
    regimes = autosave_regimes(autosave_df)
    df["autosave_regime"] = [regimes.get((i, ym), "unknown") for i, ym in zip(df["instrument"], df["ym"])]
    df[["material", "adsorbate", "genre"]] = _materials(df)
    df["size_nm"] = df["range_x_m"] * 1e9
    df["height_nm"] = df["range_y_m"] * 1e9

    # neighbours within the instrument-day (recomputed — the parquet's are duplicate-polluted)
    g = df.groupby("group_key", sort=False)
    geo = ["t", "off_x_m", "off_y_m", "range_x_m", "range_y_m", "angle_deg", "path"]
    nxt = g[geo].shift(-1)
    prv = g[geo].shift(1)
    df["dt_next_s"] = (nxt["t"] - df["t"]).dt.total_seconds()
    df["dt_prev_s"] = (df["t"] - prv["t"]).dt.total_seconds()
    dx = nxt["off_x_m"] - df["off_x_m"]
    dy = nxt["off_y_m"] - df["off_y_m"]
    df["move_next_frac"] = np.hypot(dx, dy) / df["range_x_m"]
    df["move_prev_frac"] = np.hypot(df["off_x_m"] - prv["off_x_m"], df["off_y_m"] - prv["off_y_m"]) / df["range_x_m"]
    df["prev_size_ratio"] = prv["range_x_m"] / df["range_x_m"]
    df["next_size_ratio"] = nxt["range_x_m"] / df["range_x_m"]
    # next centre in THIS frame's coordinates: undo this frame's scan angle (controller
    # SCAN_ANGLE = rotation of the frame axes, counter-clockwise positive), normalise by size
    th = np.deg2rad(df["angle_deg"])
    df["next_u"] = (dx * np.cos(th) + dy * np.sin(th)) / df["range_x_m"]
    df["next_v"] = (-dx * np.sin(th) + dy * np.cos(th)) / df["range_y_m"]
    df["next_w_rel"] = nxt["range_x_m"] / df["range_x_m"]
    df["next_h_rel"] = nxt["range_y_m"] / df["range_y_m"]
    df["next_angle_delta_deg"] = nxt["angle_deg"] - df["angle_deg"]
    df["next_inside"] = ((df["next_u"].abs() + df["next_w_rel"] / 2 <= 0.5 + 1e-9)
                         & (df["next_v"].abs() + df["next_h_rel"] / 2 <= 0.5 + 1e-9)).fillna(False)
    df["next_path"] = nxt["path"]

    # abort kind
    df["abort_kind"] = np.where(~df["aborted"], "complete",
                                np.where(df["acq_frac"] < S.GLANCE_MAX_ACQ_FRAC, "glance", "deliberate"))

    # same-position runs (guide §6 step 2): ≤ RUN_MAX_GAP_S apart, centre within
    # STAY_MAX_MOVE_FRAC of the field, same size (± RUN_SIZE_RTOL)
    same_size = (df["prev_size_ratio"] - 1.0).abs() <= S.RUN_SIZE_RTOL
    same_run = ((df["dt_prev_s"] <= S.RUN_MAX_GAP_S) & (df["move_prev_frac"] < S.STAY_MAX_MOVE_FRAC)
                & same_size).fillna(False)
    run_start = ~same_run
    run_no = run_start.astype(int).groupby(df["group_key"], sort=False).cumsum() - 1
    df["run_id"] = df["group_key"] + ":r" + run_no.astype(str)
    rg = df.groupby("run_id", sort=False)
    df["step_idx"] = rg.cumcount().astype(int)
    df["run_len"] = rg["path"].transform("size").astype(int)
    is_last = df["step_idx"] == df["run_len"] - 1
    # what broke the run after its last frame: a moved centre is a relocation whatever the
    # gap (≤ LONG_STOP_S); "pause" is a gap at the same place; "resize" the same place, new size
    end_kind = pd.Series(
        np.select(
            [df["dt_next_s"].isna(), df["dt_next_s"] > S.LONG_STOP_S,
             df["move_next_frac"] >= S.STAY_MAX_MOVE_FRAC, df["dt_next_s"] > S.RUN_MAX_GAP_S],
            ["day_end", "long_stop", "relocate", "pause"], default="resize"),
        index=df.index)
    df["run_end_kind"] = end_kind.where(is_last).groupby(df["run_id"], sort=False).transform("last")
    df["run_last_complete"] = (~df["aborted"]).where(is_last).groupby(df["run_id"], sort=False).transform("last").astype(bool)
    info["n_frames"] = int(len(df))
    info["n_runs"] = int(df["run_id"].nunique())
    return df, info


# ── helpers ──────────────────────────────────────────────────────────────────
def _sid(task: str, *parts) -> str:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{task}-{h[:16]}"


def _context(df: pd.DataFrame, key: str, max_n: int | None) -> tuple[pd.Series, pd.Series]:
    """For each row: the earlier rows of the same ``key`` group (last ``max_n`` of them, oldest
    first) as a JSON list of ``rel_path`` (relative to the raw mirror root, like the sample's
    own ``rel_path``), and the seconds from each of their starts to this start.
    ``df`` must be in time order within every group."""
    paths_out = np.empty(len(df), dtype=object)
    dts_out = np.empty(len(df), dtype=object)
    t_ns = df["t"].values.astype("datetime64[ns]").astype("int64")
    paths = df["rel_path"].astype(str).values
    for _, idx in df.groupby(key, sort=False).indices.items():
        idx = np.asarray(idx)
        for k, i in enumerate(idx):
            lo = 0 if max_n is None else max(0, k - max_n)
            prev = idx[lo:k]
            paths_out[i] = json.dumps([paths[j] for j in prev])
            dts_out[i] = json.dumps([round((t_ns[i] - t_ns[j]) / 1e9, 3) for j in prev])
    return pd.Series(paths_out, index=df.index), pd.Series(dts_out, index=df.index)


def _finish(d: pd.DataFrame, spec: S.ManifestSpec) -> pd.DataFrame:
    out = d[spec.names].reset_index(drop=True)
    for c in spec.cols:
        if c.dtype == "int":
            out[c.name] = out[c.name].astype(int)
        elif c.dtype == "bool":
            out[c.name] = out[c.name].astype(bool)
        elif c.dtype == "float":
            out[c.name] = out[c.name].astype(float)
        else:
            out[c.name] = out[c.name].astype(str) if not c.nullable else out[c.name]
    S.assert_valid(out, spec)
    return out


# ── builders ─────────────────────────────────────────────────────────────────
def build_t1(frames: pd.DataFrame) -> pd.DataFrame:
    """Continue vs stop — see :data:`schema.T1`."""
    base = frames[frames["autosave_regime"] == "on"]
    # context = ALL earlier frames of the run (they ended before k), including the short
    # glances that are themselves not samples; computed before the head-rows filter
    ctx_paths, ctx_dt = _context(base, "run_id", None)
    d = base[base["acq_rows"] >= S.T1_HEAD_ROWS].copy()
    d["context_paths_json"] = ctx_paths.loc[d.index]
    d["context_dt_s_json"] = ctx_dt.loc[d.index]
    d["head_rows"] = S.T1_HEAD_ROWS
    d["stopped"] = d["aborted"]
    d["sample_id"] = [_sid("T1", p) for p in d["path"]]
    return _finish(d, S.T1)


def build_t2(frames: pd.DataFrame) -> pd.DataFrame:
    """Where next — see :data:`schema.T2`."""
    m = ((frames["move_next_frac"] > 0) & (frames["dt_next_s"] <= S.T2_MAX_GAP_S) & frames["next_inside"]
         & (frames["next_size_ratio"] < 1.0) & (frames["acq_frac"] >= S.T2_MIN_ACQ_FRAC))
    d = frames[m.fillna(False)].copy()
    ctx_paths, ctx_dt = _context(frames, "group_key", S.CONTEXT_FRAMES)   # the day's earlier frames, all of them
    d["context_paths_json"] = ctx_paths.loc[d.index]
    d["context_dt_s_json"] = ctx_dt.loc[d.index]
    d["zoom_in"] = d["next_size_ratio"] <= S.T2_ZOOM_MAX_RATIO
    d["sample_id"] = [_sid("T2", p) for p in d["path"]]
    return _finish(d, S.T2)


def next_action(dt_next_s: float, move_next_frac: float, next_size_ratio: float) -> tuple[str, str]:
    """``(coarse, fine)`` next-action labels for one frame."""
    if dt_next_s > S.LONG_STOP_S:
        return "long_stop", "long_stop"
    if move_next_frac >= S.RELOCATE_MIN_MOVE_FRAC:
        return "relocate", "relocate"
    if abs(next_size_ratio - 1.0) > S.SAME_SIZE_RTOL:
        return "stay", "zoom"
    if move_next_frac >= S.STAY_MAX_MOVE_FRAC:
        return "stay", "pan"
    return "stay", "restart_same"


def build_t3(frames: pd.DataFrame) -> pd.DataFrame:
    """Stay vs relocate — see :data:`schema.T3`."""
    d = frames[frames["dt_next_s"].notna()].copy()
    ctx_paths, ctx_dt = _context(frames, "group_key", S.CONTEXT_FRAMES)
    d["context_paths_json"] = ctx_paths.loc[d.index]
    d["context_dt_s_json"] = ctx_dt.loc[d.index]
    labels = [next_action(a, b, c) for a, b, c in zip(d["dt_next_s"], d["move_next_frac"], d["next_size_ratio"])]
    d["next_action"] = [x[0] for x in labels]
    d["next_action_fine"] = [x[1] for x in labels]
    d["relocate"] = d["next_action"] != "stay"
    d["sample_id"] = [_sid("T3", p) for p in d["path"]]
    return _finish(d, S.T3)


_SI = {"": 1.0, "A": 1.0, "mA": 1e-3, "uA": 1e-6, "µA": 1e-6, "nA": 1e-9, "pA": 1e-12, "fA": 1e-15}


def read_setpoint_a(path: str, *, max_bytes: int = 262_144) -> float | None:
    """Current setpoint (A) from a raw ``.sxm`` header's ``:Z-CONTROLLER:`` table, or None."""
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes)
    except OSError:
        return None
    i = raw.find(b":Z-CONTROLLER:")
    if i < 0:
        return None
    lines = raw[i:].split(b"\n")[1:3]
    if len(lines) < 2:
        return None
    names = [x.strip().lower() for x in lines[0].decode("latin-1").split("\t")]
    vals = [x.strip() for x in lines[1].decode("latin-1").split("\t")]
    if "setpoint" not in names:
        return None
    j = names.index("setpoint")
    if j >= len(vals):
        return None
    tok = vals[j].split()
    if not tok:
        return None
    try:
        value = float(tok[0])
    except ValueError:
        return None
    unit = tok[1] if len(tok) > 1 else ""
    if unit not in _SI:
        return None
    return value * _SI[unit]


def build_t4(frames: pd.DataFrame, *, read_headers: bool = False) -> pd.DataFrame:
    """Working-point prior — see :data:`schema.T4`."""
    d = frames[(frames["material"] != "unknown") & frames["bias_v"].notna() & frames["t_fwd_s"].notna()
               & (frames["t_fwd_s"] > 0)].copy()
    d["controller"] = d["z_ctrl_on"].fillna("").astype(str).str.strip() if "z_ctrl_on" in d.columns else ""
    d["rec_temp_k"] = pd.to_numeric(d["rec_temp"], errors="coerce") if "rec_temp" in d.columns else np.nan
    d["size_key"] = d["size_nm"].round(3)
    d["complete"] = ~d["aborted"]
    keys = ["group_key", "material", "adsorbate", "size_key", "bias_v", "t_fwd_s", "nx", "controller", "z_feedback"]
    agg = (d.groupby(keys, sort=False, dropna=False)
           .agg(first_path=("path", "first"), n_frames=("path", "size"), n_complete=("complete", "sum"),
                rec_temp_k=("rec_temp_k", "first"), instrument=("instrument", "first"), year=("year", "first"),
                autosave_regime=("autosave_regime", "first"), genre=("genre", "first"))
           .reset_index())
    agg["size_nm"] = agg["size_key"]
    agg["speed_nm_s"] = agg["size_nm"] / agg["t_fwd_s"]
    agg["split"] = agg["group_key"].map(S.split_of)
    agg["setpoint_a"] = np.nan
    if read_headers:
        agg["setpoint_a"] = [read_setpoint_a(p) for p in agg["first_path"]]
    agg["setpoint_a"] = pd.to_numeric(agg["setpoint_a"], errors="coerce").astype(float)
    agg["sample_id"] = [_sid("T4", *k) for k in zip(*[agg[c] for c in keys])]
    return _finish(agg, S.T4)


BUILDERS = {"T1": build_t1, "T2": build_t2, "T3": build_t3, "T4": build_t4}


def build_all(frames: pd.DataFrame, out_dir: Path, *, tasks=("T1", "T2", "T3", "T4"), read_headers: bool = False,
              info: dict | None = None) -> dict[str, Path]:
    """Build the requested manifests, write ``<out_dir>/T?.parquet`` and ``meta.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames": dict(info or {}), "tasks": {},
            "thresholds": {k: getattr(S, k) for k in dir(S) if k.isupper() and isinstance(getattr(S, k), (int, float))}}
    written: dict[str, Path] = {}
    for task in tasks:
        kw = {"read_headers": read_headers} if task == "T4" else {}
        df = BUILDERS[task](frames, **kw)
        p = out_dir / f"{task}.parquet"
        df.to_parquet(p, index=False)
        written[task] = p
        spec = S.SPECS[task]
        entry = {"rows": int(len(df)), "path": str(p), "by_split": df["split"].value_counts().to_dict(),
                 "by_instrument": df["instrument"].value_counts().to_dict(),
                 "by_regime": df["autosave_regime"].value_counts().to_dict(),
                 "inputs": spec.input_columns, "labels": spec.label_columns}
        for lab in spec.label_columns:
            s = df[lab]
            if s.dtype == bool or str(s.dtype) == "object" or str(s.dtype).startswith("str"):
                entry[f"label_{lab}"] = {str(k): int(v) for k, v in s.value_counts().items()}
        meta["tasks"][task] = entry
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stmbench.trackA.manifests")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the T1–T4 manifests")
    b.add_argument("--index-dir", default=None, help=f"directory with {INDEX_FILE} etc. (default $STM_BENCH_CORPUS_INDEX)")
    b.add_argument("--out", default=None, help="output directory (default $STM_BENCH_DATA/trackA)")
    b.add_argument("--limit", type=int, default=0, help="first N sequence rows per instrument (0 = all)")
    b.add_argument("--since", default=None, help="YYYY-MM: ignore frames before this month")
    b.add_argument("--instruments", default="", help="comma list, e.g. Rig_B (default: all in the sequence)")
    b.add_argument("--tasks", default="T1,T2,T3,T4")
    b.add_argument("--read-headers", action="store_true",
                   help="T4: read the current setpoint from each working point's first raw .sxm header")
    b.add_argument("--raw-root", default=None,
                   help="raw mirror root rel_path is made against (default $STM_BENCH_RAW / stmsim.paths.raw_mirror_root)")
    a = ap.parse_args(argv)

    index_dir = Path(a.index_dir) if a.index_dir else default_index_dir()
    out_dir = Path(a.out) if a.out else default_out_dir()
    instruments = [s.strip() for s in a.instruments.split(",") if s.strip()] or None
    tasks = tuple(s.strip().upper() for s in a.tasks.split(",") if s.strip())
    t0 = time.time()
    ix, seq, auto, raw_root = load_inputs(index_dir, instruments=instruments, since=a.since, limit=a.limit,
                                          raw_root=a.raw_root)
    if auto is None:
        print(f"warning: {AUTOSAVE_FILE} not found in {index_dir} — autosave_regime will be 'unknown' and T1 empty",
              file=sys.stderr)
    frames, info = prepare_frames(ix, seq, auto, raw_root=raw_root)
    if info["n_outside_raw_root"]:
        print(f"warning: {info['n_outside_raw_root']} frames dropped — not under raw_root {info['raw_root']!r} "
              f"(index common root {info['index_common_root']!r})", file=sys.stderr)
    print(f"frames: sequence={info['n_sequence']} merged={info['n_merged']} valid={info['n_valid']} "
          f"dup_dropped={info['n_duplicates_dropped']} kept={info['n_frames']} runs={info['n_runs']} "
          f"raw_root={info['raw_root']!r}")
    written = build_all(frames, out_dir, tasks=tasks, read_headers=a.read_headers, info=info)
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    for task, p in written.items():
        e = meta["tasks"][task]
        print(f"{task}: {e['rows']} rows  splits={e['by_split']}  instruments={e['by_instrument']}  → {p}")
    print(f"done in {time.time() - t0:.1f}s → {out_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
