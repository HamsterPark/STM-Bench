"""I–Z barrier heights (κ, φ) and dI/dV(V) LDOS templates from the real ``.dat`` corpus.

DESIGN.md §4.5 row "I–Z κ/φ、I(V) 模板": the index (``calibrate/index_dat.py`` →
``$STM_BENCH_DATA/index/dat_index.parquet``) says *which* files are Z-spectroscopy and
bias-spectroscopy sweeps and the ``Comment01`` field; this module reads a
bounded sample of the actual files, fits every I–Z with the ``assess_iz`` recipe and folds
the dI/dV sweeps of each clean substrate into one median template on a common bias grid.

Outputs (both under ``stmsim.paths.calib_dir()``):

``iz_phi.json``
    per material (and for the unlabelled majority) the quantiles of κ (1/nm) and φ (eV) over
    the fits that pass the contract — points above ``1e-4 × max|I|`` only, R² > 0.9,
    0.5 ≤ φ ≤ 8 eV, no log-current jumps — exactly what
    ``mast.vision.spectroscopy.assess_iz`` calls *clean*; the failures are counted, not
    dropped silently.
``sts_templates.json``
    per material the median normalised dI/dV on ``v_grid`` (−1…+1 V, 10 mV), the number of
    curves behind every grid point, and a Shockley-onset check against the literature onsets
    (Au −0.49, Ag −0.065, Cu −0.44 V). ``stmsim.physics.junction.load_ldos_template`` reads it.

Provenance caveats the JSON carries explicitly
    * **Material comes from ``Comment01`` only.** An overlayer comment (``FeO/Au(111)``,
      ``B/Ag(111)``, ``BCN/Cu``, ``S/Cu(111)``) is *not* the clean substrate and is rejected;
      ``Ag(111)/mica`` is (the substrate is what sits on mica). Files with no usable comment
      are ``unlabelled`` — that is the whole Z-spectroscopy population of the qPlus profile
      (Rig_B 2023–2025 writes no comment and the same-day ``.sxm`` comments are working-point
      notes), so the I–Z φ prior is an *instrument* prior, not a per-material one.
    * ``archive/Rig_A/<year>`` is a byte mirror of ``corpus/Rig_A SPM data/<year>``; files are
      de-duplicated on (basename, size, saved date) before sampling.
    * The lock-in column is used as dI/dV only when it agrees in shape with the numeric
      derivative of I(V) (corr > 0.5); otherwise the numeric derivative is used and the
      choice is counted per material.

Run (STM-Bench root, MAST venv)::

    python -m stmsim.calibrate.iz_templates [--index PATH] [--out-dir DIR] [--max-per-class 600]

No MAST import is needed: the ``.dat`` parser below is self-contained (same header/columns
contract as ``mast.io.nanonis_files.read_dat``: ``key<TAB>value`` header lines, a ``[DATA]``
marker, one column-name line, tab-separated numeric rows, ragged rows NaN-padded).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from stmsim.paths import calib_dir, index_dir

KAPPA_PER_NM_PER_SQRT_EV = 5.123          # κ[1/nm] = 5.123 √φ[eV]  (assess_iz / junction.py)
IZ_FLOOR_FRAC = 1e-4                       # points with |I| > 1e-4 × max|I| enter the fit
IZ_R2_MIN = 0.90
IZ_PHI_RANGE_EV = (0.5, 8.0)
IZ_JUMP_MAD = 8.0                          # log-current step > 8 MAD = tip jump
IZ_MIN_SIGNAL_A = 10e-12                   # absolute: max|I| below 10 pA = no tunnelling signal, not an I–Z

SHOCKLEY_ONSET_V = {"Au": -0.49, "Ag": -0.065, "Cu": -0.44}
V_GRID = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.01), 3)
NORM_WINDOW_V = (0.05, 0.20)               # every clean-substrate template = 1 here (above all onsets)

_MATERIAL_RE = re.compile(
    r"^\s*(?P<el>au|ag|cu)\s*(?:\(\s*111\s*\))?"
    r"(?:\s+(?:sharp\s+tip|image\s*[\d,]*|clean(?:\s+surface)?|flat|terrace|new))*\s*$",
    re.IGNORECASE)


# ── .dat parsing ─────────────────────────────────────────────────────────────

def parse_dat_text(text: str) -> dict:
    """controller ``.dat`` → ``{"header": {key: value}, "columns": {name: ndarray}}``.

    Header rows are ``key<TAB>value`` (a trailing tab is common); the column-name line is the
    one directly after ``[DATA]``; rows that fail to parse are skipped; ragged rows are padded
    with NaN to the modal width (a truncated last row must not lose the sweep).
    """
    header: dict[str, str] = {}
    lines = text.splitlines()
    data_start = None
    for i, line in enumerate(lines):
        s = line.rstrip("\r\n")
        if s.strip() == "[DATA]":
            data_start = i + 1
            break
        if not s.strip():
            continue
        parts = s.split("\t")
        key = parts[0].strip()
        if key:
            header[key] = "\t".join(parts[1:]).strip()
    if data_start is None or data_start >= len(lines):
        return {"header": header, "columns": {}}
    names = [c.strip() for c in lines[data_start].rstrip("\r\n").split("\t") if c.strip()]
    rows: list[list[float]] = []
    for line in lines[data_start + 1:]:
        s = line.strip()
        if not s:
            continue
        vals: list[float] = []
        for tok in s.split("\t"):
            tok = tok.strip()
            try:
                vals.append(float(tok))
            except ValueError:
                vals.append(np.nan)
        if any(np.isfinite(v) for v in vals):
            rows.append(vals)
    if not rows or not names:
        return {"header": header, "columns": {}}
    width = Counter(len(r) for r in rows).most_common(1)[0][0]
    width = min(width, len(names))
    arr = np.full((len(rows), width), np.nan)
    for k, r in enumerate(rows):
        n = min(len(r), width)
        arr[k, :n] = r[:n]
    return {"header": header, "columns": {names[j]: arr[:, j] for j in range(width)}}


def read_dat(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_dat_text(f.read())


def _pick(cols: dict, prefixes: tuple[str, ...]) -> str | None:
    """First column whose lower-cased name starts with one of ``prefixes`` (in prefix order)."""
    low = {n.lower(): n for n in cols}
    for p in prefixes:
        for ln, n in low.items():
            if ln.startswith(p):
                return n
    return None


CURRENT_PREFIXES = ("current [avg] (a)", "current (a)", "current [avg]", "current")
BIAS_PREFIXES = ("bias calc (v)", "bias [avg] (v)", "bias (v)", "bias calc", "bias")
Z_PREFIXES = ("z rel (m)", "z rel", "z (m)")
LOCKIN_PREFIXES = ("li demod 1 x [avg] (a)", "li demod 1 x (a)", "li x 1 omega [avg] (a)",
                   "li x 1 omega (a)", "li demod 1 x", "li x 1 omega")


# ── material labelling ───────────────────────────────────────────────────────

def material_from_comment(comment: str | None) -> str | None:
    """``"Ag(111)"`` / ``"Au(111) sharp tip"`` / ``"Ag(111)/mica"`` → element; overlayers → None.

    The rule: the *first* slash-separated segment must be the bare substrate (element, optional
    (111), optional harmless descriptor). ``FeO/Au(111)`` and ``B/Ag(111)`` name an adsorbate
    first and are rejected; ``Ag(111)/mica`` names the substrate first and is accepted.
    """
    if not comment:
        return None
    first = str(comment).split("/")[0]
    m = _MATERIAL_RE.match(first)
    if not m:
        return None
    return m.group("el").capitalize()


# ── I–Z fit (assess_iz contract) ─────────────────────────────────────────────

def fit_iz(z_m, i_a) -> dict:
    """log|I| vs z straight-line fit — the ``assess_iz`` recipe, numbers in the junction's units.

    Returns kappa_per_nm, phi_ev, r2, n_ok, n_jumps, clean, why. ``clean`` is the assess_iz
    verdict (R² > 0.9, 0.5 ≤ φ ≤ 8 eV, no jumps); ``why`` names the first failed criterion.
    """
    z = np.asarray(z_m, float).ravel() * 1e9
    i = np.abs(np.asarray(i_a, float).ravel())
    out = {"kappa_per_nm": float("nan"), "phi_ev": float("nan"), "r2": float("nan"),
           "n_ok": 0, "n_jumps": 0, "clean": False, "why": ""}
    if z.size != i.size or z.size < 5:
        out["why"] = "too_few"
        return out
    floor = max(1e-30, IZ_FLOOR_FRAC * float(np.nanmax(i)))
    ok = np.isfinite(z) & np.isfinite(i) & (i > floor)
    out["n_ok"] = int(ok.sum())
    if ok.sum() < 5:
        out["why"] = "too_few_above_floor"
        return out
    zz, ll = z[ok], np.log(i[ok])
    order = np.argsort(zz)
    zz, ll = zz[order], ll[order]
    if float(np.ptp(zz)) <= 0:
        out["why"] = "no_z_range"
        return out
    slope, intercept = np.polyfit(zz, ll, 1)
    pred = slope * zz + intercept
    ss_res = float(np.sum((ll - pred) ** 2))
    ss_tot = float(np.sum((ll - ll.mean()) ** 2)) + 1e-12
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    kappa = abs(float(slope)) / 2.0
    phi = (kappa / KAPPA_PER_NM_PER_SQRT_EV) ** 2
    d = np.diff(ll)
    mad = float(np.median(np.abs(d - np.median(d)))) + 1e-9
    n_jumps = int(np.sum(np.abs(d - np.median(d)) > IZ_JUMP_MAD * mad))
    out.update(kappa_per_nm=kappa, phi_ev=phi, r2=r2, n_jumps=n_jumps)
    if r2 <= IZ_R2_MIN:
        out["why"] = "r2"
    elif not (IZ_PHI_RANGE_EV[0] <= phi <= IZ_PHI_RANGE_EV[1]):
        out["why"] = "phi_range"
    elif n_jumps:
        out["why"] = "jumps"
    else:
        out["clean"] = True
    return out


def fit_iz_file(d: dict, min_signal_a: float = IZ_MIN_SIGNAL_A) -> dict | None:
    """I–Z fit of a parsed ``.dat`` (needs a ``Z rel`` and a current column), plus sweep facts.

    ``min_signal_a`` is an *absolute* floor on max|I| — independent of the relative
    ``1e-4 × max|I|`` fit floor, which by construction cannot say "there is no signal": a
    zero-bias Δf(z) sweep of pure preamp noise has a max and a 1e-4 × max like any other
    curve. Below it the file is ``why="no_signal"`` and never fitted.
    """
    cols = d.get("columns") or {}
    zc, ic = _pick(cols, Z_PREFIXES), _pick(cols, CURRENT_PREFIXES)
    if zc is None or ic is None:
        return None
    z, i = np.asarray(cols[zc], float), np.asarray(cols[ic], float)
    fin = np.isfinite(i) & np.isfinite(z)
    bias = _float(d.get("header", {}).get("Bias>Bias (V)"))
    if not np.isfinite(bias):
        # 2023-era Rig_B files leave the header blank but record the bias as a column
        bc = _pick(cols, ("bias (v)", "bias [avg] (v)"))
        if bc is not None and np.isfinite(cols[bc]).any():
            bias = float(np.nanmedian(np.asarray(cols[bc], float)))
    base = {"kappa_per_nm": float("nan"), "phi_ev": float("nan"), "r2": float("nan"), "n_ok": 0,
            "n_jumps": 0, "clean": False, "bias_v": bias}
    if fin.sum() < 5:
        return {**base, "why": "no_finite_samples", "sweep_pm": float("nan"), "i_max_pa": float("nan")}
    i_max = float(np.max(np.abs(i[fin])))
    base.update(sweep_pm=float(np.max(z[fin]) - np.min(z[fin])) * 1e12, i_max_pa=i_max * 1e12)
    if i_max < min_signal_a:
        return {**base, "why": "no_signal"}
    r = fit_iz(z[fin], i[fin])
    r.update({k: base[k] for k in ("sweep_pm", "i_max_pa", "bias_v")})
    return r


def bias_bin(bias_v: float) -> str:
    """Coarse |bias| class for the I–Z breakdown (Δf(z) at 0 V vs 1 mV I–Z vs volts)."""
    b = abs(bias_v) if np.isfinite(bias_v) else float("nan")
    if not np.isfinite(b):
        return "unknown"
    if b < 0.0005:
        return "~0"
    if b < 0.005:
        return "<5mV"
    if b < 0.05:
        return "5-50mV"
    if b < 0.5:
        return "50-500mV"
    return ">=0.5V"


# ── dI/dV curves and templates ───────────────────────────────────────────────

def _float(s) -> float:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return float("nan")


def _smooth(y: np.ndarray, n: int) -> np.ndarray:
    n = max(int(n) | 1, 1)
    if n <= 1 or y.size < n:
        return y
    k = np.ones(n) / n
    pad = n // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, k, mode="valid")


def didv_from_columns(cols: dict, smooth_frac: float = 0.02,
                      lockin_min_corr: float = 0.5) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(V sorted, dI/dV, source) from one sweep; ``source`` ∈ {"lockin", "numeric"}.

    The numeric derivative is always computed; the lock-in X column replaces it only when the
    two agree in shape (Pearson > ``lockin_min_corr``) — a lock-in that was off, mis-phased or
    on another channel then does not masquerade as a spectrum.
    """
    vc, ic = _pick(cols, BIAS_PREFIXES), _pick(cols, CURRENT_PREFIXES)
    if vc is None or ic is None:
        return None
    v, i = np.asarray(cols[vc], float), np.asarray(cols[ic], float)
    ok = np.isfinite(v) & np.isfinite(i)
    if ok.sum() < 20:
        return None
    v, i = v[ok], i[ok]
    order = np.argsort(v)
    v, i = v[order], i[order]
    if float(np.ptp(v)) < 0.05:
        return None
    n_s = max(5, int(round(smooth_frac * v.size)))
    num = np.gradient(_smooth(i, n_s), v)
    src = "numeric"
    out = num
    lc = _pick(cols, LOCKIN_PREFIXES)
    if lc is not None:
        li = np.asarray(cols[lc], float)[ok][order]
        if np.all(np.isfinite(li)) and float(np.std(li)) > 0 and float(np.std(num)) > 0:
            c = float(np.corrcoef(_smooth(li, n_s), num)[0, 1])
            if c > lockin_min_corr:
                out, src = li, "lockin"
    return v, out, src


def curve_is_usable(v: np.ndarray, i: np.ndarray, saturation_a: float = 9.9e-9,
                    max_spikes: int = 2, min_jump_frac: float = 0.02) -> tuple[bool, str]:
    """Reject saturated sweeps and sweeps with tip switches.

    A tip switch is a *single-sample* jump of the current: ``|ΔI| > 8 × median|ΔI|`` **and**
    ``|ΔI| > min_jump_frac × max|I|``. The second clause keeps a noiseless smooth curve with a
    change of slope (a Shockley onset) from tripping the first — ``assess_iv``'s bare 8-MAD rule
    does on synthetic data, because the MAD of a piecewise-constant derivative is ~0.
    """
    if float(np.nanmax(np.abs(i))) >= saturation_a:
        return False, "saturated"
    order = np.argsort(v)
    isn = i[order] / (float(np.nanmax(np.abs(i))) + 1e-30)
    d1 = np.abs(np.diff(isn))
    thr = max(8.0 * float(np.median(d1)), min_jump_frac)
    n_spikes = int(np.sum(d1 > thr))
    if n_spikes > max_spikes:
        return False, "spikes"
    return True, ""


def normalise_to_window(v: np.ndarray, y: np.ndarray, window: tuple[float, float] = NORM_WINDOW_V
                        ) -> np.ndarray | None:
    """Divide by the mean over ``window`` (so sign and preamp gain drop out); None if uncovered."""
    m = (v >= window[0]) & (v <= window[1])
    if m.sum() < 3:
        return None
    ref = float(np.mean(y[m]))
    if not np.isfinite(ref) or abs(ref) < 1e-30:
        return None
    return y / ref


def build_template(curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray = V_GRID,
                   min_n: int = 5) -> dict:
    """Median of per-curve normalised dI/dV on ``grid``; points backed by < ``min_n`` curves are NaN.

    Each curve is interpolated only inside its own sweep range, so a ±200 mV sweep votes for
    nothing beyond ±200 mV.
    """
    stack = np.full((len(curves), grid.size), np.nan)
    for k, (v, y) in enumerate(curves):
        inside = (grid >= v.min()) & (grid <= v.max())
        stack[k, inside] = np.interp(grid[inside], v, y)
    n = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # all-NaN grid points are expected
        med = np.nanmedian(stack, axis=0) if len(curves) else np.full(grid.size, np.nan)
        p25 = np.nanpercentile(stack, 25, axis=0) if len(curves) else med
        p75 = np.nanpercentile(stack, 75, axis=0) if len(curves) else med
    keep = n >= min_n
    med = np.where(keep, med, np.nan)
    return {"v_grid": [float(x) for x in grid], "didv": _nanlist(med), "p25": _nanlist(np.where(keep, p25, np.nan)),
            "p75": _nanlist(np.where(keep, p75, np.nan)), "n_per_point": [int(x) for x in n],
            "n_curves": int(len(curves)),
            "v_covered": ([float(grid[keep].min()), float(grid[keep].max())] if keep.any() else None)}


def onset_check(grid: np.ndarray, didv: np.ndarray, expected_v: float,
                tol_v: float = 0.06, min_step: float = 1.15) -> dict:
    """Is there a conductance step at the Shockley onset the literature gives?

    Below = mean over [exp−0.25, exp−0.08] V, above = mean over [exp+0.05, exp+0.20] V;
    detected onset = first grid point past the below-window where the (5-point smoothed)
    template crosses the midpoint. ``ok`` needs both windows covered, step ≥ ``min_step`` and
    |detected − expected| ≤ ``tol_v``; otherwise ``why`` says which.
    """
    g, y = np.asarray(grid, float), np.asarray(didv, float)
    out = {"expected_v": float(expected_v), "detected_v": None, "step_ratio": None, "ok": False, "why": ""}
    below = (g >= expected_v - 0.25) & (g <= expected_v - 0.08) & np.isfinite(y)
    above = (g >= expected_v + 0.05) & (g <= expected_v + 0.20) & np.isfinite(y)
    if below.sum() < 3 or above.sum() < 3:
        out["why"] = "window_not_covered"
        return out
    lo, hi = float(np.mean(y[below])), float(np.mean(y[above]))
    out["step_ratio"] = (hi / lo) if abs(lo) > 1e-12 else None
    ys = _smooth(np.where(np.isfinite(y), y, np.nan), 5)
    mid = 0.5 * (lo + hi)
    start = int(np.flatnonzero(below).max())
    cand = np.flatnonzero((np.arange(g.size) > start) & np.isfinite(ys) & (ys >= mid) & (g <= expected_v + 0.25))
    if cand.size:
        out["detected_v"] = float(g[cand.min()])
    if out["step_ratio"] is None or out["step_ratio"] < min_step:
        out["why"] = "no_step"
    elif out["detected_v"] is None or abs(out["detected_v"] - expected_v) > tol_v:
        out["why"] = "onset_off"
    else:
        out["ok"] = True
    return out


def _nanlist(a) -> list:
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, float)]


def _quantiles(x) -> dict | None:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if x.size == 0:
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {"n": int(x.size), "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4])}


# ── corpus run ───────────────────────────────────────────────────────────────

def _year_of(path: str) -> str:
    """Year from a directory component ``2016`` / ``201706`` / ``20191101``; ``"?"`` if none."""
    m = re.search(r"/(20\d\d)(?:\d\d|\d{4})?(?:/|$)", path.replace("\\", "/"))
    return m.group(1) if m else "?"


def select_rows(df, max_per_class: int, seed: int = 0):
    """Classify + de-duplicate + bound the index; returns (iz_rows, iv_rows) DataFrames.

    Adds ``material`` (Au/Ag/Cu/None from Comment01) and ``kind`` (iz / iv). Duplicates
    (mirror trees) collapse on (basename, size, saved date).
    """
    import pandas as pd

    d = df[(df["err"].fillna("") == "")].copy()
    exp = d["h:Experiment"].fillna("").str.lower()
    d["kind"] = np.where(exp.str.contains("z spectroscopy"), "iz",
                         np.where(exp.str.contains("bias spectroscopy"), "iv", ""))
    d = d[d["kind"] != ""]
    d["comment"] = d["h:Comment01"].fillna("")
    d["material"] = d["comment"].map(material_from_comment)
    d["base"] = d["path"].str.replace("\\", "/").str.rsplit("/", n=1).str[-1]
    d = d.drop_duplicates(subset=["base", "size", "h:Saved Date"])
    parts = []
    for (kind, mat), g in d.groupby(["kind", d["material"].fillna("unlabelled")]):
        if len(g) > max_per_class:
            g = g.sample(max_per_class, random_state=seed)
        parts.append(g)
    d = pd.concat(parts) if parts else d.iloc[:0]
    return d[d["kind"] == "iz"], d[d["kind"] == "iv"]


def run_iz(rows) -> dict:
    """Fit every selected I–Z file; per-material and per-(group, year) quantiles + failure tallies."""
    fits: list[dict] = []
    for r in rows.itertuples():
        try:
            d = read_dat(r.path)
        except OSError as exc:
            fits.append({"material": r.material, "group": r.group, "year": _year_of(r.path),
                         "clean": False, "why": f"read:{type(exc).__name__}"})
            continue
        f = fit_iz_file(d)
        if f is None:
            f = {"clean": False, "why": "no_z_or_current_column"}
        f.update(material=(r.material if isinstance(r.material, str) else None), group=r.group,
                 year=_year_of(r.path))
        fits.append(f)

    def summary(sub: list[dict], with_bias: bool = True) -> dict:
        clean = [f for f in sub if f.get("clean")]
        out = {"n_files": len(sub), "n_clean": len(clean),
               "why_failed": dict(Counter(f.get("why", "") for f in sub if not f.get("clean"))),
               "phi_ev": _quantiles([f["phi_ev"] for f in clean]),
               "kappa_per_nm": _quantiles([f["kappa_per_nm"] for f in clean]),
               "r2": _quantiles([f["r2"] for f in clean]),
               "sweep_pm": _quantiles([f.get("sweep_pm") for f in sub if "sweep_pm" in f]),
               "i_max_pa": _quantiles([f.get("i_max_pa") for f in sub if "i_max_pa" in f]),
               "bias_v": _quantiles([f.get("bias_v") for f in sub if "bias_v" in f]),
               "phi_ev_all_fits": _quantiles([f.get("phi_ev") for f in sub if "phi_ev" in f])}
        if with_bias:
            # a 0 V "Z spectroscopy" is a Δf(z) curve on the qPlus rig, not an I–Z: keep them apart
            out["by_bias_bin"] = {b: summary(g, with_bias=False)
                                  for b, g in _groupby(sub, lambda f: bias_bin(f.get("bias_v", float("nan")))).items()}
        return out

    out = {"materials": {}, "unlabelled": None, "by_group_year": {}, "n_files": len(fits)}
    for mat in ("Au", "Ag", "Cu"):
        sub = [f for f in fits if f["material"] == mat]
        out["materials"][mat] = summary(sub) if sub else {"n_files": 0, "n_clean": 0}
    out["unlabelled"] = summary([f for f in fits if f["material"] is None])
    for key, sub in _groupby(fits, lambda f: f"{f['group']}/{f['year']}").items():
        out["by_group_year"][key] = summary(sub, with_bias=False)
    return out


def run_iv(rows, min_n: int = 5) -> dict:
    """Per material: normalised dI/dV template on V_GRID, onset check, source/rejection tallies."""
    curves: dict[str, list] = {"Au": [], "Ag": [], "Cu": []}
    tally: dict[str, Counter] = {m: Counter() for m in curves}
    ranges: dict[str, list] = {m: [] for m in curves}
    per_curve: dict[str, list] = {m: [] for m in curves}
    for r in rows.itertuples():
        mat = r.material if isinstance(r.material, str) else None
        if mat not in curves:
            continue
        try:
            d = read_dat(r.path)
        except OSError as exc:
            tally[mat][f"read:{type(exc).__name__}"] += 1
            continue
        cols = d.get("columns") or {}
        vc, ic = _pick(cols, BIAS_PREFIXES), _pick(cols, CURRENT_PREFIXES)
        if vc is None or ic is None:
            tally[mat]["no_bias_or_current_column"] += 1
            continue
        v_raw, i_raw = np.asarray(cols[vc], float), np.asarray(cols[ic], float)
        okm = np.isfinite(v_raw) & np.isfinite(i_raw)
        if okm.sum() < 20:
            tally[mat]["too_few_points"] += 1
            continue
        usable, why = curve_is_usable(v_raw[okm], i_raw[okm])
        if not usable:
            tally[mat][why] += 1
            continue
        got = didv_from_columns(cols)
        if got is None:
            tally[mat]["no_derivative"] += 1
            continue
        v, y, src = got
        yn = normalise_to_window(v, y)
        if yn is None:
            tally[mat]["norm_window_not_covered"] += 1
            continue
        curves[mat].append((v, yn))
        tally[mat][src] += 1
        ranges[mat].append((float(v.min()), float(v.max())))
        per_curve[mat].append({"step_ratio": step_ratio_at(v, yn, SHOCKLEY_ONSET_V[mat]), "source": src,
                               "year": _year_of(r.path), "comment": str(getattr(r, "comment", "") or "")})
    out = {"v_grid": [float(x) for x in V_GRID], "norm_window_v": list(NORM_WINDOW_V), "materials": {}}
    for mat, cs in curves.items():
        t = build_template(cs, V_GRID, min_n=min_n)
        t["onset"] = onset_check(V_GRID, np.asarray(t["didv"], dtype=float), SHOCKLEY_ONSET_V[mat])
        pc = per_curve[mat]
        # the same step ratio curve by curve: a template can only be as good as the curves behind it
        t["onset"]["per_curve_step_ratio"] = {
            "all": _quantiles([c["step_ratio"] for c in pc]),
            "by_source": {s: _quantiles([c["step_ratio"] for c in g]) for s, g in _groupby(pc, lambda c: c["source"]).items()},
            "by_year": {y: _quantiles([c["step_ratio"] for c in g]) for y, g in sorted(_groupby(pc, lambda c: c["year"]).items())},
            "by_comment": {y: _quantiles([c["step_ratio"] for c in g])
                           for y, g in sorted(_groupby(pc, lambda c: c["comment"]).items())}}
        t["tally"] = dict(tally[mat])
        t["sweep_ranges_v"] = dict(Counter(f"{a:+.2f}..{b:+.2f}" for a, b in ranges[mat]).most_common(8))
        t["n_files_selected"] = int((rows["material"] == mat).sum())
        out["materials"][mat] = t
    return out


def step_ratio_at(v: np.ndarray, y: np.ndarray, onset_v: float) -> float | None:
    """mean dI/dV over [onset+0.05, onset+0.20] / mean over [onset−0.25, onset−0.08]; None if uncovered."""
    below = (v >= onset_v - 0.25) & (v <= onset_v - 0.08)
    above = (v >= onset_v + 0.05) & (v <= onset_v + 0.20)
    if below.sum() < 3 or above.sum() < 3:
        return None
    lo = float(np.mean(y[below]))
    return float(np.mean(y[above]) / lo) if abs(lo) > 1e-12 else None


def _groupby(items, key) -> dict:
    out: dict = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", default=str(index_dir() / "dat_index.parquet"),
                    help="dat index parquet (default $STM_BENCH_DATA/index/dat_index.parquet)")
    ap.add_argument("--out-dir", default=str(calib_dir()),
                    help="where iz_phi.json / sts_templates.json go (default $STM_BENCH_DATA/calib)")
    ap.add_argument("--max-per-class", type=int, default=600,
                    help="bounded sample per (kind, material) class after de-duplication")
    ap.add_argument("--min-n", type=int, default=5, help="curves a template grid point needs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import pandas as pd

    t0 = time.time()
    df = pd.read_parquet(a.index)
    iz_rows, iv_rows = select_rows(df, a.max_per_class, a.seed)
    print(f"index {len(df)} rows → {len(iz_rows)} I–Z + {len(iv_rows)} I(V) files selected "
          f"(max {a.max_per_class}/class)", flush=True)
    meta = {"index": str(a.index), "n_index_rows": int(len(df)), "max_per_class": a.max_per_class,
            "seed": a.seed, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "contract": {"floor": f"{IZ_FLOOR_FRAC:g} * max|I|", "r2_min": IZ_R2_MIN,
                         "phi_range_ev": list(IZ_PHI_RANGE_EV), "jump_mad": IZ_JUMP_MAD,
                         "kappa_const_per_nm_per_sqrt_ev": KAPPA_PER_NM_PER_SQRT_EV},
            "material_source": "Comment01 only (overlayer comments rejected); path is not used"}
    iz = run_iz(iz_rows)
    iz["meta"] = meta
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "iz_phi.json").write_text(json.dumps(iz, indent=1), encoding="utf-8")
    print(f"I–Z: {iz['n_files']} files; unlabelled clean {iz['unlabelled']['n_clean']}/"
          f"{iz['unlabelled']['n_files']}  phi p50="
          f"{(iz['unlabelled']['phi_ev'] or {}).get('p50', float('nan')):.2f} eV  "
          f"failed={iz['unlabelled']['why_failed']}", flush=True)
    for mat, s in iz["materials"].items():
        print(f"  {mat}: n={s['n_files']} clean={s['n_clean']} phi={s.get('phi_ev')}")
    sts = run_iv(iv_rows, a.min_n)
    sts["meta"] = meta
    (out_dir / "sts_templates.json").write_text(json.dumps(sts, indent=1), encoding="utf-8")
    for mat, t in sts["materials"].items():
        print(f"STS {mat}: curves={t['n_curves']} covered={t['v_covered']} onset={t['onset']} "
              f"tally={t['tally']}", flush=True)
    print(f"wrote {out_dir / 'iz_phi.json'} and {out_dir / 'sts_templates.json'} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
