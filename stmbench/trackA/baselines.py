"""Track A baselines. Every baseline produces an *answers* table (``sample_id``, ``answer`` JSON)
so it is scored through exactly the same path as a model (``score.py``).

T1 — 继续还是停 (label = the operator stopped the JUDGED frame, the last of ``input_paths``,
of which only the first ``head_rows`` acquired rows may be read)
  * ``drift_rule`` (+ ``_dz`` / ``_dx`` / ``_corr`` single-feature variants): the guide's rule.
    Between two consecutive frames at the same spot the head rows tell whether the image
    is still moving (lateral shift ``dx``), the height still drifting (``dz``) and whether the
    profile has settled (``corr``). Operators let the frame run when it stops moving —
    glance→glance medians 94 pm / 0.39 nm / 0.885 vs last-glance→completed 26 pm / 0.20 nm /
    0.973 (data guide §1, aesthetics README §4.5, ``probe/drift_glance.py``). The rule scores
    the **last** step — from the last earlier frame to the head rows of the judged frame: the
    further it sits on the "still moving" side of the two medians, the higher ``p_stop`` (=
    the judged frame will be stopped too). :func:`paired_final_more_stable` reproduces the
    guide's 69–78 % figure (fraction of episodes whose final step is calmer than the mean of
    the earlier ones).
  * ``step_order`` — a **negative control** that must be reported next to every T1 number: it
    looks at nothing but how many frames have happened so far. If a manifest leaks the
    episode's end (e.g. every last step positive), this "model" scores far from 0.5.
  * ``pixel_head`` — the second negative control (design §5.1 「单帧像素模型负对照 ≈0.5」): a
    single-frame model on the judged frame's head rows alone (16×16 resample of the
    three-step-rendered strip → logistic regression, or nearest-centroid without sklearn),
    trained on the ``train`` split. It sees no sequence; the guide says a single frame does
    not carry the reason for a stop, so it should sit near 0.5 — if it does not, either the
    head rows leak the outcome or the guide is wrong, and both are worth knowing.
  * ``majority`` — constant answer, train-split prevalence as ``p_stop``.

T2 — 下一步去哪:  ``center`` / ``random`` / ``flattest`` (lowest row-aligned roughness window).
T3 — 换地方还是留下:  ``majority`` (three-class by default; ``t3_binary`` folds).
T4 — 工作点先验:  ``material_median`` (per-material median bias and log10 setpoint on the train
  split; unseen material → global median; sigma = 1.4826·MAD for the log-likelihood metric).

Reference numbers in :data:`DRIFT_REF` come from the guide, not from any manifest — they are
a documented prior, not a calibrated threshold (an uncalibrated threshold has no veto; here it
only orders samples for AUROC, and the hard ``stop`` bit is its 0.5 crossing).

Frame references in ``input_paths`` may be relative (manifest ``rel_path``); they are resolved
under :func:`stmsim.paths.raw_mirror_root` by ``render.resolve_path`` when opened.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np

from .render import (
    flatten_three_step, frame_start_time, head_band, load_scan, oriented, plane_fit, resolve_path, row_median_align,
)

K_ROWS = 8
MAX_SHIFT_PX = 60
# (glance→glance median, last-glance→completed median) — data guide §1 / README §4.5
DRIFT_REF: dict[str, tuple[float, float]] = {"dz_pm": (94.0, 26.0), "dx_nm": (0.39, 0.20), "corr": (0.885, 0.973)}
DZ_FLOOR_PM = 1.0
DX_FLOOR_NM = 0.01
PIXEL_GRID = (16, 16)


# ── manifest helpers ────────────────────────────────────────────────────────

def _rows(obj: Any) -> list[dict]:
    if obj is None:
        return []
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        return obj.to_dict("records")
    return [dict(r) for r in obj]


def as_paths(v: Any) -> list[str]:
    """``input_paths`` cell → list of str (list / tuple / ndarray / JSON string / single path)."""
    if v is None:
        return []
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                v = json.loads(s)
            except ValueError:
                return [s]
        else:
            return [s] if s else []
    if hasattr(v, "tolist"):
        v = v.tolist()
    return [str(p) for p in v]


def episode_key(row: dict) -> Any:
    """Samples of one glance episode share their first input path (or an ``episode`` column)."""
    for k in ("episode", "ep", "episode_id"):
        if row.get(k) is not None:
            return row[k]
    paths = as_paths(row.get("input_paths"))
    return paths[0] if paths else row.get("sample_id")


def _label_int(v: Any) -> int:
    if isinstance(v, str):
        return 1 if v.strip().lower() in ("1", "true", "stop", "yes") else 0
    return int(bool(v))


# ── glance-step features (the drift rule's input) ───────────────────────────

@lru_cache(maxsize=4096)
def _frame_head(path: str, k_rows: int) -> dict | None:
    """Small per-frame record: the ``k_rows``-row band at each geometric edge (``top`` = rows
    0..k−1, ``bottom`` = the last k rows — each ``None`` unless every row of that band was
    acquired), its row-aligned mean profile and mean Z, the start time, the pixel scale and
    ``start_side`` — the edge where this frame's acquisition began (``scan_dir`` up → bottom,
    else top), which is exactly the band ``render.head_band`` shows a T1 model.

    Cached so an episode's frames are read once even though every step of the episode is a
    separate manifest sample.
    """
    scan = load_scan(path)
    fr = oriented(scan, "Z")
    a = fr.get("forward")
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    fin = np.isfinite(a).all(axis=1)
    if int(fin.sum()) < k_rows:
        return None
    ny, nx = a.shape
    header = scan.get("header") or {}
    m_per_px = (fr["width_nm"] * 1e-9 / nx) if fr.get("width_nm") else None
    scan_dir = str(header.get("scan_dir", "")).strip().lower() or "down"
    rec: dict[str, Any] = {
        "t": frame_start_time(header),
        "m_per_px": m_per_px,
        "nx": nx, "ny": ny,
        "scan_dir": scan_dir,
        "start_side": "bottom" if scan_dir == "up" else "top",
    }
    for side, ok, band in (("top", bool(fin[:k_rows].all()), a[:k_rows]),
                           ("bottom", bool(fin[-k_rows:].all()), a[-k_rows:])):
        if not ok:
            rec[f"profile_{side}"] = None
            rec[f"zmean_{side}"] = None
            continue
        p = (band - np.median(band, axis=1)[:, None]).mean(axis=0)
        rec[f"profile_{side}"] = p - p.mean()
        rec[f"zmean_{side}"] = float(np.mean(band))
    return rec


def _xshift(p: np.ndarray, q: np.ndarray, max_shift: int) -> tuple[int, float]:
    """Lateral shift (px) maximising the normalised correlation of two head-row profiles."""
    n = p.size
    maxs = int(max(1, min(max_shift, n // 4)))
    best, bs = -2.0, 0
    sl = slice(maxs, n - maxs)
    pp = p[sl]
    npp = np.linalg.norm(pp) + 1e-30
    for s in range(-maxs, maxs + 1):
        qq = np.roll(q, s)[sl]
        c = float(np.dot(pp, qq) / (npp * (np.linalg.norm(qq) + 1e-30)))
        if c > best:
            best, bs = c, s
    return bs, best


def glance_step_features(paths: Iterable[str], *, k_rows: int = K_ROWS, max_shift_px: int = MAX_SHIFT_PX) -> list[dict]:
    """Per consecutive pair of frames: ``dx_px, dx_nm, corr, dz_pm, dt_s, vx_nm_per_min, vz_pm_per_min``.

    Each pair is compared on the **same geometric band**: the later frame's ``start_side``
    (where its acquisition began — its head band, the only part of a judged T1 frame a model
    may see) against the earlier frame's band at that same edge. Of the later frame nothing
    but that band is read, so the last step of a T1 sample is head-only by construction.
    Pairs without a common band (the earlier frame never reached that edge, size mismatch,
    too few rows) are skipped.
    """
    heads = [_frame_head(str(resolve_path(p)), k_rows) for p in paths]
    out: list[dict] = []
    for k in range(1, len(heads)):
        h0, h1 = heads[k - 1], heads[k]
        if h0 is None or h1 is None:
            continue
        side = h1["start_side"]
        p0, p1 = h0.get(f"profile_{side}"), h1.get(f"profile_{side}")
        if p0 is None or p1 is None or p0.size != p1.size:
            continue
        s, c = _xshift(p0, p1, max_shift_px)
        mpp = h1["m_per_px"] or h0["m_per_px"]
        dz_pm = abs(h1[f"zmean_{side}"] - h0[f"zmean_{side}"]) * 1e12
        dt = (h1["t"] - h0["t"]).total_seconds() if (h0["t"] and h1["t"]) else None
        dx_nm = abs(s) * mpp * 1e9 if mpp else None
        feat = {"step": k, "dx_px": abs(int(s)), "dx_nm": dx_nm, "corr": c, "dz_pm": dz_pm, "dt_s": dt,
                "vx_nm_per_min": (dx_nm / max(dt, 1.0) * 60.0) if (dx_nm is not None and dt is not None) else None,
                "vz_pm_per_min": (dz_pm / max(dt, 1.0) * 60.0) if dt is not None else None,
                "side": side}
        out.append(feat)
    return out


# ── the drift rule ──────────────────────────────────────────────────────────

def _instability(feat: dict, key: str) -> float | None:
    """Signed score: 0 at the midpoint of the two reference medians, +1 at the glance→glance median
    (still moving), −1 at the glance→completed median (settled)."""
    v = feat.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    hi, lo = DRIFT_REF[key]
    if key == "corr":
        mid = (hi + lo) / 2.0
        half = (lo - hi) / 2.0
        return float((mid - v) / half)
    v = max(float(v), DZ_FLOOR_PM if key == "dz_pm" else DX_FLOOR_NM)
    mid = math.sqrt(hi * lo)
    half = math.log(hi / mid)
    return float(math.log(v / mid) / half)


def drift_rule_p_stop(feats: list[dict], variant: str = "combined") -> tuple[float | None, dict]:
    """``p_stop`` from the last step (earlier frame → judged frame's head); ``None`` (abstain)
    when there is no step yet."""
    if not feats:
        return None, {}
    last = feats[-1]
    keys = {"combined": ("dz_pm", "dx_nm", "corr"), "dz": ("dz_pm",), "dx": ("dx_nm",), "corr": ("corr",)}[variant]
    parts = [z for z in (_instability(last, k) for k in keys) if z is not None]
    if not parts:
        return None, last
    z = float(np.mean(parts))
    return 1.0 / (1.0 + math.exp(-2.0 * z)), last


def paired_final_more_stable(steps: Any, *, keys: tuple[str, ...] = ("dz_pm", "dx_nm", "corr")) -> dict:
    """The guide's 69–78 % check on a step table (columns ``ep, is_final`` + features).

    For each episode with ≥1 final and ≥1 earlier step: final value vs mean of the earlier steps;
    a step is "more stable" when dz/dx are smaller or corr is larger. Returns per-feature
    ``{frac_final_more_stable, n_episodes, median_diff}``.
    """
    rows = _rows(steps)
    by_ep: dict[Any, dict[str, list]] = {}
    for r in rows:
        d = by_ep.setdefault(r["ep"], {"final": [], "earlier": []})
        d["final" if bool(r.get("is_final")) else "earlier"].append(r)
    out: dict[str, dict] = {}
    for key in keys:
        diffs = []
        for d in by_ep.values():
            f = [float(r[key]) for r in d["final"] if r.get(key) is not None and not _nan(r[key])]
            e = [float(r[key]) for r in d["earlier"] if r.get(key) is not None and not _nan(r[key])]
            if f and e:
                diffs.append(float(np.mean(f) - np.mean(e)))
        if not diffs:
            out[key] = {"frac_final_more_stable": None, "n_episodes": 0, "median_diff": None}
            continue
        arr = np.asarray(diffs)
        better = (arr > 0) if key == "corr" else (arr < 0)
        out[key] = {"frac_final_more_stable": float(better.mean()), "n_episodes": int(arr.size),
                    "median_diff": float(np.median(arr))}
    return out


def _nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _head_rows(row: dict, default: int) -> int:
    """Rows of the judged frame the rule may read (schema ``head_rows``; default the guide's 8)."""
    v = row.get("head_rows")
    try:
        n = int(v) if v is not None and v == v else default
    except (TypeError, ValueError):
        n = default
    return max(2, n)


def step_table_from_manifest(rows: list[dict], *, k_rows: int = K_ROWS) -> list[dict]:
    """Build the ``(ep, step, is_final, features)`` table :func:`paired_final_more_stable` wants
    from T1 manifest rows: each row's last step is final iff its label says the judged frame ran."""
    from .score import normalize_manifest
    out = []
    for r in normalize_manifest(rows, "T1"):
        feats = glance_step_features(as_paths(r.get("input_paths")), k_rows=_head_rows(r, k_rows))
        if not feats:
            continue
        last = dict(feats[-1])
        last.update(ep=episode_key(r), is_final=(_label_int(r["label"]) == 0), sample_id=r["sample_id"])
        out.append(last)
    return out


# ── T1 baselines ────────────────────────────────────────────────────────────

def _t1_drift(rows: list[dict], train: list[dict], rng: np.random.Generator, *, variant: str = "combined",
              k_rows: int = K_ROWS, **_: Any) -> list[dict]:
    out = []
    for r in rows:
        feats = glance_step_features(as_paths(r.get("input_paths")), k_rows=_head_rows(r, k_rows))
        p, last = drift_rule_p_stop(feats, variant)
        if p is None:
            ans = {"abstain": True, "reason": "fewer than two comparable frames", "n_steps": len(feats)}
        else:
            ans = {"stop": p >= 0.5, "p_stop": p, "n_steps": len(feats),
                   "features": {k: last.get(k) for k in ("dx_nm", "dz_pm", "corr", "dt_s")}}
        out.append({"sample_id": r["sample_id"], "answer": json.dumps(ans)})
    return out


def _t1_step_order(rows: list[dict], train: list[dict], rng: np.random.Generator, **_: Any) -> list[dict]:
    out = []
    for r in rows:
        n = len(as_paths(r.get("input_paths")))
        p = 1.0 / (1.0 + n)  # the more frames so far, the "closer to the end" — knows nothing else
        out.append({"sample_id": r["sample_id"], "answer": json.dumps({"stop": p >= 0.5, "p_stop": p, "n_glances": n,
                                                                       "negative_control": True})})
    return out


# ── the single-frame pixel model (negative control) ─────────────────────────

def resample_grid(a: np.ndarray, shape: tuple[int, int] = PIXEL_GRID) -> np.ndarray:
    """Block-mean resample of a 2-D array onto ``shape`` (bins may overlap when the source is
    smaller than the target along an axis, e.g. 8 head rows → 16). NaN counts as 0."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"expected a non-empty 2-D array, got shape {a.shape}")
    a = np.where(np.isfinite(a), a, 0.0)
    ny, nx = a.shape
    ty, tx = shape
    ey = np.linspace(0, ny, ty + 1)
    ex = np.linspace(0, nx, tx + 1)
    out = np.empty((ty, tx), dtype=float)
    for j in range(ty):
        j0 = int(math.floor(ey[j]))
        j1 = max(j0 + 1, int(math.ceil(ey[j + 1])))
        for i in range(tx):
            i0 = int(math.floor(ex[i]))
            i1 = max(i0 + 1, int(math.ceil(ex[i + 1])))
            out[j, i] = a[j0:j1, i0:i1].mean()
    return out


def pixel_head_features(strip: np.ndarray, grid: tuple[int, int] = PIXEL_GRID) -> np.ndarray:
    """Feature vector of a head strip: the three-step rendering (what a model would see) resampled
    to ``grid`` and flattened. Nothing but the strip's own pixels enters."""
    img, _ = flatten_three_step(np.asarray(strip, dtype=float))
    return resample_grid(img, grid).ravel()


class PixelHeadModel:
    """Binary classifier on standardised pixel features: scikit-learn ``LogisticRegression``
    when importable (``kind="logreg"``), else a nearest-centroid rule (``kind="centroid"``:
    ``p = d0 / (d0 + d1)``, the relative closeness to the positive centroid). ``kind="auto"``
    picks the first available; a named kind is honoured or raises."""

    def __init__(self, kind: str = "auto", C: float = 1.0) -> None:
        self.kind_requested = kind
        self.C = C
        self.kind: str | None = None
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.n_train = 0
        self._clf = None
        self._centroids: tuple[np.ndarray, np.ndarray] | None = None

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mu) / self.sd

    def fit(self, X: Any, y: Any) -> "PixelHeadModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray([_label_int(v) for v in y], dtype=int)
        if X.ndim != 2 or X.shape[0] != y.size:
            raise ValueError(f"X {X.shape} and y {y.shape} disagree")
        if y.size < 2 or len(set(y.tolist())) < 2:
            raise ValueError("need at least one sample of each class to fit")
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Z = self._z(X)
        self.n_train = int(y.size)
        if self.kind_requested in ("auto", "logreg"):
            try:
                from sklearn.linear_model import LogisticRegression
            except ImportError:
                if self.kind_requested == "logreg":
                    raise
            else:
                self._clf = LogisticRegression(C=self.C, max_iter=1000)
                self._clf.fit(Z, y)
                self.kind = "logreg"
                return self
        if self.kind_requested not in ("auto", "centroid"):
            raise ValueError(f"unknown model kind {self.kind_requested!r}")
        self._centroids = (Z[y == 0].mean(axis=0), Z[y == 1].mean(axis=0))
        self.kind = "centroid"
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """P(label = 1) per row."""
        if self.kind is None:
            raise RuntimeError("fit first")
        Z = self._z(np.atleast_2d(np.asarray(X, dtype=float)))
        if self.kind == "logreg":
            return np.asarray(self._clf.predict_proba(Z)[:, 1], dtype=float)
        c0, c1 = self._centroids
        d0 = np.linalg.norm(Z - c0, axis=1)
        d1 = np.linalg.norm(Z - c1, axis=1)
        den = d0 + d1
        return np.where(den > 0, d0 / np.where(den > 0, den, 1.0), 0.5)


@lru_cache(maxsize=4096)
def _head_strip(path: str, k_rows: int) -> np.ndarray | None:
    """The judged frame's head band (``render.head_band``: the first ``k_rows`` acquired rows,
    from the edge its scan started at) — the only pixels the pixel model ever reads."""
    scan = load_scan(path)
    fr = oriented(scan, "Z")
    a = fr.get("forward")
    if a is None:
        return None
    header = scan.get("header") or {}
    strip = head_band(np.asarray(a, dtype=float), header.get("scan_dir"), int(k_rows))
    return None if strip.shape[0] < k_rows else strip


def _t1_pixel_head(rows: list[dict], train: list[dict], rng: np.random.Generator, *, k_rows: int = K_ROWS,
                   grid: tuple[int, int] = PIXEL_GRID, model_kind: str = "auto", **_: Any) -> list[dict]:
    """See the module docstring. Trained on ``train`` (never on the target rows); every target
    row is answered from its own head strip alone."""
    def _features(r: dict) -> np.ndarray | None:
        paths = as_paths(r.get("input_paths"))
        if not paths:
            return None
        try:
            strip = _head_strip(str(resolve_path(paths[-1])), _head_rows(r, k_rows))
        except Exception:  # noqa: BLE001 — unreadable frame → no features
            return None
        return None if strip is None else pixel_head_features(strip, grid)

    X, y = [], []
    for r in train:
        f = _features(r)
        if f is not None:
            X.append(f)
            y.append(_label_int(r["label"]))
    model: PixelHeadModel | None = None
    why = ""
    if len(y) < 2 or len(set(y)) < 2:
        why = f"pixel model untrainable: {len(y)} readable train rows, classes {sorted(set(y))}"
    else:
        model = PixelHeadModel(model_kind).fit(np.vstack(X), y)
    out = []
    for r in rows:
        if model is None:
            ans: dict[str, Any] = {"abstain": True, "reason": why, "negative_control": True}
        else:
            f = _features(r)
            if f is None:
                ans = {"abstain": True, "reason": "judged frame unreadable or shorter than head_rows",
                       "negative_control": True}
            else:
                p = float(model.predict_proba(f[None, :])[0])
                ans = {"stop": p >= 0.5, "p_stop": p, "negative_control": True, "model": model.kind,
                       "n_train": model.n_train, "grid": list(grid)}
        out.append({"sample_id": r["sample_id"], "answer": json.dumps(ans)})
    return out


# ── T2 / T3 / T4 baselines ──────────────────────────────────────────────────

def _majority(rows: list[dict], train: list[dict], rng: np.random.Generator, *, task: str = "T1", **_: Any) -> list[dict]:
    src = train or rows
    if task == "T1":
        prev = float(np.mean([_label_int(r["label"]) for r in src])) if src else 0.5
        ans = {"stop": prev >= 0.5, "p_stop": prev, "majority_of": "train" if train else "target"}
    else:
        labels = [str(r["label"]).strip().lower() for r in src]
        cnt = Counter(labels)
        top = cnt.most_common(1)[0][0] if cnt else "stay"
        p_rel = cnt.get("relocate", 0) / len(labels) if labels else 0.5
        ans = {"action": top, "p_relocate": p_rel, "probs": {k: v / len(labels) for k, v in cnt.items()},
               "majority_of": "train" if train else "target"}
    return [{"sample_id": r["sample_id"], "answer": json.dumps(ans)} for r in rows]


def _t2_range_nm(r: dict, default: float = 100.0) -> float:
    for k in ("range_nm", "size_nm", "width_nm"):
        v = r.get(k)
        if v is not None and not _nan(v) and float(v) > 0:
            return float(v)
    paths = as_paths(r.get("input_paths"))
    if paths:
        try:
            from .render import load_header
            w = str(load_header(paths[-1]).get("scan_range", "")).split()
            if w:
                return float(w[0]) * 1e9
        except Exception:  # noqa: BLE001
            pass
    return default


def _t2_height_nm(r: dict, width_nm: float) -> float:
    """The frame's height: manifest ``height_nm`` when present, else the width (square)."""
    for k in ("height_nm", "range_y_nm"):
        v = r.get(k)
        if v is not None and not _nan(v) and float(v) > 0:
            return float(v)
    return width_nm


def _t2_box_nm(r: dict, rng_nm: float, box_nm: float | None) -> float:
    for k in ("box_nm", "next_range_nm", "next_size_nm"):
        v = r.get(k)
        if v is not None and not _nan(v) and float(v) > 0:
            return float(v)
    return box_nm if box_nm else rng_nm / 4.0


def _t2_center(rows: list[dict], train: list[dict], rng: np.random.Generator, *, box_nm: float | None = None, **_: Any) -> list[dict]:
    out = []
    for r in rows:
        w = _t2_range_nm(r)
        out.append({"sample_id": r["sample_id"],
                    "answer": json.dumps({"dx_nm": 0.0, "dy_nm": 0.0, "size_nm": _t2_box_nm(r, w, box_nm), "candidates": [[0.0, 0.0]]})})
    return out


def _t2_random(rows: list[dict], train: list[dict], rng: np.random.Generator, *, box_nm: float | None = None,
               n_candidates: int = 5, **_: Any) -> list[dict]:
    out = []
    for r in rows:
        w = _t2_range_nm(r)
        h = _t2_height_nm(r, w)
        b = _t2_box_nm(r, w, box_nm)
        half_x = max(0.0, (w - b) / 2.0)
        half_y = max(0.0, (h - b) / 2.0)
        cx = rng.uniform(-half_x, half_x, size=n_candidates)
        cy = rng.uniform(-half_y, half_y, size=n_candidates)
        c = np.column_stack([cx, cy]).round(4).tolist()
        out.append({"sample_id": r["sample_id"],
                    "answer": json.dumps({"dx_nm": c[0][0], "dy_nm": c[0][1], "size_nm": b, "candidates": c})})
    return out


def flattest_windows(a: np.ndarray, *, box_px: int, stride_px: int | None = None, n_best: int = 5) -> list[tuple[float, float, float]]:
    """Lowest-roughness square windows of a top-first frame → ``[(col_centre, row_centre, roughness)]``.

    Roughness = std of the plane-fitted, row-aligned window (NaN rows ignored; windows with <50 %
    finite pixels skipped). Coordinates are pixel centres, row 0 at the top.
    """
    a = row_median_align(plane_fit(np.asarray(a, dtype=float)))
    ny, nx = a.shape
    box = int(max(2, min(box_px, nx, ny)))
    stride = int(max(1, stride_px or box // 2))
    cands = []
    for j in range(0, ny - box + 1, stride):
        for i in range(0, nx - box + 1, stride):
            win = a[j:j + box, i:i + box]
            fin = np.isfinite(win)
            if fin.mean() < 0.5:
                continue
            cands.append((i + box / 2.0, j + box / 2.0, float(np.std(win[fin]))))
    cands.sort(key=lambda t: t[2])
    return cands[:n_best]


def _t2_flattest(rows: list[dict], train: list[dict], rng: np.random.Generator, *, box_nm: float | None = None,
                 n_candidates: int = 5, **_: Any) -> list[dict]:
    out = []
    for r in rows:
        paths = as_paths(r.get("input_paths"))
        w = _t2_range_nm(r)
        b = _t2_box_nm(r, w, box_nm)
        ans: dict[str, Any]
        try:
            fr = oriented(load_scan(paths[-1]), "Z")
            a = np.asarray(fr["forward"], dtype=float)
            ny, nx = a.shape
            nm_px = w / nx
            h_nm = _t2_height_nm(r, fr.get("height_nm") or w)
            wins = flattest_windows(a, box_px=int(round(b / nm_px)), n_best=n_candidates)
            if not wins:
                raise ValueError("no window with enough acquired pixels")
            cands = [[round((ci - nx / 2.0) * nm_px, 4), round((ny / 2.0 - cj) * (h_nm / ny), 4)] for ci, cj, _ in wins]
            ans = {"dx_nm": cands[0][0], "dy_nm": cands[0][1], "size_nm": b, "candidates": cands,
                   "roughness_pm": [round(s * 1e12, 2) for _, _, s in wins]}
        except Exception as e:  # noqa: BLE001 — unreadable frame → abstain, never a silent (0, 0)
            ans = {"abstain": True, "reason": f"frame unreadable: {type(e).__name__}: {e}"[:200]}
        out.append({"sample_id": r["sample_id"], "answer": json.dumps(ans)})
    return out


def _t4_label(v: Any) -> tuple[float, float | None, float | None] | None:
    """``(bias_v, setpoint_a | None, speed_nm_s | None)`` — setpoint/speed are optional labels."""
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return None

    def f(x: Any) -> float | None:
        try:
            y = float(x)
        except (TypeError, ValueError):
            return None
        return None if (math.isnan(y) or y == 0) else y

    try:
        if isinstance(v, dict):
            b = f(v.get("bias_v"))
            return (b, f(v.get("setpoint_a")), f(v.get("speed_nm_s"))) if b is not None else None
        arr = np.asarray(v, dtype=float).ravel()
        b = f(arr[0])
        return (b, f(arr[1]) if arr.size > 1 else None, f(arr[2]) if arr.size > 2 else None) if b is not None else None
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _t4_material_median(rows: list[dict], train: list[dict], rng: np.random.Generator, *, material_key: str = "material",
                        **_: Any) -> list[dict]:
    src = train or rows
    groups: dict[Any, list[tuple[float, float | None, float | None]]] = {}
    for r in src:
        lab = _t4_label(r.get("label"))
        if lab is None:
            continue
        groups.setdefault(r.get(material_key), []).append(lab)

    def _stats(labs: list[tuple[float, float | None, float | None]]) -> dict:
        b = np.asarray([x for x, _, _ in labs])
        mb = float(np.median(b))
        sb = max(1.4826 * float(np.median(np.abs(b - mb))), 0.02)
        out = {"bias_v": mb, "bias_sigma_v": sb, "n_train": int(b.size)}
        s = np.log10(np.abs(np.asarray([y for _, y, _ in labs if y is not None])))
        if s.size:
            ms = float(np.median(s))
            out.update(setpoint_a=10.0 ** ms, setpoint_log10_sigma=max(1.4826 * float(np.median(np.abs(s - ms))), 0.1),
                       n_train_setpoint=int(s.size))
        sp = np.asarray([z for _, _, z in labs if z is not None])
        if sp.size:
            out["scan_speed_nm_per_s"] = float(np.median(sp))
        return out

    per = {k: _stats(v) for k, v in groups.items() if v}
    all_labs = [lab for v in groups.values() for lab in v]
    glob = _stats(all_labs) if all_labs else None
    out = []
    for r in rows:
        st = per.get(r.get(material_key)) or glob
        if st is None:
            ans = {"abstain": True, "reason": "no training labels"}
        else:
            ans = dict(st, material=str(r.get(material_key)), fallback=(r.get(material_key) not in per))
        out.append({"sample_id": r["sample_id"], "answer": json.dumps(ans)})
    return out


BASELINES: dict[str, tuple[str, Callable[..., list[dict]], dict]] = {
    "drift_rule": ("T1", _t1_drift, {"variant": "combined"}),
    "drift_rule_dz": ("T1", _t1_drift, {"variant": "dz"}),
    "drift_rule_dx": ("T1", _t1_drift, {"variant": "dx"}),
    "drift_rule_corr": ("T1", _t1_drift, {"variant": "corr"}),
    "step_order": ("T1", _t1_step_order, {}),
    "pixel_head": ("T1", _t1_pixel_head, {}),
    "majority": ("T1|T3", _majority, {}),
    "center": ("T2", _t2_center, {}),
    "random": ("T2", _t2_random, {}),
    "flattest": ("T2", _t2_flattest, {}),
    "material_median": ("T4", _t4_material_median, {}),
}

# reported next to every T1 number: what a model that cannot see the sequence scores
NEGATIVE_CONTROLS = ("step_order", "pixel_head")


def baselines_for(task: str) -> list[str]:
    task = str(task).upper()
    return [n for n, (t, _, _) in BASELINES.items() if task in t.split("|")]


def run_baseline(name: str, manifest: Any, *, task: str | None = None, split: str | None = "test",
                 seed: int = 0, t3_binary: bool = False, **kw: Any) -> list[dict]:
    """Answers (``sample_id``, ``answer`` JSON string) for the rows of ``manifest`` in ``split``.

    Train-split rows (``split == "train"``) feed the data-driven baselines (majority, material
    median, pixel_head); the target rows never do. ``t3_binary`` folds T3 labels the way
    ``score.score_t3(t3_binary=True)`` does, so a train-fitted baseline sees the same classes.
    """
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    tasks, fn, defaults = BASELINES[name]
    from .score import normalize_manifest
    rows_all = normalize_manifest(manifest, task or (tasks if "|" not in tasks else None), t3_binary=t3_binary)
    if task is None:
        present = {str(r.get("task", "")).upper() for r in rows_all} - {""}
        cands = [t for t in tasks.split("|") if not present or t in present]
        if len(cands) != 1:
            raise ValueError(f"baseline {name!r} serves {tasks}; pass task= to pick one (manifest has {sorted(present)})")
        task = cands[0]
    task = task.upper()
    if task not in tasks.split("|"):
        raise ValueError(f"baseline {name!r} is for {tasks}, not {task}")
    rows_task = [r for r in rows_all if r.get("task") is None or str(r["task"]).upper() == task]
    target = [r for r in rows_task if not split or str(r.get("split", "")) == split]
    train = [r for r in rows_task if str(r.get("split", "")) == "train"]
    rng = np.random.default_rng(seed)
    params = {**defaults, "task": task, **kw}
    return fn(target, train, rng, **params)
