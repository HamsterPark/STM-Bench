"""Track A scoring. Models and baselines are scored the same way: from an *answers* table
``(sample_id, answer)`` where ``answer`` is the JSON the model produced (see ``prompts``).

Metrics (design §P6 / data guide §5):

* **T1** AUROC of ``p_stop`` against ``label`` (1 = the operator stopped THIS frame — the
  one shown head-only, the last of ``input_paths`` — before it completed; the manifest's
  ``stopped``), pooled and **by group** (instrument-day); report the mean over groups that
  contain both classes. Accuracy of the hard ``stop`` decision is reported alongside.
* **T2** IoU of the predicted box vs the operator's next frame, centre distance normalised by
  the shown frame's width, and top-k hit (any of the first k candidates lands inside the
  target box). The target box is ``next_w_rel × width`` by ``next_h_rel × height`` — a
  non-square frame scales ``dy`` and the box height by its own height.
* **T3** three-class (stay / relocate / long_stop, the manifest's ``next_action``) accuracy,
  per-class recall, one-vs-rest AUROC per class from ``probs`` and its macro mean. The
  binary fold (``t3_binary=True``: relocate + long_stop → ``relocate``, folded on both the
  label and the answer) keeps the old ``auroc_relocate`` reading.
* **T4** tolerance hit-rate (bias: ``max(abs_tol, rel_tol·|label|)``; setpoint: factor-of-two in
  log10) and Gaussian log-likelihood (bias in V, setpoint in log10 A) with the model's own
  ``sigma`` or a default width.
* **判不了** is scored separately: ``abstain_rate`` = abstentions / answered; abstained samples
  are excluded from every other metric. A sample with **no** answer row is ``missing`` (also
  reported, also excluded) — silence is not an abstention.

AUROC uses scikit-learn when importable, else the hand-rolled rank AUROC below (ties averaged);
the two agree to 1e-12 on the same input.

Labels are coerced leniently (:func:`coerce_label`): a manifest may store a T2 list as a
JSON string or ndarray, a T4 dict as a JSON string, a T1 int as bool/float.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from . import schema as SCHEMA
from .prompts import parse_answer

# ── helpers ─────────────────────────────────────────────────────────────────

def _rows(obj: Any) -> list[dict]:
    """DataFrame / iterable-of-dicts → list of dicts."""
    if obj is None:
        return []
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        return obj.to_dict("records")
    return [dict(r) for r in obj]


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


# ── manifest adapter ────────────────────────────────────────────────────────
# Two manifest shapes are accepted everywhere (score, baselines, cli):
#   * the contract: sample_id, task, input_paths, label, group, split (+ strat columns);
#   * the built manifests of ``stmbench.trackA.manifests`` / ``schema``: per-task label
#     columns (T1 ``stopped``, T2 ``next_u/next_v/next_w_rel/next_h_rel``, T3
#     ``next_action``/``relocate``, T4 ``bias_v/speed_nm_s/setpoint_a``), ``rel_path`` = this
#     frame (relative to stmsim.paths.raw_mirror_root(); the absolute ``path`` is meta and
#     only used when rel_path is absent), ``context_paths_json`` = earlier frames (oldest
#     first, same convention), ``group_key``.
# ``normalize_manifest`` maps the second onto the first; every other column is kept. The
# relative references are resolved when a frame is opened (``render.resolve_path``).

_TASK_BY_LABEL_COLUMN = (("T1", "stopped"), ("T2", "next_u"), ("T3", "next_action"), ("T4", "speed_nm_s"))
T3_CLASSES = tuple(SCHEMA.NEXT_ACTIONS)                # ("stay", "relocate", "long_stop")
T3_BINARY_CLASSES = ("stay", "relocate")
_T3_NOT_STAY = tuple(c for c in T3_CLASSES if c != "stay")


def _json_list(v: Any) -> list:
    if v is None or _is_nan(v):
        return []
    if isinstance(v, (bytes, bytearray)):
        v = v.decode("utf-8", "replace")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return [v] if v.strip() else []
    if hasattr(v, "tolist"):
        v = v.tolist()
    return list(v) if isinstance(v, (list, tuple)) else [v]


def infer_task(columns: Iterable[str]) -> str | None:
    cols = set(columns)
    for task, col in _TASK_BY_LABEL_COLUMN:
        if col in cols:
            return task
    return None


def _frame_ref(r: dict) -> str:
    """The reference a consumer opens for this frame: ``rel_path`` (relative to the raw
    mirror root) when the manifest carries one, else the absolute ``path``."""
    rel = r.get("rel_path")
    if rel is not None and not _is_nan(rel) and str(rel).strip():
        return str(rel)
    return str(r["path"])


def fold_t3_label(label: Any) -> str:
    """Three-class → binary: everything that is not ``stay`` is ``relocate``."""
    s = str(label).strip().lower()
    return "stay" if s == "stay" else "relocate"


def normalize_manifest(obj: Any, task: str | None = None, *, t3_binary: bool = False) -> list[dict]:
    """Rows in the contract shape, whatever shape ``obj`` came in. See the note above.

    T3's label is the three-class ``next_action`` by default; ``t3_binary=True`` folds it to
    stay / relocate (the manifest's ``relocate`` column, or :func:`fold_t3_label`).
    """
    rows = _rows(obj)
    if not rows:
        return []
    if "label" in rows[0] and "input_paths" in rows[0]:
        if t3_binary:
            task_ = (task or str(rows[0].get("task") or "")).upper()
            if task_ == "T3":
                rows = [dict(r, label=fold_t3_label(r["label"])) for r in rows]
        return rows
    task = (task or infer_task(rows[0].keys()) or "").upper()
    if task not in ("T1", "T2", "T3", "T4"):
        raise ValueError("cannot infer the task from the manifest columns; pass task=")
    out = []
    for r in rows:
        n = dict(r)
        n.setdefault("task", task)
        if n.get("group") is None:
            n["group"] = r.get("group_key")
        ctx = [str(p) for p in _json_list(r.get("context_paths_json"))]
        if task == "T1":
            n["input_paths"] = ctx + [_frame_ref(r)]
            n["label"] = int(bool(r["stopped"]))
        elif task == "T2":
            size = _float(r.get("size_nm")) or 0.0
            height = _float(r.get("height_nm")) or _float(r.get("range_y_nm")) or size
            n["input_paths"] = ctx + [_frame_ref(r)]
            n["label"] = [float(r["next_u"]) * size, float(r["next_v"]) * height]
            n.setdefault("range_nm", size)
            n.setdefault("height_nm", height)
            w_rel, h_rel = _float(r.get("next_w_rel")), _float(r.get("next_h_rel"))
            if w_rel and size and n.get("box_nm") is None:
                n["box_nm"] = w_rel * size
            if h_rel and height and n.get("box_h_nm") is None:
                n["box_h_nm"] = h_rel * height
        elif task == "T3":
            n["input_paths"] = ctx + [_frame_ref(r)]
            if t3_binary:
                rel = r.get("relocate")
                if rel is not None and not _is_nan(rel):
                    n["label"] = "relocate" if bool(rel) else "stay"
                else:
                    n["label"] = fold_t3_label(r["next_action"])
            else:
                n["label"] = str(r["next_action"]).strip().lower()
        else:
            n["input_paths"] = []  # a text-only prior: the frame itself would carry the answer
            n["label"] = {"bias_v": _float(r.get("bias_v")), "setpoint_a": _float(r.get("setpoint_a")),
                          "speed_nm_s": _float(r.get("speed_nm_s"))}
        out.append(n)
    return out


def coerce_label(task: str, label: Any) -> Any:
    task = str(task).upper()
    if isinstance(label, (bytes, bytearray)):
        label = label.decode("utf-8", "replace")
    if isinstance(label, str):
        s = label.strip()
        if task in ("T2", "T4") or s[:1] in "[{":
            try:
                label = json.loads(s)
            except ValueError:
                pass
    if task == "T1":
        if isinstance(label, str):
            s = label.strip().lower()
            return 1 if s in ("1", "true", "stop", "yes") else 0
        return int(bool(label))
    if task == "T2":
        arr = np.asarray(label, dtype=float).ravel()
        if arr.size < 2:
            raise ValueError(f"T2 label needs [dx_nm, dy_nm], got {label!r}")
        return [float(arr[0]), float(arr[1])]
    if task == "T3":
        return str(label).strip().lower()
    if task == "T4":
        if isinstance(label, dict):
            return {"bias_v": _float(label.get("bias_v")), "setpoint_a": _float(label.get("setpoint_a")),
                    "speed_nm_s": _float(label.get("speed_nm_s"))}
        arr = np.asarray(label, dtype=float).ravel()
        return {"bias_v": _float(arr[0]), "setpoint_a": _float(arr[1]) if arr.size > 1 else None,
                "speed_nm_s": _float(arr[2]) if arr.size > 2 else None}
    raise ValueError(f"unknown task {task!r}")


def answers_by_id(answers: Any) -> dict[Any, dict]:
    out: dict[Any, dict] = {}
    for r in _rows(answers):
        # keyed by str: a parquet manifest may carry int ids while an answers table carries strings
        out[str(r["sample_id"])] = parse_answer(r.get("answer"))
    return out


def _float(v: Any, default: float | None = None) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


# ── AUROC ───────────────────────────────────────────────────────────────────

def rank_auroc(y: Iterable[Any], s: Iterable[Any]) -> float | None:
    """Mann–Whitney AUROC with average ranks for ties. ``None`` unless both classes present."""
    y = np.asarray([int(bool(v)) for v in y])
    s = np.asarray([float(v) for v in s], dtype=float)
    if y.size == 0:
        return None
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average rank, 1-based
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc(y: Iterable[Any], s: Iterable[Any]) -> float | None:
    y = list(y)
    s = list(s)
    if len(set(int(bool(v)) for v in y)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score([int(bool(v)) for v in y], [float(v) for v in s]))
    except Exception:  # noqa: BLE001 — sklearn absent or unhappy → hand-rolled
        return rank_auroc(y, s)


# ── common bookkeeping ──────────────────────────────────────────────────────

def _split_rows(task: str, manifest: Any, answers: Any, split: str | None, *,
                t3_binary: bool = False) -> tuple[list[dict], dict, dict]:
    """→ (scored rows with 'label'/'ans', abstain counts, missing counts)."""
    ans_map = answers_by_id(answers)
    rows = []
    n_total = n_abstain = n_missing = 0
    for r in normalize_manifest(manifest, task, t3_binary=t3_binary):
        if split and str(r.get("split", "")) != split:
            continue
        if r.get("task") is not None and str(r["task"]).upper() != task:
            continue
        n_total += 1
        a = ans_map.get(str(r["sample_id"]))
        if a is None:
            n_missing += 1
            continue
        if a.get("abstain"):
            n_abstain += 1
            continue
        rr = dict(r)
        rr["label"] = coerce_label(task, r["label"])
        rr["ans"] = a
        rows.append(rr)
    n_answered = n_total - n_missing
    counts = {"n_total": n_total, "n_answered": n_answered, "n_abstain": n_abstain,
              "abstain_rate": (n_abstain / n_answered) if n_answered else None,
              "n_missing": n_missing, "n_scored": len(rows)}
    return rows, counts, ans_map


def _by_group(rows: list[dict], key: str) -> dict[Any, list[dict]]:
    g: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        g[r.get(key)].append(r)
    return dict(g)


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not _is_nan(x)]
    return float(np.mean(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not _is_nan(x)]
    return float(np.median(xs)) if xs else None


# ── T1 ──────────────────────────────────────────────────────────────────────

def _p_stop(ans: dict) -> float | None:
    p = _float(ans.get("p_stop"))
    if p is None and "stop" in ans:
        p = 1.0 if ans["stop"] in (True, 1, "true", "True") else 0.0
    return p


def score_t1(manifest: Any, answers: Any, *, split: str | None = "test", group_key: str = "group") -> dict:
    """Continue vs stop. ``label`` = 1 when the operator stopped the judged frame (the last of
    ``input_paths``, shown head-only) before it completed; ``p_stop`` is scored against it."""
    rows, counts, _ = _split_rows("T1", manifest, answers, split)
    out: dict[str, Any] = dict(task="T1", **counts)
    scored = [(r["label"], _p_stop(r["ans"]), r) for r in rows if _p_stop(r["ans"]) is not None]
    out["n_with_score"] = len(scored)
    out["auroc_pooled"] = auroc([y for y, _, _ in scored], [p for _, p, _ in scored]) if scored else None
    by = {}
    for g, rs in _by_group([r for _, _, r in scored], group_key).items():
        by[str(g)] = auroc([r["label"] for r in rs], [_p_stop(r["ans"]) for r in rs])
    out["auroc_by_group"] = by
    valid = [v for v in by.values() if v is not None]
    out["auroc_group_mean"] = _mean(valid)
    out["n_groups_scored"] = len(valid)
    hard = [(r["label"], r["ans"]) for r in rows if "stop" in r["ans"]]
    if hard:
        out["accuracy"] = float(np.mean([int(a["stop"] in (True, 1, "true", "True")) == y for y, a in hard]))
    out["prevalence"] = _mean([float(r["label"]) for r in rows])
    return out


# ── T2 ──────────────────────────────────────────────────────────────────────

def iou_boxes(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two axis-aligned boxes given as (cx, cy, w, h)."""
    ax0, ax1 = a[0] - a[2] / 2, a[0] + a[2] / 2
    ay0, ay1 = a[1] - a[3] / 2, a[1] + a[3] / 2
    bx0, bx1 = b[0] - b[2] / 2, b[0] + b[2] / 2
    by0, by1 = b[1] - b[3] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _frame_range_nm(row: dict, default: float) -> float:
    for k in ("range_nm", "size_nm", "width_nm"):
        v = _float(row.get(k))
        if v:
            return v
    paths = row.get("input_paths")
    try:
        from .render import load_header
        p = list(paths)[-1] if paths is not None and len(paths) else None
        if p:
            w = str(load_header(p).get("scan_range", "")).split()
            if w:
                return float(w[0]) * 1e9
    except Exception:  # noqa: BLE001 — header unreadable → default
        pass
    return default


def _target_box_nm(row: dict, range_nm: float, box_nm: float | None) -> float:
    for k in ("box_nm", "next_range_nm", "next_size_nm"):
        v = _float(row.get(k))
        if v:
            return v
    return box_nm if box_nm else range_nm / 4.0


def _candidates(ans: dict) -> list[tuple[float, float]]:
    cands: list[tuple[float, float]] = []
    dx, dy = _float(ans.get("dx_nm")), _float(ans.get("dy_nm"))
    if dx is not None and dy is not None:
        cands.append((dx, dy))
    for c in ans.get("candidates") or []:
        try:
            cx, cy = float(c[0]), float(c[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not cands or (cx, cy) != cands[0]:
            cands.append((cx, cy))
    return cands


def score_t2(manifest: Any, answers: Any, *, split: str | None = "test", group_key: str = "group",
             box_nm: float | None = None, default_range_nm: float = 100.0, top_k: tuple[int, ...] = (1, 3, 5)) -> dict:
    rows, counts, _ = _split_rows("T2", manifest, answers, split)
    out: dict[str, Any] = dict(task="T2", **counts)
    ious, dists, hits = [], [], {k: [] for k in top_k}
    per_group: dict[str, list[float]] = defaultdict(list)
    n_geom = 0
    for r in rows:
        cands = _candidates(r["ans"])
        if not cands:
            continue
        n_geom += 1
        rng = _frame_range_nm(r, default_range_nm)
        tb = _target_box_nm(r, rng, box_nm)
        th = _float(r.get("box_h_nm")) or tb          # the operator's next frame may not be square
        pb = _float(r["ans"].get("size_nm")) or tb
        lx, ly = r["label"]
        px, py = cands[0]
        iou = iou_boxes((px, py, pb, pb), (lx, ly, tb, th))
        dist = math.hypot(px - lx, py - ly) / rng if rng else float("nan")
        ious.append(iou)
        dists.append(dist)
        per_group[str(r.get(group_key))].append(iou)
        for k in top_k:
            hit = any(abs(cx - lx) <= tb / 2 and abs(cy - ly) <= th / 2 for cx, cy in cands[:k])
            hits[k].append(1.0 if hit else 0.0)
    out["n_with_geometry"] = n_geom
    out["mean_iou"] = _mean(ious)
    out["median_center_dist_norm"] = _median(dists)
    out["mean_center_dist_norm"] = _mean(dists)
    for k in top_k:
        out[f"hit@{k}"] = _mean(hits[k])
    out["iou_by_group"] = {g: _mean(v) for g, v in per_group.items()}
    return out


# ── T3 ──────────────────────────────────────────────────────────────────────

def _action(ans: dict) -> str:
    return str(ans.get("action", "") or "").strip().lower()


def fold_t3_answer(ans: dict) -> dict:
    """A three-class answer folded to stay / relocate: ``action`` long_stop → relocate,
    ``probs.relocate`` = 1 − ``probs.stay`` (or relocate + long_stop), ``p_relocate`` likewise.
    An answer that was binary already passes through unchanged."""
    a = dict(ans)
    if _action(a) in _T3_NOT_STAY:
        a["action"] = "relocate"
    probs = a.get("probs") if isinstance(a.get("probs"), dict) else None
    if probs:
        p_stay = _float(probs.get("stay"))
        others = [_float(probs.get(c)) for c in _T3_NOT_STAY]
        if p_stay is not None:
            p_rel = max(0.0, min(1.0, 1.0 - p_stay))
        elif any(p is not None for p in others):
            p_rel = float(sum(p for p in others if p is not None))
        else:
            p_rel = _float(a.get("p_relocate"))
        if p_rel is not None:
            a["probs"] = {"stay": 1.0 - p_rel, "relocate": p_rel}
            a["p_relocate"] = p_rel
    return a


def _t3_class_prob(ans: dict, c: str) -> float | None:
    """Score for class ``c``: ``probs[c]``; for relocate also ``p_relocate``; else the hard action."""
    probs = ans.get("probs") if isinstance(ans.get("probs"), dict) else None
    p = _float(probs.get(c)) if probs else None
    if p is None and c == "relocate":
        p = _float(ans.get("p_relocate"))
    if p is None and _action(ans):
        p = 1.0 if _action(ans) == c else 0.0
    return p


def _p_relocate(ans: dict) -> float | None:
    return _t3_class_prob(ans, "relocate")


def score_t3(manifest: Any, answers: Any, *, split: str | None = "test", group_key: str = "group",
             t3_binary: bool = False) -> dict:
    """Stay / relocate / long_stop. Three-class by default: accuracy, per-class recall, a
    confusion table, one-vs-rest AUROC per class and their macro mean. ``t3_binary=True``
    folds label (relocate + long_stop → relocate) and answer (:func:`fold_t3_answer`) alike;
    ``auroc_relocate`` is the relocate-vs-rest AUROC in either mode."""
    rows, counts, _ = _split_rows("T3", manifest, answers, split, t3_binary=t3_binary)
    out: dict[str, Any] = dict(task="T3", label_mode="binary" if t3_binary else "three_class", **counts)
    if t3_binary:
        for r in rows:
            r["ans"] = fold_t3_answer(r["ans"])
    present = {r["label"] for r in rows}
    order = T3_BINARY_CLASSES if t3_binary else T3_CLASSES
    classes = [c for c in order if c in present] + sorted(present - set(order))
    out["classes"] = classes
    out["n_by_class"] = {c: sum(1 for r in rows if r["label"] == c) for c in classes}
    pred = [(r["label"], _action(r["ans"]), r) for r in rows if _action(r["ans"])]
    out["n_with_action"] = len(pred)
    out["accuracy"] = _mean([1.0 if y == a else 0.0 for y, a, _ in pred]) if pred else None
    out["accuracy_by_group"] = {str(g): _mean([1.0 if r["label"] == _action(r["ans"]) else 0.0 for r in rs])
                                for g, rs in _by_group([r for _, _, r in pred], group_key).items()}
    out["recall_by_class"] = {c: _mean([1.0 if a == c else 0.0 for y, a, _ in pred if y == c]) for c in classes}
    conf: dict[str, dict[str, int]] = {c: {} for c in classes}
    for y, a, _ in pred:
        conf.setdefault(y, {})[a] = conf.setdefault(y, {}).get(a, 0) + 1
    out["confusion"] = conf
    out["majority_class"] = Counter(r["label"] for r in rows).most_common(1)[0][0] if rows else None
    aucs: dict[str, float | None] = {}
    for c in classes:
        sc = [(int(r["label"] == c), _t3_class_prob(r["ans"], c)) for r in rows]
        sc = [(y, p) for y, p in sc if p is not None]
        aucs[c] = auroc([y for y, _ in sc], [p for _, p in sc]) if sc else None
    out["auroc_one_vs_rest"] = aucs
    out["auroc_macro"] = _mean(list(aucs.values()))
    out["auroc_relocate"] = aucs.get("relocate")
    return out


# ── T4 ──────────────────────────────────────────────────────────────────────

def _gauss_ll(x: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-9)
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


def score_t4(manifest: Any, answers: Any, *, split: str | None = "test", group_key: str = "material",
             bias_abs_tol_v: float = 0.05, bias_rel_tol: float = 0.25, setpoint_log10_tol: float = math.log10(2.0),
             default_bias_sigma_v: float = 0.2, default_setpoint_log10_sigma: float = 0.5) -> dict:
    """Bias is always scored; setpoint and scan speed only where the label carries them
    (the built T4 manifest has ``setpoint_a`` NaN unless raw headers were read, and
    ``speed_nm_s`` from the index). ``hit_rate_both`` = bias ∧ setpoint over samples with both."""
    rows, counts, _ = _split_rows("T4", manifest, answers, split)
    out: dict[str, Any] = dict(task="T4", **counts)
    hb, hs, hsp, hboth, ll, llb, lls = [], [], [], [], [], [], []
    per_group: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        ans = r["ans"]
        pb = _float(ans.get("bias_v"))
        if pb is None:
            continue
        lab = r["label"]
        lb = _float(lab.get("bias_v"))
        if lb is None:
            continue
        tol_b = max(bias_abs_tol_v, bias_rel_tol * abs(lb))
        hit_b = abs(pb - lb) <= tol_b
        hb.append(1.0 if hit_b else 0.0)
        per_group[str(r.get(group_key))].append(1.0 if hit_b else 0.0)
        sb = _float(ans.get("bias_sigma_v")) or default_bias_sigma_v
        b_ll = _gauss_ll(lb, pb, sb)
        llb.append(b_ll)
        total_ll = b_ll
        ls, ps = _float(lab.get("setpoint_a")), _float(ans.get("setpoint_a"))
        if ls and ps:
            hit_s = abs(math.log10(abs(ps)) - math.log10(abs(ls))) <= setpoint_log10_tol
            hs.append(1.0 if hit_s else 0.0)
            hboth.append(1.0 if (hit_b and hit_s) else 0.0)
            ss = _float(ans.get("setpoint_log10_sigma")) or default_setpoint_log10_sigma
            s_ll = _gauss_ll(math.log10(abs(ls)), math.log10(abs(ps)), ss)
            lls.append(s_ll)
            total_ll += s_ll
        ll.append(total_ll)
        lsp, psp = _float(lab.get("speed_nm_s")), _float(ans.get("scan_speed_nm_per_s"))
        if lsp and psp:
            hsp.append(1.0 if abs(math.log10(abs(psp)) - math.log10(abs(lsp))) <= setpoint_log10_tol else 0.0)
    out["n_with_values"] = len(hb)
    out["n_with_setpoint"] = len(hs)
    out["n_with_speed"] = len(hsp)
    out["hit_rate_bias"] = _mean(hb)
    out["hit_rate_setpoint"] = _mean(hs)
    out["hit_rate_speed"] = _mean(hsp)
    out["hit_rate_both"] = _mean(hboth)
    out["mean_log_likelihood"] = _mean(ll)
    out["mean_log_likelihood_bias"] = _mean(llb)
    out["mean_log_likelihood_setpoint"] = _mean(lls)
    out["hit_rate_bias_by_group"] = {g: _mean(v) for g, v in per_group.items()}
    out["tolerances"] = {"bias_abs_tol_v": bias_abs_tol_v, "bias_rel_tol": bias_rel_tol,
                         "setpoint_log10_tol": setpoint_log10_tol, "speed_log10_tol": setpoint_log10_tol}
    return out


# ── dispatcher ──────────────────────────────────────────────────────────────

SCORERS = {"T1": score_t1, "T2": score_t2, "T3": score_t3, "T4": score_t4}


def score(task: str, manifest: Any, answers: Any, **kw) -> dict:
    task = str(task).upper()
    if task not in SCORERS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(SCORERS)}")
    if task != "T3":
        kw.pop("t3_binary", None)          # only T3 has a fold; the CLI passes it uniformly
    return SCORERS[task](manifest, answers, **kw)


def score_by(task: str, manifest: Any, answers: Any, column: str, **kw) -> dict[str, dict]:
    """Stratified report: the task's scorer run once per value of ``column`` (e.g. era, genre)."""
    groups = _by_group(normalize_manifest(manifest, task, t3_binary=bool(kw.get("t3_binary"))), column)
    return {str(k): score(task, rows, answers, **kw) for k, rows in groups.items()}


def abstain_rate(manifest: Any, answers: Any, *, split: str | None = "test", task: str | None = None) -> dict:
    ans_map = answers_by_id(answers)
    n = n_abs = n_missing = 0
    for r in normalize_manifest(manifest, task):
        if split and str(r.get("split", "")) != split:
            continue
        n += 1
        a = ans_map.get(str(r["sample_id"]))
        if a is None:
            n_missing += 1
        elif a.get("abstain"):
            n_abs += 1
    answered = n - n_missing
    return {"n_total": n, "n_answered": answered, "n_abstain": n_abs, "n_missing": n_missing,
            "abstain_rate": (n_abs / answered) if answered else None}
