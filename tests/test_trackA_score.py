"""Track A: scoring on tiny synthetic tables, rendering + drift features on simulator frames."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stmbench.trackA import schema as SCHEMA
from stmbench.trackA import score as S
from stmbench.trackA.baselines import (
    BASELINES, NEGATIVE_CONTROLS, PixelHeadModel, baselines_for, drift_rule_p_stop, flattest_windows,
    paired_final_more_stable, pixel_head_features, resample_grid, run_baseline,
)
from stmbench.trackA.prompts import ABSTAIN, ANSWER_SCHEMAS, TASK_INSTRUCTIONS, build_prompt, context_keys, parse_answer
from stmbench.trackA.render import (
    PNG_SIGNATURE, data_root, flatten_three_step, header_summary, png_bytes_gray, png_shape, render_array, resolve_path,
)

from tests.conftest import requires_mast


# ── AUROC ───────────────────────────────────────────────────────────────────

def test_rank_auroc_known_values_and_ties():
    assert S.rank_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert S.rank_auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert S.rank_auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    # one tie straddling the classes counts half
    assert S.rank_auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)
    assert S.rank_auroc([0, 0, 0], [0.1, 0.2, 0.3]) is None  # single class → undefined
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.integers(0, 5, 200).astype(float)  # heavy ties
    assert S.rank_auroc(y, s) == pytest.approx(sk.roc_auc_score(y, s), abs=1e-12)


# ── T1 ──────────────────────────────────────────────────────────────────────

def _t1_manifest():
    rows = []
    # group A: separable; group B: anti-separable; group C: one class only (no AUROC)
    for i, (g, y, p) in enumerate([("A", 1, 0.9), ("A", 1, 0.8), ("A", 0, 0.2), ("A", 0, 0.1),
                                    ("B", 1, 0.1), ("B", 0, 0.9),
                                    ("C", 1, 0.5), ("C", 1, 0.6)]):
        rows.append({"sample_id": f"s{i}", "task": "T1", "input_paths": [f"x/{i}.sxm"], "label": y,
                     "group": g, "split": "test", "_p": p})
    rows.append({"sample_id": "train0", "task": "T1", "input_paths": ["x/t.sxm"], "label": 1, "group": "A", "split": "train"})
    return rows


def test_score_t1_by_group_abstain_and_missing():
    rows = _t1_manifest()
    answers = [{"sample_id": r["sample_id"], "answer": json.dumps({"stop": r["_p"] >= 0.5, "p_stop": r["_p"]})}
               for r in rows if r["split"] == "test" and r["sample_id"] != "s7"]
    # s6 abstains, s7 is missing entirely
    answers = [a for a in answers if a["sample_id"] != "s6"] + [{"sample_id": "s6", "answer": '{"abstain": true}'}]
    res = S.score_t1(rows, answers, split="test")
    assert res["n_total"] == 8 and res["n_missing"] == 1 and res["n_abstain"] == 1
    assert res["abstain_rate"] == pytest.approx(1 / 7)
    assert res["auroc_by_group"]["A"] == 1.0 and res["auroc_by_group"]["B"] == 0.0
    assert "C" not in res["auroc_by_group"]  # its only scored row is s6... which abstained → no rows
    assert res["auroc_group_mean"] == pytest.approx(0.5) and res["n_groups_scored"] == 2
    assert res["accuracy"] == pytest.approx(4 / 6)
    assert res["auroc_pooled"] == pytest.approx(S.rank_auroc([1, 1, 0, 0, 1, 0], [0.9, 0.8, 0.2, 0.1, 0.1, 0.9]))


def test_score_t1_free_text_answer_with_abstain_word_counts_as_abstain():
    rows = _t1_manifest()[:2]
    answers = [{"sample_id": "s0", "answer": f"我觉得{ABSTAIN}，看不出漂移"}, {"sample_id": "s1", "answer": "garbage ((("}]
    res = S.score_t1(rows, answers, split="test")
    assert res["n_abstain"] == 2 and res["n_scored"] == 0 and res["auroc_pooled"] is None


def test_t1_step_order_and_majority_baselines_shape():
    rows = _t1_manifest()
    for name in ("step_order", "majority"):
        ans = run_baseline(name, rows, task="T1", split="test")
        assert {a["sample_id"] for a in ans} == {r["sample_id"] for r in rows if r["split"] == "test"}
        parsed = [parse_answer(a["answer"]) for a in ans]
        assert all(0 <= p["p_stop"] <= 1 for p in parsed)
    maj = parse_answer(run_baseline("majority", rows, task="T1", split="test")[0]["answer"])
    assert maj["majority_of"] == "train" and maj["p_stop"] == 1.0  # train split has a single positive row
    assert "step_order" in baselines_for("T1") and "drift_rule" in baselines_for("T1")
    assert "pixel_head" in baselines_for("T1")
    assert set(NEGATIVE_CONTROLS) == {"step_order", "pixel_head"} and all(BASELINES[n][0] == "T1" for n in NEGATIVE_CONTROLS)
    assert set(baselines_for("T2")) == {"center", "random", "flattest"}


def test_drift_rule_orders_by_instability_and_abstains_without_steps():
    calm = [{"dz_pm": 20.0, "dx_nm": 0.1, "corr": 0.98}]
    moving = [{"dz_pm": 120.0, "dx_nm": 0.6, "corr": 0.80}]
    p_calm, _ = drift_rule_p_stop(calm)
    p_move, _ = drift_rule_p_stop(moving)
    assert p_calm < 0.5 < p_move
    for v in ("dz", "dx", "corr"):
        assert drift_rule_p_stop(calm, v)[0] < drift_rule_p_stop(moving, v)[0]
    assert drift_rule_p_stop([])[0] is None
    # only the LAST step matters
    assert drift_rule_p_stop(moving + calm)[0] == p_calm


def test_paired_final_more_stable_synthetic():
    steps = []
    for ep in range(10):
        # final step calmer in 8 episodes, wilder in 2
        wild = ep < 2
        steps.append({"ep": ep, "is_final": False, "dz_pm": 100.0, "dx_nm": 0.4, "corr": 0.88})
        steps.append({"ep": ep, "is_final": False, "dz_pm": 80.0, "dx_nm": 0.3, "corr": 0.90})
        steps.append({"ep": ep, "is_final": True, "dz_pm": 200.0 if wild else 20.0, "dx_nm": 0.9 if wild else 0.1,
                      "corr": 0.5 if wild else 0.99})
    res = paired_final_more_stable(steps)
    for k in ("dz_pm", "dx_nm", "corr"):
        assert res[k]["n_episodes"] == 10 and res[k]["frac_final_more_stable"] == pytest.approx(0.8)
    assert res["dz_pm"]["median_diff"] < 0 and res["corr"]["median_diff"] > 0


# ── the single-frame pixel model (negative control, plan §5.1) ─────────────

def _strips(rng: np.random.Generator, n: int, rows: int = 8, cols: int = 64) -> tuple[list[np.ndarray], list[int]]:
    """Class 0: atomic-row-like sinusoid; class 1: a step edge. Both survive the three-step
    rendering (which removes contrast), so a pixel model has something to learn."""
    X, y = [], []
    jj, ii = np.mgrid[0:rows, 0:cols]
    for k in range(n):
        lab = k % 2
        noise = rng.normal(0, 5e-12, (rows, cols))
        if lab == 0:
            a = 2e-11 * np.sin(2 * np.pi * (ii + rng.integers(0, 16)) / 16.0) + noise
        else:
            a = 5e-11 * (ii >= cols // 2 + rng.integers(-4, 5)) + noise
        a = a + 3e-10 * ii + 1e-10 * jj                                   # a tilt the plane fit removes
        X.append(a)
        y.append(lab)
    return X, y


def test_resample_grid_and_pixel_head_features_shape():
    a = np.arange(8 * 64, dtype=float).reshape(8, 64)
    g = resample_grid(a, (16, 16))
    assert g.shape == (16, 16)
    assert g[0, 0] == pytest.approx(a[0, 0:4].mean()) and g[15, 15] == pytest.approx(a[7, 60:64].mean())
    assert (np.diff(g, axis=1) > 0).all()                                   # monotone along x is preserved
    assert resample_grid(np.full((8, 64), np.nan)).sum() == 0.0            # NaN counts as 0
    assert pixel_head_features(a).shape == (256,)
    with pytest.raises(ValueError):
        resample_grid(np.zeros((0, 4)))


@pytest.mark.parametrize("kind", ["auto", "centroid"])
def test_pixel_head_model_learns_from_head_pixels_alone(kind):
    rng = np.random.default_rng(3)
    strips, labels = _strips(rng, 40)
    feats = np.vstack([pixel_head_features(s) for s in strips])
    m = PixelHeadModel(kind).fit(feats[:28], labels[:28])
    assert m.kind == ("centroid" if kind == "centroid" else m.kind) and m.n_train == 28
    if kind == "auto":
        try:
            import sklearn  # noqa: F401
            assert m.kind == "logreg"
        except ImportError:
            assert m.kind == "centroid"
    p = m.predict_proba(feats[28:])
    assert p.shape == (12,) and ((p >= 0) & (p <= 1)).all()
    assert S.rank_auroc(labels[28:], p) == 1.0
    with pytest.raises(ValueError):
        PixelHeadModel(kind).fit(feats[:4], [1, 1, 1, 1])                # a single class cannot be fitted
    with pytest.raises(ValueError):
        PixelHeadModel("bogus").fit(feats[:4], [0, 1, 0, 1])


def test_pixel_head_baseline_abstains_when_untrainable():
    rows = [{"sample_id": "a", "task": "T1", "input_paths": ["nope/1.sxm"], "label": 1, "group": "g", "split": "test"},
            {"sample_id": "t", "task": "T1", "input_paths": ["nope/2.sxm"], "label": 1, "group": "g", "split": "train"}]
    ans = {a["sample_id"]: parse_answer(a["answer"]) for a in run_baseline("pixel_head", rows, split="test")}
    assert ans["a"]["abstain"] is True and "untrainable" in ans["a"]["reason"] and ans["a"]["negative_control"] is True


# ── T2 ──────────────────────────────────────────────────────────────────────

def test_iou_boxes():
    assert S.iou_boxes((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert S.iou_boxes((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert S.iou_boxes((0, 0, 10, 10), (20, 0, 10, 10)) == 0.0


def test_score_t2_iou_center_distance_topk():
    rows = [
        {"sample_id": "a", "task": "T2", "input_paths": ["x.sxm"], "label": [10.0, -5.0], "group": "g", "split": "test",
         "range_nm": 100.0, "box_nm": 20.0},
        {"sample_id": "b", "task": "T2", "input_paths": ["y.sxm"], "label": "[0, 0]", "group": "g", "split": "test",
         "range_nm": 50.0, "box_nm": 10.0},
    ]
    answers = [
        {"sample_id": "a", "answer": json.dumps({"dx_nm": 10.0, "dy_nm": -5.0, "size_nm": 20.0})},               # exact
        {"sample_id": "b", "answer": json.dumps({"dx_nm": 20.0, "dy_nm": 0.0, "size_nm": 10.0,
                                                 "candidates": [[20.0, 0.0], [-20.0, 0.0], [1.0, 1.0]]})},  # 3rd hits
    ]
    res = S.score_t2(rows, answers, split="test")
    assert res["n_with_geometry"] == 2
    assert res["mean_iou"] == pytest.approx(0.5)
    assert res["mean_center_dist_norm"] == pytest.approx((0.0 + 20 / 50) / 2)
    assert res["hit@1"] == 0.5 and res["hit@3"] == 1.0 and res["hit@5"] == 1.0
    assert res["iou_by_group"]["g"] == pytest.approx(0.5)


def test_score_t2_non_square_frame_scales_dy_and_box_height():
    # 200 × 100 nm frame (nx=256, ny=128): next_v = −0.1 of the HEIGHT is −10 nm, not −20
    row = {"sample_id": "n", "group_key": "g", "split": "test", "rel_path": "d/n.sxm", "path": "E:/raw/d/n.sxm",
           "size_nm": 200.0, "height_nm": 100.0, "nx": 256, "ny": 128,
           "next_u": 0.25, "next_v": -0.1, "next_w_rel": 0.2, "next_h_rel": 0.1}
    n = S.normalize_manifest([row])[0]
    assert n["label"] == [50.0, -10.0] and n["range_nm"] == 200.0 and n["height_nm"] == 100.0
    assert n["box_nm"] == pytest.approx(40.0) and n["box_h_nm"] == pytest.approx(10.0)
    assert n["input_paths"] == ["d/n.sxm"]                                 # rel_path, not the absolute meta path
    # a square 40×40 answer on the 40×10 target: inter 400 / union 1600
    ans = [{"sample_id": "n", "answer": json.dumps({"dx_nm": 50.0, "dy_nm": -10.0, "size_nm": 40.0})}]
    res = S.score_t2([row], ans, split="test")
    assert res["mean_iou"] == pytest.approx(0.25) and res["hit@1"] == 1.0
    # an answer at dy = −22 misses the 40 × 10 box centred at −10 (rows −15…−5) …
    ans2 = [{"sample_id": "n", "answer": json.dumps({"dx_nm": 50.0, "dy_nm": -22.0, "size_nm": 40.0})}]
    assert S.score_t2([row], ans2, split="test")["hit@1"] == 0.0
    # … but on a square 200 × 200 frame the same next_v/next_h_rel is a 40 × 20 box centred at −20 (−30…−10): a hit
    sq = dict(row, height_nm=200.0, ny=256)
    assert S.normalize_manifest([sq])[0]["label"] == [50.0, -20.0]
    assert S.score_t2([sq], ans2, split="test")["hit@1"] == 1.0


def test_t2_center_and_random_baselines_and_flattest_windows():
    rows = [{"sample_id": "a", "task": "T2", "input_paths": ["nope.sxm"], "label": [0, 0], "group": "g", "split": "test",
             "range_nm": 100.0, "box_nm": 25.0}]
    c = parse_answer(run_baseline("center", rows, split="test")[0]["answer"])
    assert c["dx_nm"] == 0 and c["dy_nm"] == 0 and c["size_nm"] == 25.0
    r1 = parse_answer(run_baseline("random", rows, split="test", seed=3)[0]["answer"])
    r2 = parse_answer(run_baseline("random", rows, split="test", seed=3)[0]["answer"])
    assert r1 == r2 and len(r1["candidates"]) == 5 and all(abs(v) <= 37.5 for xy in r1["candidates"] for v in xy)
    # a 100 × 50 frame: the random centres stay inside the shorter axis too
    tall = [dict(rows[0], height_nm=50.0)]
    r3 = parse_answer(run_baseline("random", tall, split="test", seed=3)[0]["answer"])
    assert all(abs(xy[1]) <= 12.5 for xy in r3["candidates"]) and all(abs(xy[0]) <= 37.5 for xy in r3["candidates"])
    # flattest: a plane with one rough patch → the roughest window is never chosen
    a = np.zeros((64, 64)) + np.arange(64)[None, :] * 1e-12
    rng = np.random.default_rng(0)
    a[8:24, 8:24] += rng.normal(0, 1e-10, (16, 16))
    a[-8:] = np.nan                                        # unacquired tail rows
    wins = flattest_windows(a, box_px=16, n_best=3)
    assert wins and all(not (8 <= ci <= 24 and 8 <= cj <= 24) for ci, cj, _ in wins)
    # a frame unreadable on disk → abstain, not (0, 0)
    f = parse_answer(run_baseline("flattest", rows, split="test")[0]["answer"])
    assert f["abstain"] is True


# ── T3 ──────────────────────────────────────────────────────────────────────

def test_score_t3_accuracy_auroc_and_majority():
    rows = [{"sample_id": f"s{i}", "task": "T3", "input_paths": ["x.sxm"], "label": lab, "group": "g", "split": "test"}
            for i, lab in enumerate(["stay", "stay", "relocate", "relocate"])]
    rows.append({"sample_id": "t", "task": "T3", "input_paths": ["x.sxm"], "label": "Relocate", "group": "g", "split": "train"})
    answers = [{"sample_id": "s0", "answer": json.dumps({"action": "stay", "p_relocate": 0.1})},
               {"sample_id": "s1", "answer": json.dumps({"action": "relocate", "p_relocate": 0.8})},
               {"sample_id": "s2", "answer": json.dumps({"action": "relocate", "p_relocate": 0.9})},
               {"sample_id": "s3", "answer": json.dumps({"action": "relocate", "p_relocate": 0.7})}]
    res = S.score_t3(rows, answers, split="test")
    assert res["label_mode"] == "three_class" and res["classes"] == ["stay", "relocate"]   # schema order, present classes
    assert res["accuracy"] == 0.75 and res["recall_by_class"] == {"stay": 0.5, "relocate": 1.0}
    assert res["auroc_relocate"] == pytest.approx(0.75) == res["auroc_one_vs_rest"]["relocate"]
    assert res["auroc_one_vs_rest"]["stay"] == pytest.approx(0.75)         # hard actions: one stay wrong
    assert res["auroc_macro"] == pytest.approx(0.75)
    assert res["confusion"] == {"stay": {"stay": 1, "relocate": 1}, "relocate": {"relocate": 2}}
    maj = parse_answer(run_baseline("majority", rows, task="T3", split="test")[0]["answer"])
    assert maj["action"] == "relocate" and maj["p_relocate"] == 1.0


def _t3_three_class_rows():
    labs = ["stay", "stay", "relocate", "relocate", "long_stop", "long_stop"]
    rows = [{"sample_id": f"s{i}", "group_key": "g", "split": "test", "rel_path": f"d/{i}.sxm", "path": f"E:/raw/d/{i}.sxm",
             "size_nm": 20.0, "height_nm": 20.0, "next_action": lab, "relocate": lab != "stay",
             "next_action_fine": "restart_same" if lab == "stay" else lab}
            for i, lab in enumerate(labs)]
    rows += [dict(rows[0], sample_id=f"t{i}", split="train", next_action=lab, relocate=lab != "stay",
                  next_action_fine="restart_same" if lab == "stay" else lab)
             for i, lab in enumerate(["stay", "stay", "stay", "relocate", "long_stop"])]
    return rows


def test_score_t3_three_class_default_and_binary_fold():
    rows = _t3_three_class_rows()
    P = lambda s, r, l: {"stay": s, "relocate": r, "long_stop": l}          # noqa: E731
    answers = [
        {"sample_id": "s0", "answer": json.dumps({"action": "stay", "probs": P(0.7, 0.2, 0.1)})},
        {"sample_id": "s1", "answer": json.dumps({"action": "relocate", "probs": P(0.3, 0.5, 0.2)})},   # wrong
        {"sample_id": "s2", "answer": json.dumps({"action": "relocate", "probs": P(0.1, 0.8, 0.1)})},
        {"sample_id": "s3", "answer": json.dumps({"action": "long_stop", "probs": P(0.1, 0.4, 0.5)})},  # wrong in 3-class, right folded
        {"sample_id": "s4", "answer": json.dumps({"action": "long_stop", "probs": P(0.1, 0.1, 0.8)})},
        {"sample_id": "s5", "answer": json.dumps({"action": "stay", "probs": P(0.5, 0.1, 0.4)})},       # wrong either way
    ]
    three = S.score("T3", rows, answers, split="test")
    assert three["label_mode"] == "three_class" and three["classes"] == ["stay", "relocate", "long_stop"]
    assert three["n_by_class"] == {"stay": 2, "relocate": 2, "long_stop": 2}
    assert three["accuracy"] == pytest.approx(3 / 6)
    assert three["recall_by_class"] == {"stay": 0.5, "relocate": 0.5, "long_stop": 0.5}
    ovr = three["auroc_one_vs_rest"]
    assert set(ovr) == {"stay", "relocate", "long_stop"}
    assert ovr["stay"] == pytest.approx(S.rank_auroc([1, 1, 0, 0, 0, 0], [0.7, 0.3, 0.1, 0.1, 0.1, 0.5]))
    assert ovr["long_stop"] == pytest.approx(S.rank_auroc([0, 0, 0, 0, 1, 1], [0.1, 0.2, 0.1, 0.5, 0.8, 0.4]))
    assert three["auroc_macro"] == pytest.approx(np.mean([ovr["stay"], ovr["relocate"], ovr["long_stop"]]))
    # the fold: relocate + long_stop → relocate on the label AND on the answer
    two = S.score("T3", rows, answers, split="test", t3_binary=True)
    assert two["label_mode"] == "binary" and two["classes"] == ["stay", "relocate"]
    assert two["n_by_class"] == {"stay": 2, "relocate": 4}
    assert two["accuracy"] == pytest.approx(4 / 6)                         # s3 now right; s1, s5 still wrong
    assert two["auroc_relocate"] == pytest.approx(S.rank_auroc([0, 0, 1, 1, 1, 1], [0.3, 0.7, 0.9, 0.9, 0.9, 0.5]))
    assert two["auroc_macro"] == pytest.approx(np.mean([two["auroc_one_vs_rest"]["stay"], two["auroc_relocate"]]))
    # normalize_manifest: three-class label by default, fold on request; contract rows fold too
    assert [r["label"] for r in S.normalize_manifest(rows)[:6]] == ["stay", "stay", "relocate", "relocate", "long_stop", "long_stop"]
    assert [r["label"] for r in S.normalize_manifest(rows, t3_binary=True)[:6]] == ["stay", "stay"] + ["relocate"] * 4
    contract = S.normalize_manifest(rows)
    assert S.normalize_manifest(contract, "T3", t3_binary=True)[4]["label"] == "relocate"
    # the folded answer keeps the mass that left "stay"
    assert S.fold_t3_answer({"action": "long_stop", "probs": P(0.1, 0.4, 0.5)}) == {
        "action": "relocate", "probs": {"stay": pytest.approx(0.1), "relocate": pytest.approx(0.9)}, "p_relocate": pytest.approx(0.9)}
    assert S.fold_t3_answer({"action": "stay", "p_relocate": 0.2}) == {"action": "stay", "p_relocate": 0.2}
    # the majority baseline is fitted on the folded train labels when the fold is on
    maj3 = parse_answer(run_baseline("majority", rows, task="T3", split="test")[0]["answer"])
    maj2 = parse_answer(run_baseline("majority", rows, task="T3", split="test", t3_binary=True)[0]["answer"])
    assert set(maj3["probs"]) == {"stay", "relocate", "long_stop"} and maj3["probs"]["stay"] == pytest.approx(0.6)
    assert set(maj2["probs"]) == {"stay", "relocate"} and maj2["p_relocate"] == pytest.approx(0.4)
    # a non-T3 scorer ignores the fold flag instead of choking on it
    assert S.score("T1", _t1_manifest(), [], split="test", t3_binary=True)["n_total"] == 8


def test_cli_score_t3_three_class_default_and_binary_flag(tmp_path, capsys):
    import pandas as pd
    from stmbench.trackA.cli import main
    mpath = tmp_path / "T3.parquet"
    pd.DataFrame(_t3_three_class_rows()).to_parquet(mpath, index=False)
    out3, out2 = tmp_path / "three.json", tmp_path / "two.json"
    assert main(["score", "--task", "T3", "--manifest", str(mpath), "--baseline", "majority", "--out", str(out3)]) == 0
    assert main(["score", "--task", "T3", "--manifest", str(mpath), "--baseline", "majority", "--t3-binary",
                 "--out", str(out2), "--by", "group_key"]) == 0
    capsys.readouterr()
    three = json.loads(out3.read_text(encoding="utf-8"))
    two = json.loads(out2.read_text(encoding="utf-8"))
    assert three["label_mode"] == "three_class" and three["overall"]["classes"] == ["stay", "relocate", "long_stop"]
    assert three["overall"]["accuracy"] == pytest.approx(2 / 6) and "auroc_macro" in three["overall"]
    assert two["label_mode"] == "binary" and two["overall"]["classes"] == ["stay", "relocate"]
    assert two["by"]["group_key"]["g"]["classes"] == ["stay", "relocate"]


# ── T4 ──────────────────────────────────────────────────────────────────────

def test_score_t4_tolerance_and_log_likelihood():
    rows = [{"sample_id": "a", "task": "T4", "input_paths": ["x.sxm"], "label": {"bias_v": 0.1, "setpoint_a": 1e-10},
             "material": "Au", "group": "g", "split": "test"},
            {"sample_id": "b", "task": "T4", "input_paths": ["x.sxm"], "label": '{"bias_v": 1.0, "setpoint_a": 5e-11}',
             "material": "Cu", "group": "g", "split": "test"}]
    answers = [{"sample_id": "a", "answer": json.dumps({"bias_v": 0.12, "setpoint_a": 1.5e-10})},   # both hit
               {"sample_id": "b", "answer": json.dumps({"bias_v": 0.5, "setpoint_a": 5e-11,
                                                        "bias_sigma_v": 0.5, "setpoint_log10_sigma": 0.2})}]  # bias miss
    res = S.score_t4(rows, answers, split="test")
    assert res["hit_rate_bias"] == 0.5 and res["hit_rate_setpoint"] == 1.0 and res["hit_rate_both"] == 0.5
    assert res["hit_rate_bias_by_group"] == {"Au": 1.0, "Cu": 0.0} and res["n_with_setpoint"] == 2
    ll_b = -0.5 * ((1.0 - 0.5) / 0.5) ** 2 - math.log(0.5 * math.sqrt(2 * math.pi))
    ll_s = -math.log(0.2 * math.sqrt(2 * math.pi))
    ll_a = (-0.5 * (0.02 / 0.2) ** 2 - math.log(0.2 * math.sqrt(2 * math.pi))
            - 0.5 * (math.log10(1.5) / 0.5) ** 2 - math.log(0.5 * math.sqrt(2 * math.pi)))
    assert res["mean_log_likelihood"] == pytest.approx((ll_a + ll_b + ll_s) / 2)


def test_t4_material_median_baseline_uses_train_and_falls_back():
    train = [{"sample_id": f"t{i}", "task": "T4", "input_paths": [], "label": {"bias_v": b, "setpoint_a": s},
              "material": m, "split": "train"}
             for i, (m, b, s) in enumerate([("Au", 0.1, 1e-10), ("Au", 0.2, 2e-10), ("Au", 0.1, 1e-10),
                                            ("Cu", 1.0, 5e-11)])]
    test = [{"sample_id": "x", "task": "T4", "input_paths": [], "label": {"bias_v": 0.1, "setpoint_a": 1e-10}, "material": "Au", "split": "test"},
            {"sample_id": "y", "task": "T4", "input_paths": [], "label": {"bias_v": 0.5, "setpoint_a": 1e-10}, "material": "HOPG", "split": "test"}]
    ans = {a["sample_id"]: parse_answer(a["answer"]) for a in run_baseline("material_median", train + test, split="test")}
    assert ans["x"]["bias_v"] == pytest.approx(0.1) and ans["x"]["setpoint_a"] == pytest.approx(1e-10) and not ans["x"]["fallback"]
    assert ans["y"]["fallback"] and ans["y"]["n_train"] == 4
    res = S.score_t4(train + test, [{"sample_id": k, "answer": json.dumps(v)} for k, v in ans.items()], split="test")
    assert res["hit_rate_both"] == 0.5


# ── prompts / parsing ───────────────────────────────────────────────────────

def test_parse_answer_variants():
    assert parse_answer({"stop": True, "p_stop": 0.9})["abstain"] is False
    assert parse_answer('{"abstain": true, "reason": "x"}')["abstain"] is True
    assert parse_answer('{"answer": "判不了"}')["abstain"] is True
    assert parse_answer('前面有话 {"stop": false, "p_stop": 0.2} 后面有话')["p_stop"] == 0.2
    assert parse_answer(None)["parse_error"] is True
    assert parse_answer("nonsense")["abstain"] is True


def test_build_prompt_hides_working_point_for_t4_and_lists_frames():
    class R:
        def __init__(self, s):
            self.png = b"\x89PNG"
            self.summary = s

    s = {"size_nm": [100, 100], "pixels": [256, 256], "bias_v": 0.123, "setpoint_a": 9.9e-11, "comment": "SECRET", "acq_frac": 0.5}
    p4 = build_prompt("T4", {"sample_id": "k", "material": "Au(111)", "label": {"bias_v": 0.123, "setpoint_a": 9.9e-11}}, [R(s)])
    assert "0.123" not in p4["user"] and "SECRET" not in p4["user"] and "9.9e-11" not in p4["user"]
    assert "Au(111)" in p4["user"] and p4["images"] == [b"\x89PNG"] and p4["schema"] is ANSWER_SCHEMAS["T4"]
    p1 = build_prompt("T1", {"sample_id": "k", "group": "Rig_B-2025-01-21"}, [R(s), R(s), R(s)])
    assert "第 3/3 帧" in p1["user"] and ABSTAIN in p1["system"] and "0.123" in p1["user"]
    with pytest.raises(ValueError):
        build_prompt("T9", {}, [])


def test_t1_prompt_judges_the_head_only_frame_and_t3_offers_three_classes():
    t1 = TASK_INSTRUCTIONS["T1"]
    assert f"前 {SCHEMA.T1_HEAD_ROWS} 行" in t1 and "扫完之前" in t1 and "下一次起扫" not in t1
    t3 = TASK_INSTRUCTIONS["T3"]
    assert all(c in t3 for c in SCHEMA.NEXT_ACTIONS) and "long_stop" in ANSWER_SCHEMAS["T3"]["oneOf"][0]["properties"]["action"]["enum"]
    assert f"{SCHEMA.LONG_STOP_S / 60:g} 分钟" in t3
    # the judged frame (head-only render) is tagged as such; an earlier frame shows its acquired fraction
    head = {"size_nm": [20, 20], "head_rows_shown": 8, "rows_total": 256}
    full = {"size_nm": [20, 20], "acq_frac": 0.03}
    p = build_prompt("T1", {"sample_id": "k"}, [{"png": b"\x89PNG", "summary": full}, {"png": b"\x89PNG", "summary": head}])
    assert "第 1/2 帧（已扫 3%）" in p["user"] and "第 2/2 帧（只显示刚扫出的前 8 行" in p["user"]


def test_prompt_context_keys_derive_from_the_schema_and_never_carry_a_label():
    for task, spec in SCHEMA.SPECS.items():
        keys = set(context_keys(task))
        assert not (keys & set(spec.label_columns)) and not (keys & set(spec.by_role(SCHEMA.ROLE_META)))
        assert not (keys & spec.leak_forbidden)
        assert "path" not in keys and "rel_path" not in keys and "context_paths_json" not in keys
        assert set(spec.by_role(SCHEMA.ROLE_STRAT)) <= keys
    assert {"step_idx", "head_rows", "autosave_regime", "context_dt_s_json"} <= set(context_keys("T1"))
    assert {"material", "adsorbate", "z_feedback", "controller"} <= set(context_keys("T4")) and "bias_v" not in context_keys("T4")
    # a built-manifest T1 row: strat/input columns reach the prompt, label and meta never do
    row = {"sample_id": "k", "instrument": "Rig_B", "year": 2025, "autosave_regime": "on", "genre": "clean_metal",
           "material": "Au(111)", "step_idx": 2, "head_rows": 8, "stopped": True, "acq_frac": 0.03, "run_len": 5,
           "abort_kind": "glance", "next_path": "E:/raw/next.sxm", "rel_path": "d/x.sxm", "path": "E:/raw/d/x.sxm"}
    user = build_prompt("T1", row, [])["user"]
    for shown in ("autosave_regime", "clean_metal", "Au(111)", "step_idx"):
        assert shown in user
    for hidden in ("stopped", "acq_frac", "run_len", "abort_kind", "glance", "next.sxm", "x.sxm"):
        assert hidden not in user


# ── rendering (pure numpy) ──────────────────────────────────────────────────

def test_three_step_render_and_png_roundtrip():
    rng = np.random.default_rng(1)
    jj, ii = np.mgrid[0:40, 0:50]
    a = 3e-10 * ii + 1e-10 * jj + rng.normal(0, 5e-12, (40, 50))     # tilted plane + noise
    a += (np.arange(40) % 3 == 0)[:, None] * 2e-10                     # row DC jumps
    a[30:] = np.nan                                                    # aborted at row 30
    img, stats = flatten_three_step(a)
    fin = np.isfinite(img)
    assert fin[:30].all() and not fin[30:].any()
    assert 0.4 < np.nanmean(img) < 0.6 and stats["mad"] > 0
    png = png_bytes_gray(img)
    assert png[:8] == PNG_SIGNATURE and png_shape(png) == (40, 50)
    Image = pytest.importorskip("PIL.Image")
    import io
    im = np.asarray(Image.open(io.BytesIO(png)))
    assert im.shape == (40, 50) and (im[30:] == 0).all() and im[:30].std() > 10
    _, _, acq, n = render_array(a)
    assert (acq, n) == (30, 40)
    png2, _, _, _ = render_array(a, crop_to_acquired=True)
    assert png_shape(png2) == (30, 50)
    # flat frame must not divide by zero
    png3 = png_bytes_gray(flatten_three_step(np.zeros((4, 4)))[0])
    assert png_shape(png3) == (4, 4)


def test_header_summary_and_data_root(monkeypatch):
    h = {"scan_range": "1.000000E-07  5.000000E-08", "scan_pixels": [256, 128], "bias": "1.000000E-01",
         "z-controller>setpoint": "5.000000E-11 A", "scan_time": "5.860000E-01 5.860000E-01", "scan_angle": "0.0",
         "scan_dir": "up", "rec_date": "21.01.2025", "rec_time": "19:17:00", "comment": "Cu(111)",
         "z-controller>controller_status": "ON"}
    s = header_summary(h)
    assert s["size_nm"] == [100.0, 50.0] and s["pixels"] == [256, 128] and s["bias_v"] == 0.1
    assert s["setpoint_a"] == 5e-11 and s["scan_speed_nm_per_s"] == pytest.approx(100 / 0.586)
    assert s["rec_time"] == "2025-01-21 19:17:00" and s["comment"] == "Cu(111)" and s["z_controller_on"] is True
    s4 = header_summary(h, include_comment=False, include_time=False)
    assert "comment" not in s4 and "rec_time" not in s4
    assert header_summary({"rec_date": "01.01.1904", "rec_time": "08:00:00"})["rec_time"] is None
    monkeypatch.setenv("STM_BENCH_DATA", "Z:/somewhere")
    assert data_root() == Path("Z:/somewhere")


def test_relative_frame_references_resolve_under_the_raw_mirror_root(monkeypatch):
    monkeypatch.setenv("STM_BENCH_RAW", "Q:/mirror root")
    monkeypatch.setenv("STM_BENCH_DATA", "Z:/somewhere")
    assert resolve_path("corpus/Rig_B SPM data/2025/01/21/Ag0001.sxm") == Path("Q:/mirror root/corpus/Rig_B SPM data/2025/01/21/Ag0001.sxm")
    assert resolve_path("E:/abs/x.sxm") == Path("E:/abs/x.sxm")             # absolute passes through (old manifests, sim frames)
    row = {"sample_id": "k", "group_key": "g", "split": "test", "rel_path": "d/3.sxm", "path": "E:/raw/d/3.sxm",
           "context_paths_json": '["d/1.sxm", "d/2.sxm"]', "head_rows": 8, "step_idx": 2, "stopped": False}
    n = S.normalize_manifest([row])[0]
    assert n["input_paths"] == ["d/1.sxm", "d/2.sxm", "d/3.sxm"]
    assert [str(resolve_path(p)) for p in n["input_paths"]] == [str(Path("Q:/mirror root") / f"d/{i}.sxm") for i in (1, 2, 3)]


# ── simulator frames (needs MAST's reader) ──────────────────────────────────

def _world(tmp_path: Path, seed: int = 1):
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "session", time_scale=1.0)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.scan.nx = w.scan.ny = 48
    w.scan.w = w.scan.h = 20e-9
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.05
    w.scan.comment = "Au(111) 4K test"
    return w


def _frame(w, *, rows: int | None = None, cx: float = 0.0, gap_s: float = 15.0) -> str:
    w.move_xy(cx, 0.0)
    w.scan.cx = cx
    w.scan_start()
    if rows is None:
        w.clock.advance_sim(w.scan.frame_time_s + 0.5)
    else:
        w.clock.advance_sim(rows * (w.scan.line_time_fwd_s + w.scan.line_time_bwd_s) + 0.01)
        w.scan_stop()
    path = w.save_frame()
    w.clock.advance_sim(gap_s)
    return path


@requires_mast
def test_render_simulated_sxm_frame(tmp_path):
    from stmbench.trackA.render import render_frame
    w = _world(tmp_path)
    full = _frame(w)
    part = _frame(w, rows=12)
    r = render_frame(full)
    assert r.png[:8] == PNG_SIGNATURE and png_shape(r.png) == (48, 48) and r.acq_rows == 48 and r.acq_frac == 1.0
    assert r.summary["pixels"] == [48, 48] and r.summary["size_nm"] == [20.0, 20.0]
    assert r.summary["bias_v"] == pytest.approx(w.bias_v) and r.summary["setpoint_a"] == pytest.approx(w.zctrl.setpoint_a)
    assert r.summary["comment"] == "Au(111) 4K test" and r.summary["rec_time"].startswith("2026-09-01")
    assert r.summary["scan_speed_nm_per_s"] == pytest.approx(20 / 0.05)
    r4 = render_frame(full, include_comment=False, include_time=False)
    assert "comment" not in r4.summary and "rec_time" not in r4.summary
    rp = render_frame(part)
    assert 8 <= rp.acq_rows < 48 and rp.summary["acq_frac"] == pytest.approx(rp.acq_rows / 48)
    assert png_shape(render_frame(part, crop_to_acquired=True).png) == (rp.acq_rows, 48)


@requires_mast
def test_drift_features_and_drift_rule_on_simulated_glances(tmp_path):
    from stmbench.trackA.baselines import glance_step_features
    w = _world(tmp_path)
    nm_px = 20.0 / 48
    g1 = _frame(w, rows=12, cx=0.0)
    g2 = _frame(w, rows=12, cx=4 * nm_px * 1e-9)     # the image moved by 4 px between glances
    g3 = _frame(w, rows=12, cx=4 * nm_px * 1e-9)     # ... and then stood still
    full = _frame(w, cx=4 * nm_px * 1e-9)
    feats = glance_step_features([g1, g2, g3, full])
    assert [f["step"] for f in feats] == [1, 2, 3] and all(f["side"] == "top" for f in feats)
    # scanner creep after a move can shave a pixel off the nominal 4 px; still ≥ 3 and ≫ the settled steps
    assert 3 <= feats[0]["dx_px"] <= 5 and feats[0]["dx_nm"] == pytest.approx(feats[0]["dx_px"] * nm_px, rel=1e-3)
    assert feats[1]["dx_px"] <= 1 and feats[2]["dx_px"] <= 1
    assert all(f["dt_s"] == pytest.approx(15.0 + 12 * 0.1 + 0.01, abs=1.5) for f in feats[:2])
    assert feats[1]["corr"] > 0.95 and feats[0]["corr"] > 0.8
    assert all(np.isfinite(f["dz_pm"]) for f in feats)
    # the moving step is scored as "will stop again" more strongly than the settled step
    p_move = drift_rule_p_stop(feats[:1])[0]
    p_still = drift_rule_p_stop(feats[:2])[0]
    assert p_move > p_still
    # through the baseline runner: one sample per episode step
    rows = [{"sample_id": "e0_1", "task": "T1", "input_paths": [g1, g2], "label": 1, "group": "sim", "split": "test"},
            {"sample_id": "e0_2", "task": "T1", "input_paths": [g1, g2, g3], "label": 0, "group": "sim", "split": "test"},
            {"sample_id": "e0_0", "task": "T1", "input_paths": [g1], "label": 1, "group": "sim", "split": "test"}]
    ans = run_baseline("drift_rule", rows, split="test")
    parsed = {a["sample_id"]: parse_answer(a["answer"]) for a in ans}
    assert parsed["e0_0"]["abstain"] is True                      # a single glance has no step yet
    assert parsed["e0_1"]["p_stop"] > parsed["e0_2"]["p_stop"]
    res = S.score_t1(rows, ans, split="test")
    assert res["n_abstain"] == 1 and res["auroc_pooled"] == 1.0
    assert BASELINES["step_order"][0] == "T1"
    # the judged frame contributes its head band only: a complete judged frame and a 12-row glance at
    # the same spot give the same last step (the tail of the complete frame is never read)
    same = glance_step_features([g3, full])[0]
    assert same["dx_px"] <= 1 and same["side"] == "top"


@requires_mast
def test_pixel_head_negative_control_on_simulated_frames(tmp_path):
    w = _world(tmp_path)
    parts = [_frame(w, rows=12) for _ in range(3)]
    fulls = [_frame(w) for _ in range(3)]
    rows = []
    for i, (p, f) in enumerate(zip(parts, fulls)):
        split = "train" if i < 2 else "test"
        rows.append({"sample_id": f"p{i}", "task": "T1", "input_paths": [p], "label": 1, "group": "sim", "split": split, "head_rows": 8})
        rows.append({"sample_id": f"f{i}", "task": "T1", "input_paths": [f], "label": 0, "group": "sim", "split": split, "head_rows": 8})
    ans = {a["sample_id"]: parse_answer(a["answer"]) for a in run_baseline("pixel_head", rows, split="test")}
    assert set(ans) == {"p2", "f2"}
    for a in ans.values():
        assert a["abstain"] is False and 0.0 <= a["p_stop"] <= 1.0 and a["negative_control"] is True
        assert a["n_train"] == 4 and a["grid"] == [16, 16] and a["model"] in ("logreg", "centroid")
    res = S.score_t1(rows, [{"sample_id": k, "answer": json.dumps(v)} for k, v in ans.items()], split="test")
    assert res["n_scored"] == 2 and res["auroc_pooled"] in (0.0, 0.5, 1.0)


@requires_mast
def test_cli_score_and_render_roundtrip(tmp_path, monkeypatch, capsys):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from stmbench.trackA.cli import main
    w = _world(tmp_path)
    g1 = _frame(w, rows=12)
    full = _frame(w)
    man = pd.DataFrame([
        {"sample_id": "a", "task": "T1", "input_paths": [g1, full], "label": 0, "group": "sim", "split": "test", "era": "2025"},
        {"sample_id": "b", "task": "T1", "input_paths": [g1], "label": 1, "group": "sim", "split": "test", "era": "2025"},
    ])
    mpath = tmp_path / "T1.parquet"
    man.to_parquet(mpath, index=False)
    out = tmp_path / "report.json"
    assert main(["score", "--task", "T1", "--manifest", str(mpath), "--baseline", "step_order", "--by", "era",
                 "--out", str(out), "--save-answers", str(tmp_path / "ans.parquet")]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["source"] == {"baseline": "step_order", "negative_control": True}
    assert rep["overall"]["auroc_pooled"] == 1.0 and rep["by"]["era"]["2025"]["n_scored"] == 2
    assert main(["score", "--task", "T1", "--manifest", str(mpath), "--answers", str(tmp_path / "ans.parquet")]) == 0
    assert '"auroc_pooled": 1.0' in capsys.readouterr().out
    assert main(["score", "--task", "T1", "--manifest", str(mpath), "--baseline", "pixel_head"]) == 0
    assert '"negative_control": true' in capsys.readouterr().out
    rdir = tmp_path / "render"
    assert main(["render", "--task", "T1", "--manifest", str(mpath), "--out", str(rdir)]) == 0
    recs = {r["sample_id"]: r for r in map(json.loads, (rdir / "prompts.jsonl").read_text(encoding="utf-8").splitlines())}
    assert set(recs) == {"a", "b"} and len(recs["a"]["images"]) + len(recs["b"]["images"]) == 3
    assert all((rdir / f).exists() for r in recs.values() for f in r["images"])
    # T1: earlier frames in full (acq_frac known), the judged frame head-only (its outcome withheld)
    a_frames = recs["a"]["frames"]
    assert a_frames[0]["summary"]["acq_frac"] == pytest.approx(12 / 48, abs=0.05) and "head_rows_shown" not in a_frames[0]["summary"]
    assert a_frames[1]["summary"]["head_rows_shown"] == SCHEMA.T1_HEAD_ROWS and "acq_frac" not in a_frames[1]["summary"]
    assert a_frames[1]["shape"] == [SCHEMA.T1_HEAD_ROWS, 48] and recs["a"]["head_rows"] == SCHEMA.T1_HEAD_ROWS
    assert "第 2/2 帧（只显示刚扫出的前 8 行" in recs["a"]["user"]
    monkeypatch.setenv("STM_BENCH_DATA", str(tmp_path / "nowhere"))
    with pytest.raises(SystemExit):
        main(["score", "--task", "T2", "--baseline", "center"])   # default manifest absent → clean exit, no traceback


# ── built-manifest adapter (the stmbench.trackA.schema column shape) ────────

def test_normalize_manifest_maps_schema_columns_to_contract():
    t1 = [{"sample_id": "T1:a", "instrument": "Rig_B", "group_key": "Rig_B:2025-05-12", "split": "test", "year": 2025,
           "path": "E:/x/3.sxm", "context_paths_json": '["E:/x/1.sxm", "E:/x/2.sxm"]', "head_rows": 8, "step_idx": 2,
           "stopped": True}]
    n = S.normalize_manifest(t1)
    assert n[0]["task"] == "T1" and n[0]["label"] == 1 and n[0]["group"] == "Rig_B:2025-05-12"
    assert n[0]["input_paths"] == ["E:/x/1.sxm", "E:/x/2.sxm", "E:/x/3.sxm"]   # no rel_path → the absolute path
    with_rel = [dict(t1[0], rel_path="x/3.sxm", context_paths_json='["x/1.sxm", "x/2.sxm"]')]
    assert S.normalize_manifest(with_rel)[0]["input_paths"] == ["x/1.sxm", "x/2.sxm", "x/3.sxm"]
    t2 = [{"sample_id": "T2:a", "group_key": "g", "split": "test", "path": "p.sxm", "size_nm": 200.0,
           "next_u": 0.25, "next_v": -0.1, "next_w_rel": 0.2, "next_h_rel": 0.1}]
    n2 = S.normalize_manifest(t2)[0]
    assert n2["task"] == "T2" and n2["label"] == [50.0, -20.0] and n2["range_nm"] == 200.0 and n2["height_nm"] == 200.0
    assert n2["box_nm"] == pytest.approx(40.0) and n2["box_h_nm"] == pytest.approx(20.0)   # no height → square
    t3 = [{"sample_id": "T3:a", "group_key": "g", "split": "test", "path": "p.sxm", "next_action": "long_stop", "relocate": True}]
    assert S.normalize_manifest(t3)[0]["label"] == "long_stop"                # three-class is the default
    assert S.normalize_manifest(t3, t3_binary=True)[0]["label"] == "relocate"
    t4 = [{"sample_id": "T4:a", "group_key": "g", "split": "test", "material": "Au", "size_nm": 100.0,
           "bias_v": 0.1, "t_fwd_s": 0.5, "speed_nm_s": 200.0, "setpoint_a": float("nan")}]
    n4 = S.normalize_manifest(t4)[0]
    assert n4["task"] == "T4" and n4["input_paths"] == [] and n4["label"]["setpoint_a"] is None
    # contract rows pass through untouched; an unknown shape is refused loudly
    assert S.normalize_manifest(_t1_manifest())[0]["label"] == 1
    with pytest.raises(ValueError):
        S.normalize_manifest([{"sample_id": "x", "foo": 1}])


def test_t4_scoring_and_median_baseline_with_nan_setpoint():
    rows = [{"sample_id": f"t{i}", "group_key": "g", "split": s, "material": m, "size_nm": 100.0,
             "bias_v": b, "t_fwd_s": 0.5, "speed_nm_s": sp, "setpoint_a": float("nan")}
            for i, (s, m, b, sp) in enumerate([("train", "Au", 0.1, 200.0), ("train", "Au", 0.2, 100.0),
                                               ("test", "Au", 0.15, 150.0), ("test", "Au", 1.0, 20.0)])]
    ans = run_baseline("material_median", rows, split="test")
    a0 = parse_answer(ans[0]["answer"])
    assert a0["bias_v"] == pytest.approx(0.15) and "setpoint_a" not in a0 and a0["scan_speed_nm_per_s"] == 150.0
    res = S.score_t4(rows, ans, split="test")
    assert res["n_with_values"] == 2 and res["n_with_setpoint"] == 0 and res["hit_rate_setpoint"] is None
    assert res["hit_rate_bias"] == 0.5 and res["hit_rate_speed"] == 0.5 and res["hit_rate_both"] is None
    assert res["mean_log_likelihood"] is not None


@requires_mast
def test_head_rows_render_hides_the_tail(tmp_path):
    from stmbench.trackA.render import render_frame
    w = _world(tmp_path)
    full = _frame(w)
    part = _frame(w, rows=12)
    for p in (full, part):
        r = render_frame(p, head_rows=8)
        assert png_shape(r.png) == (8, 48) and "acq_frac" not in r.summary and "acq_rows" not in r.summary
        assert r.summary["head_rows_shown"] == 8 and r.summary["rows_total"] == 48
