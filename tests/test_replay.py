"""The audience replay: what the ledger now records for it (the tip's history, the
instrument clock on every model step) and how a ledger becomes a timeline, words and a
page. The replay is built after the episode and never feeds a verdict."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_mast

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


def _tunnelling_world(tmp_path, seed: int = 3, time_scale: float = 20.0):
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World

    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "s", time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


# ── what the ledger records ─────────────────────────────────────────────────
def test_tip_events_carry_the_state_they_left():
    from stmsim.physics.tip import Tip

    tip = Tip(rng=np.random.default_rng(4))
    tip.crash(12.0)
    ev = tip.events[-1]
    assert ev.kind == "crash" and ev.sim_s == 12.0
    assert ev.state["flicker_dz_pm"] > 0 and ev.state["n_apex"] == tip.n_apex and ev.state["metastable"]
    tip.pick_up(13.0, "Fe")
    assert tip.events[-1].state["carried"] == "Fe"
    tip.drop(14.0)
    assert tip.events[-1].state["carried"] is None
    tip.spontaneous_change(15.0)
    assert tip.events[-1].state["n_apex"] == tip.n_apex        # the state after, not before


def test_scan_start_names_the_tip_events_its_render_drew(tmp_path):
    """Spontaneous changes are drawn at scan start with the time of the row they hit; the
    event says which slice of tip.events is this frame's, so a replay can place them."""
    w = _tunnelling_world(tmp_path)
    w.tip.lambda_per_s = 0.2                                    # a change every few rows
    w.scan.nx = w.scan.ny = 32
    w.scan.w = w.scan.h = 20e-9
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.5
    w.scan_start()
    ev = [e for e in w.events if e["kind"] == "scan_start"][-1]
    assert ev["per_row_s"] == pytest.approx(1.0) and ev["scan_dir"] == "down"
    i0, i1 = ev["tip_events"]
    assert i1 > i0, "with this hazard the render must have drawn changes"
    t0 = w.frame.t_start_sim
    for e in w.tip.events[i0:i1]:
        assert e.kind == "spontaneous_change" and e.state is not None
        assert t0 - 1e-6 <= e.sim_s <= t0 + 32 * 1.0          # a row of this frame, not "now"


def test_the_harness_stamps_the_clock_and_dumps_the_tip(tmp_path):
    from stmbench.harness.episode import sim_stamped, tip_timeline

    w = _tunnelling_world(tmp_path)
    seen = []
    hook = sim_stamped(w, seen.append)
    rec = {"kind": "tool_start", "name": "ScanAt"}
    hook(rec)
    assert seen == [rec] and rec["sim_s"] == pytest.approx(w.clock.sim(), abs=5.0)
    sim_stamped(w)({"kind": "x"})                              # no downstream hook: still fine
    initial = w.tip.snapshot()
    w.tip.crash(w.clock.sim())
    tl = json.loads(json.dumps(tip_timeline(w, initial)))
    assert tl["initial"]["flicker_dz_pm"] == 0
    assert [e["kind"] for e in tl["events"]] == ["crash"] and tl["events"][0]["i"] == 0
    assert tl["events"][0]["state"]["flicker_dz_pm"] > 0


# ── words ───────────────────────────────────────────────────────────────────
def test_tool_arguments_become_numbers_and_sentences():
    from stmbench.replay import lay

    assert lay.si("40n") == pytest.approx(40e-9) and lay.si("8.355p") == pytest.approx(8.355e-12)
    assert lay.si(0.4) == 0.4 and lay.si("abc") is None and lay.si(True) is None
    assert lay.tool_caption("ScanAt", {"center_x_m": "5n", "center_y_m": "3.5n", "size_m": "16n", "pixels": 200}) \
        == "扫一幅 16 nm 见方的图，中心在 (5, 3.5) nm（200×200 像素）"
    cap = lay.tool_caption("MoveAtomTo", {"atom_x_m": "7.75n", "atom_y_m": "6.42n", "target_x_m": "11.085n",
                                          "target_y_m": "6.428n", "manip_bias_v": 0.008, "manip_setpoint_a": "85n"})
    assert "拖 3.34 nm" in cap and "94.1 kΩ" in cap
    assert lay.tool_caption("ReportResult", {"claim_id": "stripe_period_nm", "value": 6.35, "unit": "nm"}) \
        == "提交答案：条纹间距 = 6.35 nm"
    assert lay.tool_caption("SetDriftCompensation", {"enable": False}) == "关掉漂移补偿"
    assert lay.tool_caption("GetFooBar", {}) == "查看仪器读数（GetFooBar）"
    assert lay.result_text("搬运中止,仪器未还原 {'moved': None}") == "搬运中止,仪器未还原"
    assert lay.result_text("{'enabled': True}") == "" and lay.result_text("E:/x/y.sxm ok") == ""
    assert lay.is_stub_reply("ScanAt 属于尚未加载的工具包 scan，这一次没有执行。")


def test_the_tip_is_described_by_what_it_does_to_the_image():
    from stmbench.replay.lay import tip_look

    sharp = {"radius_nm": 1.0, "n_apex": 1, "multi": False, "flicker_dz_pm": 0.0}
    assert tip_look(sharp)["kind"] == "sharp" and tip_look(sharp)["level"] == "good"
    assert tip_look({**sharp, "radius_nm": 4.0})["level"] == "warn"
    assert tip_look({**sharp, "radius_nm": 9.0})["level"] == "bad"
    double = tip_look({**sharp, "n_apex": 2, "multi": True})
    assert double["kind"] == "double" and double["cartoon"]["points"] == 2
    crashed = tip_look({**sharp, "n_apex": 3, "multi": True, "flicker_dz_pm": 150.0, "radius_nm": 5.0})
    assert crashed["kind"] == "unstable" and crashed["cartoon"]["wobble"]          # the worst reason wins
    assert tip_look({**sharp, "carried": "Fe"})["kind"] == "carrying"
    assert tip_look({**sharp, "dead": True, "flicker_dz_pm": 150.0})["kind"] == "dead"
    assert tip_look(None)["kind"] == "sharp"


def test_claims_are_retold_with_the_truth():
    from stmbench.replay.lay import claim_verdict

    row = claim_verdict("stripe_period_nm", {"kind": "scalar", "tol": {"abs": 0.15}, "reported": True, "value": 5.95,
                                             "truth": 6.364, "value_ok": False, "evidence_ok": True}, {"unit": "nm"})
    assert row["ai"] == "5.95 nm" and "6.364 nm" in row["truth"] and "±0.15 nm" in row["truth"] and not row["ok"]
    pos = claim_verdict("target_site", {"kind": "position", "reported": True, "value": 11.3, "value_ok": True,
                                        "evidence_ok": True, "truth": {"on_target": True, "site_nm": [0, 0]}},
                        {"unit": "nm"}, {"x_nm": 11.317, "y_nm": 6.594})
    assert pos["ok"] and pos["ai"] == "(11.3, 6.59) nm" and pos["truth"] == "原子正停在目标格点上"
    miss = claim_verdict("target_site", {"kind": "position", "reported": True, "value": 0, "value_ok": False,
                                         "evidence_ok": False, "truth": {"on_target": False, "site_nm": [0, 0],
                                                                         "atom_now_nm": [3, 4]}})
    assert "5 nm" in miss["truth"]
    con = claim_verdict("bystander_max_shift_nm", {"kind": "constraint", "reported": True, "truth": 0.0,
                                                   "value_ok": True, "evidence_ok": False},
                        {"unit": "nm", "max": 0.2}, {"value": 0.026})
    assert con["ai"] == "0.026 nm" and "上限 0.2 nm" in con["truth"] and "不算" in con["note"]


# ── ledgers → timeline ──────────────────────────────────────────────────────
def _ledger(tmp_path, *, driver, events, tip_timeline=None, tip_before=None, tip_after=None, adatoms=None,
            success=True, sim0=100.0, sim1=2000.0):
    from stmbench.replay.timeline import Ledger

    d = tmp_path / "run"
    (d / "session").mkdir(parents=True, exist_ok=True)
    tip0 = tip_before or {"radius_nm": 1.0, "n_apex": 1, "multi": False, "flicker_dz_pm": 0.0}
    ep = {"scenario_id": "P4_atom_positioning_cu111", "seed": 0, "mode": "H", "model_id": "test", "run_id": "r",
          "success": success, "time_scale": 20.0, "caps": {"max_sim_s": 5400.0},
          "truth_before": {"sim_s": sim0, "wall_s": sim0 / 20, "tip": tip0, "adatoms": adatoms},
          "truth_after": {"sim_s": sim1, "tip": tip_after or tip0},
          "verdict": {"details": {"claims": {}}}}
    (d / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    (d / "driver.json").write_text(json.dumps({"events": driver, "final_text": "done [DONE]"}), encoding="utf-8")
    (d / "events.json").write_text(json.dumps(events), encoding="utf-8")
    if tip_timeline is not None:
        (d / "tip_timeline.json").write_text(json.dumps(tip_timeline), encoding="utf-8")
    return Ledger.load(d)


def _tool(k, name, t0, t1, args=None, preview="", sim=None):
    s = {"kind": "tool_start", "tool_call_id": f"c{k}", "name": name, "t": t0, "args": args or {}}
    e = {"kind": "tool_end", "tool_call_id": f"c{k}", "name": name, "t": t1, "ok": True, "preview": preview}
    if sim is not None:
        s["sim_s"], e["sim_s"] = sim
    return [s, e]


def test_an_old_ledger_gets_its_clock_back_from_the_world_events(tmp_path):
    """Wall times only: the instrument clock ran only inside tool calls, plus a start-up the
    fit has to find (here 80 s of instrument time)."""
    from stmbench.replay.timeline import build_timeline

    t0 = 1_000_000.0
    drv = (_tool(0, "load_tool_pack", t0, t0, {"pack": "scan"})
           + _tool(1, "ScanAt", t0 + 30, t0 + 55.6, {"size_m": "40n"})          # 512 s of instrument time
           + _tool(2, "ScanAt", t0 + 200, t0 + 225.6, {"size_m": "20n"})
           + [{"kind": "message", "role": "assistant", "text": "看完了 [DONE]", "t": t0 + 300}])
    start = 100.0 + 80.0
    events = [{"kind": "scan_start", "sim_s": start + 2, "line_s": 1.0, "px": 256},
              {"kind": "scan_saved", "sim_s": start + 512, "t_start_sim": start + 2, "rows": 256, "complete": True,
               "path": "x/f001.sxm", "idx": 1, "nx": 256, "ny": 256, "w_m": 40e-9, "h_m": 40e-9, "drift_m": [0, 0]},
              {"kind": "scan_start", "sim_s": start + 514, "line_s": 1.0, "px": 256},
              {"kind": "scan_saved", "sim_s": start + 1024, "t_start_sim": start + 514, "rows": 256, "complete": True,
               "path": "x/f002.sxm", "idx": 2, "nx": 256, "ny": 256, "w_m": 20e-9, "h_m": 20e-9, "drift_m": [0, 0]}]
    tl = build_timeline(_ledger(tmp_path, driver=drv, events=events, sim1=start + 1100))
    clk = tl["clock"]
    assert clk["source"] == "calibrated" and clk["covered"] == 4 and clk["shift_s"] == pytest.approx(80.0, abs=45.0)
    s1, s2 = tl["steps"][1], tl["steps"][2]
    assert s1["sim0"] <= events[0]["sim_s"] <= s1["sim1"] + 1 and s2["sim0"] <= events[2]["sim_s"] <= s2["sim1"] + 1
    assert all(a["sim1"] <= b["sim0"] + 1e-9 for a, b in zip(tl["steps"], tl["steps"][1:]))
    assert s2["think_s"] == pytest.approx(200 - 55.6)
    assert [f["file"] for f in tl["frames"]] == ["f001.sxm", "f002.sxm"] and tl["frames"][0]["per_row_s"] == 2.0
    assert tl["messages"][0]["text"] == "看完了" and tl["final_text"] == "done"


def test_stamps_and_the_tip_timeline_are_used_as_recorded(tmp_path):
    """A change drawn by a frame that was stopped early shows from where the frame stopped;
    a change of look is a moment."""
    from stmbench.replay.timeline import build_timeline

    drv = _tool(0, "ScanAt", 10.0, 20.0, {"size_m": "20n"}, sim=(200.0, 400.0)) \
        + _tool(1, "ScanAt", 30.0, 40.0, {"size_m": "20n"}, sim=(420.0, 700.0))
    events = [{"kind": "scan_start", "sim_s": 201.0, "line_s": 0.5, "per_row_s": 1.0, "px": 256, "scan_dir": "down",
               "tip_events": [0, 1]},
              {"kind": "scan_saved", "sim_s": 300.0, "t_start_sim": 201.0, "rows": 99, "complete": False,
               "path": "f1.sxm", "idx": 1, "nx": 256, "ny": 256, "w_m": 20e-9, "h_m": 20e-9, "drift_m": [0, 0]},
              {"kind": "scan_start", "sim_s": 421.0, "line_s": 0.5, "per_row_s": 1.0, "px": 256, "tip_events": [1, 1]}]
    double = {"radius_nm": 1.2, "n_apex": 2, "multi": True, "flicker_dz_pm": 0.0}
    tt = {"initial": {"radius_nm": 1.0, "n_apex": 1, "multi": False, "flicker_dz_pm": 0.0},
          "events": [{"i": 0, "sim_s": 450.0, "kind": "spontaneous_change", "detail": {"dz_pm": 80.0}, "state": double}]}
    tl = build_timeline(_ledger(tmp_path, driver=drv, events=events, tip_timeline=tt, sim1=800.0))
    assert tl["clock"]["source"] == "stamped" and tl["steps"][0]["sim0"] == 200.0 and tl["steps"][1]["sim1"] == 700.0
    assert tl["tip_source"] == "timeline"
    assert [round(e["sim"]) for e in tl["tip"]] == [100, 300]      # clamped to the stopped frame's last row
    assert tl["tip"][1]["look"]["kind"] == "double"
    tips = [m for m in tl["moments"] if m["kind"] == "tip"]
    assert tips and tips[0]["title"] == "针尖分叉了" and tips[0]["level"] == "warn"
    assert tl["frames"][1]["file"] is None and tl["frames"][1]["rows_done"] == 256


def test_an_old_ledger_infers_the_crash_and_keeps_the_true_end_state(tmp_path):
    from stmbench.replay.timeline import build_timeline

    after = {"radius_nm": 4.5, "n_apex": 3, "multi": True, "flicker_dz_pm": 152.0, "metastable": True}
    events = [{"kind": "crash", "sim_s": 900.0, "reason": "z_set_into_surface", "outcome": "crashed", "severity": 1.1,
               "n_apex": 2, "flicker_dz_pm": 152.0},
              {"kind": "adatom_lost", "sim_s": 900.0, "atom_ids": [1], "cause": "crash", "xy_nm": [0, 0]}]
    tl = build_timeline(_ledger(tmp_path, driver=[], events=events, tip_after=after))
    assert tl["tip_source"] == "reconstructed"
    assert [e["look"]["kind"] for e in tl["tip"]] == ["sharp", "unstable"] and tl["tip"][1]["state"]["n_apex"] == 3
    kinds = [m["kind"] for m in tl["moments"] if m["sim"] == 900.0]
    assert kinds == ["crash", "tip", "atom_lost"]                   # cause before consequences
    assert tl["moments"][-1]["kind"] == "end"


def test_hops_become_drags_and_a_lost_atom_is_marked(tmp_path):
    from stmbench.replay.timeline import build_timeline

    adatoms = {"sites": [{"id": 1, "role": "target", "x_nm": 0.0, "y_nm": 0.0, "status": "on_surface"},
                         {"id": 2, "role": "bystander", "x_nm": 8.0, "y_nm": 0.0, "status": "on_surface"}],
               "lattice": {"a_nm": 0.255, "angle_deg": 0.0, "origin_nm": [0.0, 0.0]}, "params": {"sigma_nm": 0.25},
               "target": {"atom_id": 1, "site_nm": [1.0, 0.0], "start_nm": [0.0, 0.0]}}
    hop = lambda t, x: {"kind": "adatom_hop", "sim_s": t, "atom_id": 1, "cause": "folme", "n_hops": 1,
                        "to_xy_nm": [x, 0.0]}
    events = [hop(500.0, 0.25), hop(500.5, 0.5), hop(501.0, 0.75), hop(700.0, 1.0),
              {"kind": "adatom_lost", "sim_s": 900.0, "atom_ids": [2], "cause": "crash"}]
    tl = build_timeline(_ledger(tmp_path, driver=[], events=events, adatoms=adatoms))
    drags = [m for m in tl["moments"] if m["kind"] == "drag"]
    assert [d["title"] for d in drags] == ["目标原子被针尖拖动了 0.75 nm", "目标原子被针尖拖动了 0.25 nm"]
    assert tl["atoms"]["target"]["site"] == [1.0, 0.0]
    assert tl["atoms"]["moves"][-1] == {"sim": 900.0, "id": 2, "status": "lost", "cause": "crash"}
    from stmbench.replay.truth_view import AtomsAt
    at = AtomsAt(tl["atoms"])
    assert at(600.0)[1]["x"] == 0.75 and at(950.0)[2]["status"] == "lost" and at.key(501.0) == 3


def test_the_page_is_one_file_with_its_data_or_a_folder(tmp_path):
    from stmbench.replay.build import write_bundle

    data = {"meta": {"page_title": "AI 用针尖搬原子"}, "timeline": {"note": "</script><!-- x"},
            "frame": "frames/001.png"}
    one = write_bundle(data, {"frames/001.png": b"\x89PNG fake"}, tmp_path / "a", single_file=True, name="r")
    txt = one.read_text(encoding="utf-8")
    assert txt.startswith("<!doctype html>") and "<title>AI 用针尖搬原子</title>" in txt
    assert "data:image/png;base64," in txt and "<\\/script><\\!-- x" in txt and 'src="replay.js"' not in txt
    bare = write_bundle(data, {}, tmp_path / "b", single_file=True, name="r", bare=True)
    assert not bare.read_text(encoding="utf-8").startswith("<!doctype")
    idx = write_bundle(data, {"frames/001.png": b"\x89PNG fake"}, tmp_path / "c", single_file=False, name="r")
    assert idx.name == "index.html" and (tmp_path / "c" / "frames" / "001.png").is_file()
    assert (tmp_path / "c" / "replay.js").read_text(encoding="utf-8").startswith("window.REPLAY = ")


# ── the whole path, with the real simulator and MAST's frame reader ─────────
@requires_mast
def test_a_replay_is_built_from_a_ledger_and_checks_its_truth(tmp_path):
    from stmsim.scenario import Scenario

    from stmbench.harness.episode import tip_timeline
    from stmbench.replay.build import build_replay

    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    run = tmp_path / "run"
    w = sc.build_world(0, session_dir=run / "session")
    before = json.loads(json.dumps(w.truth(), default=float))
    w.scan.nx = w.scan.ny = 48
    w.scan.w = w.scan.h = 30e-9
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.05
    w.scan_start()
    time.sleep(0.4)
    w.scan_stop()
    assert w.save_frame()
    after = json.loads(json.dumps(w.truth(), default=float))
    ev_start = [e for e in w.events if e["kind"] == "scan_start"][-1]
    ep = {"scenario_id": sc.id, "seed": 0, "mode": "A", "model_id": "test-model", "run_id": "t", "success": False,
          "time_scale": sc.time_scale, "caps": {"max_sim_s": 5400.0}, "truth_before": before, "truth_after": after,
          "verdict": {"details": {"claims": {"target_site": {"kind": "position", "reported": False, "truth": {}}}}}}
    (run / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    (run / "events.json").write_text(json.dumps(w.events, default=str), encoding="utf-8")
    (run / "tip_timeline.json").write_text(json.dumps(tip_timeline(w, before["tip"]), default=float), encoding="utf-8")
    drv = _tool(0, "ScanAt", 1.0, 2.0, {"size_m": "30n"}, sim=(ev_start["sim_s"] - 0.5, after["sim_s"]))
    (run / "driver.json").write_text(json.dumps({"events": drv}), encoding="utf-8")

    page = build_replay(run, tmp_path / "out", single_file=True, log=lambda *a: None)
    txt = page.read_text(encoding="utf-8")
    data = json.loads(txt.split("window.REPLAY = ", 1)[1].split(";</script>", 1)[0].replace("<\\/", "</"))
    assert data["truth"]["available"], data["truth"]["reason"]
    fr = [f for f in data["timeline"]["frames"] if f.get("png")]
    assert len(fr) == 1 and fr[0]["png"].startswith("data:image/png;base64,") and fr[0]["ideal"].startswith("data:")
    assert data["truth"]["overview"]["png"].startswith("data:") and len(fr[0]["corners"]) == 4
    assert data["meta"]["page_title"] == "AI 用针尖搬原子" and data["timeline"]["clock"]["source"] == "stamped"
    assert "<title>AI 用针尖搬原子</title>" in txt

    # a ledger from another draw is not dressed up with this draw's sample
    ep["truth_before"]["adatoms"]["sites"][0]["x_nm"] += 1.0
    (run / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    page2 = build_replay(run, tmp_path / "out2", log=lambda *a: None)
    js = (page2.parent / "replay.js").read_text(encoding="utf-8")
    d2 = json.loads(js[len("window.REPLAY = "):-1].replace("<\\/", "</"))
    assert not d2["truth"]["available"] and "adatoms" in d2["truth"]["reason"]
    assert not (page2.parent / "overview.png").exists()
