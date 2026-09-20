"""Mode H: a person's replies go through the same loop, tool surface, budget and judge as a
model's — the port is the only thing that differs. Plus the pictures the person gets."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_mast

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


# ── no MAST needed ─────────────────────────────────────────────────────────
def test_pixel_to_scan_nm_is_the_inverse_of_the_judges_footprint_test():
    """A position the person reads off a frame must be a position the judge finds covered by
    that frame — same centre, same axes, same rotation sense, row 0 at the high-v edge."""
    from stmbench.human.render import pixel_to_scan_nm
    from stmbench.trackB.claims import frame_covering

    geom = {"cx_nm": 120.0, "cy_nm": -40.0, "w_nm": 80.0, "h_nm": 50.0, "angle_deg": 33.0, "nx": 64, "ny": 40}
    fr = {"kind": "scan_saved", "complete": True, "cx_m": 120e-9, "cy_m": -40e-9, "w_m": 80e-9, "h_m": 50e-9,
          "angle_deg": 33.0, "nx": 64, "ny": 40, "idx": 1}
    rule = {"min_fov_nm": 10, "max_nm_per_px": 5}
    for col, row in [(0, 0), (63, 0), (0, 39), (63, 39), (31.5, 19.5)]:
        x, y = pixel_to_scan_nm(geom, col, row)
        assert frame_covering([fr], x, y, rule) is fr, (col, row, x, y)
    x, y = pixel_to_scan_nm(geom, -1.2, 0)          # a pixel past the edge is outside
    assert frame_covering([fr], x, y, rule) is None
    g0 = {**geom, "angle_deg": 0.0}
    assert pixel_to_scan_nm(g0, 31.5, 19.5) == pytest.approx((120.0, -40.0))   # centre pixel = scan offset
    assert pixel_to_scan_nm(g0, 31.5, 0)[1] > pixel_to_scan_nm(g0, 31.5, 39)[1]  # row 0 is the high-y edge
    assert pixel_to_scan_nm(g0, 0, 19.5)[0] < pixel_to_scan_nm(g0, 63, 19.5)[0]


def test_flatten_removes_a_plane_and_keeps_unacquired_rows_nan():
    from stmbench.human.render import flatten

    yy, xx = np.mgrid[0:16, 0:16]
    z = 3.0 * xx + 2.0 * yy + 5.0
    z[-1] = np.nan
    f = flatten(z, "plane")
    assert np.nanmax(np.abs(f[:-1])) < 1e-9 and np.isnan(f[-1]).all()
    ln = flatten(z, "line")
    assert np.allclose(ln[0], ln[5]) and np.isnan(ln[-1]).all()
    assert np.array_equal(flatten(z, "none")[:-1], z[:-1])


def test_frame_geometry_reads_a_controller_header():
    from stmbench.human.render import frame_geometry

    h = {"scan_range": "1.200000E-07       6.000000E-08", "scan_offset": "5.000000E-08      -2.000000E-08",
         "scan_angle": "3.000000E+01", "scan_pixels": [384, 192]}
    g = frame_geometry(h)
    assert g == {"cx_nm": pytest.approx(50.0), "cy_nm": pytest.approx(-20.0), "w_nm": pytest.approx(120.0),
                 "h_nm": pytest.approx(60.0), "angle_deg": 30.0, "nx": 384, "ny": 192}
    assert frame_geometry({}) is None


def test_message_and_tool_records_are_plain_json():
    from stmbench.human.port import STUB_PREFIX, message_record, strip_markers, tool_record

    class Msg:
        type = "ai"
        content = [{"type": "text", "text": "看一下。"}, {"type": "thinking", "thinking": "草稿"}]
        tool_calls = [{"name": "GetBias", "args": {"a": 1}, "id": "t1", "type": "tool_call"}]

    r = message_record(Msg())
    assert r == {"role": "ai", "text": "看一下。", "tool_calls": [{"name": "GetBias", "args": {"a": 1}, "id": "t1"}]}

    class Tool:
        name = "StartScan"
        description = STUB_PREFIX + " scan] 开始扫描"
        schema = {"type": "object", "properties": {}}
        touches_instrument = True

    t = tool_record(Tool())
    assert t["stub"] and t["touches_instrument"] and t["name"] == "StartScan"
    json.dumps(r), json.dumps(t)
    assert strip_markers("备注 [DONE]") == "备注" and strip_markers("[ABORT] x") == "x"


def test_session_hands_actions_to_the_port_and_snapshots_plain_json():
    from stmbench.human.port import Action, HumanSession

    s = HumanSession()
    s.begin({"id": "X"}, seed=3, time_scale=20.0, policy="default", out="o", max_sim_s=100.0, max_cmds=10)
    snap = s.snapshot()
    assert snap["phase"] == "starting" and snap["seed"] == 3 and snap["budget"]["sim_max_s"] == 100.0
    with pytest.raises(RuntimeError):
        s.submit(Action(kind="tool", name="GetBias"))         # nobody is waiting for a reply yet

    class Req:
        tools = []
        messages = []
        system_prompt = "sys"
        state = {"loaded_tool_packs": ["scan"]}

    s.publish_request(Req(), call=1)
    assert s.snapshot()["phase"] == "awaiting_action" and s.transcript()["system_prompt"] == "sys"
    v = s.version
    threading.Thread(target=lambda: (time.sleep(0.05), s.submit(Action(kind="text", text="[DONE]")))).start()
    a = s.wait_action()
    assert a.kind == "text" and a.text == "[DONE]" and s.snapshot()["phase"] == "running"
    assert s.version != v and s.snapshot()["events"][0]["kind"] == "human_action"
    json.dumps(s.snapshot())
    s.cancel()
    assert s.wait_action() is None


# ── with MAST ──────────────────────────────────────────────────────────────
def _tunnelling_world(tmp_path, seed: int, time_scale: float):
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World

    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "s", time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


@requires_mast
def test_frames_and_spectra_render_for_the_person(tmp_path):
    from stmsim.io.dat_writer import write_dat

    from stmbench.human import render

    w = _tunnelling_world(tmp_path, seed=5, time_scale=20.0)
    w.scan.nx = w.scan.ny = 32
    w.scan.w = w.scan.h = 20e-9
    w.scan.cx, w.scan.cy, w.scan.angle_deg = 30e-9, -10e-9, 15.0
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.02
    w.scan_start()
    time.sleep(0.6)
    w.scan_stop()
    path = w.save_frame()
    assert path and Path(path).exists()
    png, meta = render.render_frame(path, channel="Z", direction="forward", flatten_mode="plane", max_px=64)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and meta["nx"] == 32 and meta["ny"] == 32 and meta["unit"] in ("pm", "nm")
    assert meta["hi"] > meta["lo"]
    bwd, _ = render.render_frame(path, direction="backward", flatten_mode="line")
    assert bwd[:8] == b"\x89PNG\r\n\x1a\n"
    fm = render.frame_meta(path)
    assert fm["cx_nm"] == pytest.approx(30.0) and fm["cy_nm"] == pytest.approx(-10.0)
    assert fm["w_nm"] == pytest.approx(20.0) and fm["angle_deg"] == pytest.approx(15.0) and fm["nx"] == 32
    assert "Z" in fm["channels"]
    # the geometry the page maps mouse positions through is the geometry the judge recorded
    ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
    assert (ev["cx_m"] * 1e9, ev["cy_m"] * 1e9, ev["angle_deg"]) == pytest.approx((fm["cx_nm"], fm["cy_nm"], fm["angle_deg"]))
    # a spectrum
    dat = tmp_path / "s" / "spec001.dat"
    v = np.linspace(-1.0, 1.0, 50)
    write_dat(dat, header={"Experiment": "bias spectroscopy", "X (m)": 1e-9, "Y (m)": 2e-9, "Bias>Bias (V)": 0.1},
              names=["Bias (V)", "Current (A)"], columns=[v, v ** 2 * 1e-10])
    assert render.render_dat(dat)[:8] == b"\x89PNG\r\n\x1a\n"
    dm = render.dat_meta(dat)
    assert dm["x_nm"] == pytest.approx(1.0) and dm["columns"] == ["Bias (V)", "Current (A)"] and dm["n_points"] == 50
    files = render.list_files(tmp_path / "s")
    assert {f["kind"] for f in files} == {"sxm", "dat"} and files[0]["mtime"] >= files[-1]["mtime"]


def _wait_phase(get, phase: str, timeout_s: float = 240.0) -> dict:
    t0 = time.time()
    st = get("/api/state")
    while st["phase"] != phase:
        if st["phase"] in ("error",):
            raise AssertionError(f"episode error: {st.get('error')}")
        if time.time() - t0 > timeout_s:
            raise AssertionError(f"timeout waiting for phase {phase!r}; at {st['phase']!r}")
        st = get(f"/api/state?wait=5&v={st['version']}&seq={st['seq']}")
    return st


@requires_mast
def test_gui_server_runs_an_episode_through_the_real_loop(tmp_path):
    """The whole path a person takes: start → tools offered (stubs included) → a tool call
    reaches the simulator → ReportResult folds → [DONE] → the ledger is an ordinary episode."""
    from stmbench.human.port import STUB_PREFIX
    from stmbench.human.server import BackgroundServer
    from stmbench.report.summarize import load_rows

    with BackgroundServer(tmp_path / "runs", scenario_dir=SCEN, time_scale=20.0) as srv:
        base = srv.url.rstrip("/")

        def get(path):
            with urllib.request.urlopen(base + path, timeout=90) as r:
                return json.load(r)

        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)

        page = urllib.request.urlopen(base + "/", timeout=30).read().decode("utf-8")
        assert "<title>STM-Bench" in page and "/api/state" in page
        scen = get("/api/scenarios")["scenarios"]
        assert scen[0]["id"].startswith("P") and scen[0]["task"] and scen[0]["claims"]
        assert not any("tol" in c for s in scen for c in s.get("claims", [])), "tolerances stay hidden"
        r = post("/api/start", {"scenario": "P1_barth1990_au111", "seed": 0, "time_scale": 20})
        assert r["ok"] and r["seq"] == 1
        with pytest.raises(urllib.error.HTTPError) as ei:
            post("/api/start", {"scenario": "P1_barth1990_au111", "seed": 1})
        assert ei.value.code == 409

        st = _wait_phase(get, "awaiting_action")
        tools = st["request"]["tools"]
        names = {t["name"] for t in tools}
        assert {"ReportResult", "ReportTipState", "GetBias"} <= names
        stubs = [t for t in tools if t["stub"]]
        assert stubs and all(t["description"].startswith(STUB_PREFIX) for t in stubs)
        rr = next(t for t in tools if t["name"] == "ReportResult")
        assert rr["schema"]["properties"]["claim_id"]["enum"] == [c["id"] for c in st["scenario"]["claims"]]
        assert st["budget"]["clock_paused"] is True and st["budget"]["sim_max_s"] == 3.0 * 3600
        tr = get("/api/transcript")
        assert tr["call"] == 1 and tr["messages"][0]["role"] == "human" and "ReportResult" in tr["system_prompt"]
        assert st["session_dir"] and st["run_dir"] and "H_human_default" in st["run_dir"]

        post("/api/action", {"kind": "tool", "name": "GetBias", "args": {}, "note": "看偏压 [DONE]"})
        st = _wait_phase(get, "awaiting_action")
        tr = get("/api/transcript")
        assert tr["messages"][-1]["role"] == "tool" and tr["messages"][-1]["name"] == "GetBias"
        assert tr["messages"][-2]["tool_calls"][0]["name"] == "GetBias" and "[DONE]" not in tr["messages"][-2]["text"]
        assert st["budget"]["cmds_used"] >= 1 and st["calls"] == 2
        kinds = [e["kind"] for e in st["events"]]
        assert "human_action" in kinds and "tool_start" in kinds and "tool_end" in kinds

        post("/api/action", {"kind": "tool", "name": "ReportResult",
                             "args": {"claim_id": "stripe_period_nm", "value": 6.3, "unit": "nm"}})
        st = _wait_phase(get, "awaiting_action")
        assert "尚未报告" in get("/api/transcript")["messages"][-1]["text"]
        assert get("/api/files")["files"] == []
        with pytest.raises(urllib.error.HTTPError) as ei:
            get("/api/frame?f=../episode.json")
        assert ei.value.code == 404

        post("/api/action", {"kind": "text", "text": "想一想。"})          # no marker: the continue nudge
        st = _wait_phase(get, "awaiting_action")
        assert any("继续" in m["text"] for m in get("/api/transcript")["messages"] if m["role"] == "human")

        post("/api/action", {"kind": "text", "text": "先到这里 [DONE]"})
        st = _wait_phase(get, "finished")
        res = st["result"]
        assert res["mode"] == "H" and res["model_id"] == "human" and res["success"] is False
        assert res["claims_reported"] == 1 and res["claims_verified"] == 0    # a number with no frame behind it
        sr = res["skill_result"]
        assert sr["outcome"] == "done" and sr["tool_sequence"] == ["GetBias", "ReportResult"]
        assert sr["billing"]["tokens_in"] == 0 and sr["billing"]["cost_known"] is True
        assert sr["clock_paused"] is True and sr["model_wall_s"] > 0
        assert res["billing"]["n_calls"] == 0
        out_dir = Path(res["out_dir"])
        assert (out_dir / "episode.json").exists() and "H_human_default" in str(out_dir)
        # what a replay needs: the instrument clock on every model step, the tip's history
        drv = json.loads((out_dir / "driver.json").read_text(encoding="utf-8"))
        tools = [e for e in drv["events"] if e["kind"] in ("tool_start", "tool_end")]
        assert tools and all(isinstance(e.get("sim_s"), float) for e in tools)
        assert all(a["sim_s"] <= b["sim_s"] for a, b in zip(tools, tools[1:]))
        tt = json.loads((out_dir / "tip_timeline.json").read_text(encoding="utf-8"))
        assert tt["initial"] == res["truth_before"]["tip"] and isinstance(tt["events"], list)
        rows = load_rows(tmp_path / "runs")
        assert len(rows) == 1 and rows[0]["mode"] == "H" and rows[0]["model"] == "human" and rows[0]["paper"] == "P1"

        # a second episode in the same process; cancelling it ends with [ABORT]
        r = post("/api/start", {"scenario": "P1_barth1990_au111", "seed": 1, "time_scale": 20})
        assert r["seq"] == 2
        st = _wait_phase(get, "awaiting_action")
        assert st["seq"] == 2 and st["seed"] == 1 and st["calls"] == 1
        post("/api/cancel", {})
        st = _wait_phase(get, "finished")
        assert st["result"]["skill_result"]["outcome"] == "abort" and st["result"]["seed"] == 1
        assert len(get("/api/history")["history"]) == 2
