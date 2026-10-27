"""The console (stmbench.console): the instrument in a web page. Its endpoints drive the same
instrument an MCP agent gets and refuse as it refuses; no reply carries the world's truth or the
verdict; a picked pixel maps to scan coordinates as the simulator lays the frame out; and the
episode leaves the ledger an MCP episode with the same calls leaves."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

from stmbench.console import Console, make_server
from stmbench.console import server as console_server
from stmbench.instrument import ACTIONS, ActionError, Instrument, verify
from stmbench.instrument.mcp_server import EpisodeServer

P4 = "P4_atom_positioning_cu111"
SCAN = {"center_x_nm": 0.0, "center_y_nm": 0.0, "size_nm": 8.0, "pixels": 32}
#: what a page must never carry
SECRET_KEYS = {"truth", "truth_before", "truth_after", "verdict", "success", "partial", "digest", "hidden"}


@contextmanager
def serving(console):
    srv = make_server(console, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield Client(f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        srv.shutdown()
        srv.server_close()


class Client:
    def __init__(self, base: str):
        self.base = base
        self.seen: list = []                   # every JSON reply

    def get(self, path: str, raw: bool = False):
        return self._call(urllib.request.Request(self.base + path), raw)

    def post(self, path: str, body: dict | None = None):
        return self._call(urllib.request.Request(self.base + path, data=json.dumps(body or {}).encode("utf-8"),
                                                 headers={"Content-Type": "application/json"}, method="POST"))

    def _call(self, req, raw=False):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                code, body, ctype = r.status, r.read(), r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            code, body, ctype = e.code, e.read(), e.headers.get("Content-Type", "")
        if raw:
            return code, body, ctype
        out = json.loads(body.decode("utf-8"))
        self.seen.append(out)
        return code, out


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _keys(v)


def test_an_episode_through_the_console(tmp_path):
    run = tmp_path / "run"
    with serving(Console(run, P4, 0)) as c:
        code, page, ctype = c.get("/", raw=True)
        assert code == 200 and ctype.startswith("text/html") and b"STM" in page
        _, s = c.get("/api/session")
        assert s["started"] and not s["finished"] and s["scenario_id"] == P4 and s["seed"] == 0
        assert s["claims"][0]["id"] == "target_site" and s["claims"][0]["say"]
        _, t = c.get("/api/task")
        assert "target_site" in t["task"] and "move_tip" in t["manual"]
        _, acts = c.get("/api/actions")
        assert [a["name"] for a in acts] == list(ACTIONS)

        code, out = c.post("/api/act", {"name": "scan", "args": SCAN})
        assert code == 200 and out["channels"] == ["z", "current"]
        frame = out["frame"]
        _, d = c.get("/api/data")
        assert [f["frame"] for f in d["frames"]] == [frame] and d["frames"][0]["channels"] == ["z", "current"]
        code, png, ctype = c.get(f"/api/frame.png?f={frame}&ch=current&dir=backward", raw=True)
        assert code == 200 and ctype == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n"
        _, v = c.get(f"/api/frame_values?f={frame}&max_size=8")
        assert v["shape"] == [8, 8] and v["block_px"] == 4 and v["unit"] == "pm"
        code, err, _ = c.get(f"/api/frame.png?f={frame}&ch=didv", raw=True)
        assert code == 400 and b"channels" in err                 # no lock-in map in this frame
        _, info = c.get("/api/info")
        assert info["material"] and "budget_s" in info

        code, err = c.post("/api/act", {"name": "scan", "args": {"center_x_nm": 0.0}})
        assert code == 400 and err["error"] == "scan: center_y_nm is required"       # the instrument's words
        code, err = c.post("/api/report", {"claim_id": "target_site", "x_nm": 1.0, "y_nm": 0.0})
        assert code == 400 and "value" in err["error"]
        code, r = c.post("/api/report", {"claim_id": "target_site", "value": 1.0, "x_nm": 1.0, "y_nm": 0.0,
                                         "frame": frame, "evidence": [frame]})
        assert code == 200 and r["recorded"].startswith("recorded target_site")
        _, st = c.get("/api/status")
        assert st["instrument_time_s"] > 0

        _, fin = c.post("/api/finish")
        assert fin == {"finished": True, "message": "The episode is over; the record is written."}
        code, err = c.post("/api/act", {"name": "wait", "args": {"seconds": 1}})
        assert code == 400 and "over" in err["error"]
        _, s = c.get("/api/session")
        assert s["finished"]
        code, _ = c.get("/api/nowhere")
        assert code == 404

    assert not SECRET_KEYS & set(_keys(c.seen)), "a reply carried the truth or the verdict"
    assert verify(run)["ok"]
    kinds = [json.loads(line)["kind"] for line in (run / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kinds.count("act") == 1 and kinds.count("refused") == 1 and kinds.count("report") == 1
    assert "diagnosis" not in kinds and kinds[-1] == "finish" and "observe" in kinds
    with pytest.raises(ActionError):
        Console(run, P4, 0)                                   # a run directory holds one episode


def test_the_console_and_an_mcp_agent_get_the_same_instrument(tmp_path):
    """The same calls through the page and through the MCP server: the same measurements and verdict."""
    report = {"claim_id": "target_site", "value": 1.0, "x_nm": 1.0, "y_nm": 0.0}
    moves = [("scan", SCAN), ("set_bias", {"bias_v": 0.005}), ("set_setpoint", {"current_pa": 100000.0}),
             ("move_tip", {"x_nm": 1.0, "y_nm": 0.0, "speed_nm_s": 1.0}), ("set_setpoint", {"current_pa": 200.0}),
             ("scan", SCAN)]
    with serving(Console(tmp_path / "page", P4, 0)) as c:
        frames = [c.post("/api/act", {"name": n, "args": a})[1].get("frame") for n, a in moves]
        c.post("/api/report", {**report, "frame": frames[-1], "evidence": [frames[-1]]})
        c.post("/api/finish")
    mcp = EpisodeServer(Instrument(P4, 0, out_dir=tmp_path / "mcp"))
    for i, (n, a) in enumerate(moves):
        mcp.handle({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": n, "arguments": a}})
    mcp.handle({"jsonrpc": "2.0", "id": 90, "method": "tools/call", "params": {
        "name": "report_result", "arguments": {**report, "frame": frames[-1], "evidence": [frames[-1]]}}})
    mcp.handle({"jsonrpc": "2.0", "id": 91, "method": "tools/call", "params": {"name": "finish", "arguments": {}}})
    page, agent = (json.loads((tmp_path / d / "episode.json").read_text(encoding="utf-8")) for d in ("page", "mcp"))
    assert page["digest"] == agent["digest"] and page["verdict"] == agent["verdict"]
    assert page["instrument_time_s"] == agent["instrument_time_s"]


def test_a_picked_pixel_maps_to_scan_coordinates_as_the_simulator_lays_the_frame_out(tmp_path):
    console = Console(tmp_path / "run", P4, 0)
    with serving(console) as c:
        _, out = c.post("/api/act", {"name": "scan", "args": {"center_x_nm": 2.0, "center_y_nm": -1.0, "size_nm": 8.0,
                                                               "pixels": 32, "angle_deg": 30.0}})
        st = console.inst.world.scan                          # the settings the frame was taken with
        for col, row in ((0, 0), (31, 0), (0, 31), (12, 20), (15.5, 15.5), (-0.5, -0.5)):
            _, p = c.get(f"/api/pixel?f={out['frame']}&col={col}&row={row}")
            x, y = st.pixel_xy(col, row)
            assert p["x_nm"] == pytest.approx(float(x) * 1e9, abs=1e-4)
            assert p["y_nm"] == pytest.approx(float(y) * 1e9, abs=1e-4)
        code, _ = c.get("/api/pixel?f=nosuch.sxm&col=0&row=0")
        assert code == 400
        c.post("/api/finish")


def test_without_a_scenario_the_console_offers_the_paper_scenarios(tmp_path):
    with serving(Console(tmp_path)) as c:
        _, s = c.get("/api/session")
        assert not s["started"] and P4 in s["scenarios"] and all(x.startswith("P") for x in s["scenarios"])
        code, err = c.get("/api/status")
        assert code == 400 and "scenario" in err["error"]
        for bad in ("B1_approach_noisy", f"../scenarios/{P4}", "nosuch"):
            code, _ = c.post("/api/start", {"scenario": bad, "seed": 1})
            assert code == 400
        code, s = c.post("/api/start", {"scenario": P4, "seed": 3})
        assert code == 200 and s["started"] and s["seed"] == 3
        code, _ = c.post("/api/start", {"scenario": P4, "seed": 4})
        assert code == 400                                    # one episode per console
        c.post("/api/finish")
    assert (tmp_path / f"{P4}_seed3" / "episode.json").is_file()


def test_the_page_calls_only_endpoints_the_server_serves():
    page = console_server.PAGE.read_text(encoding="utf-8")
    used = set(re.findall(r"/api/[a-z_.]+", page))
    with open(console_server.__file__, encoding="utf-8") as fh:
        served = set(re.findall(r'p == "(/api/[a-z_.]+)"', fh.read()))
    assert used and used <= served, used - served


# ── the page's script, run under node (no browser) ──────────────────────────
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _page_script() -> str:
    return re.search(r"<script>(.*?)</script>", console_server.PAGE.read_text(encoding="utf-8"), re.S).group(1)


@needs_node
def test_the_pages_script_is_valid_javascript(tmp_path):
    js = tmp_path / "page.js"
    js.write_text(_page_script(), encoding="utf-8")
    done = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@needs_node
def test_a_move_trace_opens_on_z_against_distance_and_is_selected_after_the_move(tmp_path):
    """What the page does with a move's reply and record, run on the real ones: the reply names the
    trace (not a spectrum), which the page selects like a spectrum, and its record opens on Z against
    the distance along the move; a sweep and a frame keep what they had. The units table knows
    the trace's columns."""
    with serving(Console(tmp_path / "run", P4, 0)) as c:
        _, move = c.post("/api/act", {"name": "move_tip", "args": {"x_nm": 3.0, "y_nm": 0.0, "speed_nm_s": 20.0}})
        _, trace = c.get("/api/spectrum?s=" + move["trace"])
        _, scan = c.post("/api/act", {"name": "scan", "args": SCAN})
    assert "spectrum" not in move and trace["kind"] == "move"
    pure = re.search(r"// <pure>[^\n]*\n(.*?)// </pure>", console_server.PAGE.read_text(encoding="utf-8"), re.S).group(1)
    sweep = {"kind": "bias sweep", "v": list(range(10)), "didv": [1.0] * 10, "i": [0.5] * 10}
    js = tmp_path / "pure.js"
    js.write_text(pure + """
const x = JSON.parse(require('fs').readFileSync(0, 'utf8'));
console.log(JSON.stringify({
  move: savedBy(x.move), frame: savedBy(x.scan), sweep: savedBy({ spectrum: 's.dat' }), none: savedBy({ tip_nm: [1, 2] }),
  trace_axes: defaultAxes(x.trace, numericCols(x.trace)), sweep_axes: defaultAxes(x.sweep, numericCols(x.sweep)),
  cols: numericCols(x.trace), units: [UNITS.dist, UNITS.t, UNITS.z_abs, UNITS.i],
}));
""", encoding="utf-8")
    done = subprocess.run(["node", str(js)], input=json.dumps({"move": move, "scan": scan, "trace": trace, "sweep": sweep}),
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert got["move"] == {"kind": "spectrum", "name": move["trace"]} and got["frame"] == {"kind": "frame", "name": scan["frame"]}
    assert got["sweep"] == {"kind": "spectrum", "name": "s.dat"} and got["none"] is None
    assert sorted(got["cols"]) == ["dist", "i", "t", "z_abs"] and got["trace_axes"] == ["dist", "z_abs"]
    assert got["sweep_axes"] == ["v", "didv"]                              # a sweep opens as it did
    assert got["units"] == ["m", "s", "m", "A"]
