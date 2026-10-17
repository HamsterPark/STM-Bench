"""The console: the benchmark instrument in a web page, for a person at the instrument and for an
agent that works through a screen (screenshots, mouse and keyboard).

    python -m stmbench.cli console --scenario P4_atom_positioning_cu111 --seed 0 --out runs/console_p4
    python -m stmbench.cli console --out runs/console        # pick a paper scenario and a seed on the page

One episode per process. The page drives the same :class:`~stmbench.instrument.Instrument` an
MCP agent gets — the same actions, observations, reports and refusals, on the same virtual
clock — and the same ledger lands in the run directory. The page never receives the world's
truth: it shows the task with its rule card and the manual (what an agent is told), frames as
the PNG an MCP agent sees, and ``finish`` does not tell the score. Observations are recorded in
the ledger as an agent's are, so the page reads the instrument only when asked (after an
action, or with Refresh), never on a timer.

The server binds ``127.0.0.1`` unless told otherwise (``--host``). The hidden state stays in
this process, whoever the page is served to; whoever reaches the page can operate the instrument.

Endpoints (JSON unless noted)::

    GET  /                  the page
    GET  /api/session       started, finished, the claims to report; the scenarios to pick from
    POST /api/start         {scenario, seed} — only when the console was started without one
    GET  /api/task          the task with its rule card, and the manual
    GET  /api/actions       the action table: name, doc, JSON schema
    POST /api/act           {name, args} → the instrument's reply
    GET  /api/status        the instrument's readings                      (an observation)
    GET  /api/info          the instrument's fixed facts                   (an observation)
    GET  /api/data          the saved frames (geometry, channels) and spectra (an observation)
    GET  /api/frame.png     ?f=&ch=&dir= — PNG, as an MCP agent sees it    (an observation)
    GET  /api/frame_values  ?f=&ch=&dir=&max_size= — block means, as MCP frame_values (an observation)
    GET  /api/spectrum      ?s= — a spectrum's columns                     (an observation)
    GET  /api/pixel         ?f=&col=&row= — a frame pixel → scan coordinates (geometry only)
    POST /api/report        {claim_id, value, x_nm?, y_nm?, frame?, evidence?, unit?, sigma?, note?}
    POST /api/tip_state     {tip_state, junction?, note?}
    POST /api/finish        ends the episode; the record is written

A refused call answers 400 with ``{"error": ...}``, the instrument's own words.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .. import render
from ..instrument import ACTIONS, ActionError, Instrument
from ..instrument.core import SCENARIO_DIR
from ..instrument.mcp_server import EpisodeServer
from ..instrument.render import frame_png

PAGE = Path(__file__).resolve().parent / "static" / "index.html"
REPORT_FIELDS = ("x_nm", "y_nm", "frame", "evidence", "unit", "sigma", "note")


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def paper_scenarios() -> list[str]:
    """The scenarios the instrument judges: the paper families (B1–B9 are simulator fixtures)."""
    return sorted(p.stem for p in SCENARIO_DIR.glob("P*.yaml"))


class Console:
    """The episode behind the page; every instrument call holds the lock."""

    def __init__(self, out: str | Path, scenario: str | None = None, seed: int | None = None):
        self.out = Path(out)
        self.fixed = scenario is not None
        self.lock = threading.Lock()
        self.inst: Instrument | None = None
        self.server: EpisodeServer | None = None
        if self.fixed:
            self._open(scenario, 0 if seed is None else int(seed), self.out)

    # ── the episode ──
    def _open(self, scenario: str, seed: int, run_dir: Path) -> None:
        if (run_dir / "episode.json").exists():
            raise ActionError(f"{run_dir} already holds an episode; use a new directory")
        self.inst = Instrument(scenario, int(seed), out_dir=run_dir)
        self.server = EpisodeServer(self.inst)

    def start(self, scenario: str, seed: int) -> dict:
        with self.lock:
            if self.inst is not None:
                raise ActionError("an episode is already open in this console")
            if scenario not in paper_scenarios():
                raise ActionError(f"unknown scenario {scenario!r}")
            self._open(scenario, int(seed), self.out / f"{scenario}_seed{int(seed)}")
            return self._session()

    def _inst(self) -> Instrument:
        if self.inst is None:
            raise ActionError("no episode is open: pick a scenario first")
        return self.inst

    def session(self) -> dict:
        with self.lock:
            return self._session()

    def _session(self) -> dict:
        inst = self.inst
        out: dict = {"started": inst is not None, "finished": bool(inst and inst.finished), "fixed": self.fixed}
        if inst is None:
            out["scenarios"] = paper_scenarios()
            return out
        # what the rule card states about each result, nothing more
        out.update(scenario_id=inst.scenario.id, seed=inst.seed, budget_s=inst.max_sim_s,
                   claims=[{"id": str(c["id"]), "kind": c.get("kind"), "unit": c.get("unit") or "",
                            "position": c.get("position") == "required", "say": str(c.get("say") or "").strip()}
                           for c in inst.scenario.claims if c.get("kind") != "constraint"])
        return out

    # ── what the agent is told ──
    def task(self) -> dict:
        with self.lock:
            inst = self._inst()
            return {"task": inst.task(), "manual": inst.manual()}

    @staticmethod
    def actions() -> list[dict]:
        return [{"name": s.name, "doc": s.doc, "schema": s.schema()} for s in ACTIONS.values()]

    # ── the instrument ──
    def act(self, name: str, args: dict) -> dict:
        with self.lock:
            return self._inst().act(str(name), **dict(args or {}))

    def status(self) -> dict:
        with self.lock:
            return self._inst().status()

    def info(self) -> dict:
        with self.lock:
            return self._inst().info()

    def data(self) -> dict:
        with self.lock:
            inst = self._inst()
            out = inst.list_data()
            for f in out["frames"]:                   # the channels the scan reply listed
                rec = inst.frames[f["frame"]]
                f["channels"] = ["z", "current"] + (["didv"] if rec.didv_fwd is not None else [])
            return out

    def frame_png(self, name: str, channel: str, direction: str) -> bytes:
        with self.lock:
            f = self._inst().frame(name, channel, direction)
        png, _ = frame_png(f["array"], f["channel"])
        return png

    def frame_values(self, name: str, channel: str, direction: str, max_size: int) -> dict:
        with self.lock:
            self._inst()
            resp = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "frame_values",
                "arguments": {"frame": name, "channel": channel, "direction": direction, "max_size": int(max_size)}}})
        res = resp.get("result") or {}
        text = (res.get("content") or [{}])[0].get("text", "")
        if "error" in resp or res.get("isError"):
            raise ActionError(text.removeprefix("refused: ") or str(resp.get("error")))
        return json.loads(text)

    def spectrum(self, name: str) -> dict:
        with self.lock:
            return self._inst().spectrum(name)

    def pixel(self, name: str, col: float, row: float) -> dict:
        """Scan coordinates of a point in a frame: ``col``/``row`` are pixel indices, fractional
        allowed (0 = the first pixel's centre, so a point ``x`` pixels from the left edge is
        ``col = x - 0.5``), row 0 at the top; :func:`stmbench.render.pixel_to_scan_nm`."""
        with self.lock:
            f = self._inst().frames.get(Path(str(name)).name)
            if f is None:
                raise ActionError(f"no frame {name!r}")
            g = f.geometry
        geom = {"cx_nm": g["center_nm"][0], "cy_nm": g["center_nm"][1], "w_nm": g["size_nm"],
                "h_nm": g["size_nm"], "angle_deg": g["angle_deg"], "nx": g["pixels"], "ny": g["pixels"]}
        x, y = render.pixel_to_scan_nm(geom, float(col), float(row))
        return {"frame": f.name, "x_nm": round(x, 4), "y_nm": round(y, 4)}

    # ── results ──
    def report(self, body: dict) -> dict:
        b = dict(body or {})
        if b.get("value") in (None, ""):
            raise ActionError("report: value is required")
        ev = b.get("evidence")
        if isinstance(ev, str):
            b["evidence"] = [e.strip() for e in ev.replace("，", ",").split(",") if e.strip()]
        args = {}
        for k in REPORT_FIELDS:
            v = b.get(k)
            if v in (None, "", []):
                continue
            args[k] = float(v) if k in ("x_nm", "y_nm", "sigma") else v
        with self.lock:
            return {"recorded": self._inst().report(str(b.get("claim_id") or ""), float(b["value"]), **args)}

    def tip_state(self, body: dict) -> dict:
        b = dict(body or {})
        if not str(b.get("tip_state") or "").strip():
            raise ActionError("tip_state is required")
        with self.lock:
            return {"recorded": self._inst().report_tip_state(str(b["tip_state"]), b.get("junction") or None,
                                                              str(b.get("note") or ""))}

    def finish(self) -> dict:
        with self.lock:
            self._inst().finish()                     # the verdict stays in the ledger, not on the page
            return {"finished": True, "message": "The episode is over; the record is written."}


class _Handler(BaseHTTPRequestHandler):
    console: Console

    def log_message(self, *a):                     # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(_jsonable(obj), ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}
        if not isinstance(data, dict):
            raise ValueError("the request body must be a JSON object")
        return data

    def _route(self, method: str) -> None:
        c = self.console
        url = urllib.parse.urlparse(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        p = url.path
        try:
            if method == "GET" and p in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif method == "GET" and p == "/api/session":
                self._json(c.session())
            elif method == "POST" and p == "/api/start":
                b = self._body()
                self._json(c.start(str(b.get("scenario") or ""), int(b.get("seed") or 0)))
            elif method == "GET" and p == "/api/task":
                self._json(c.task())
            elif method == "GET" and p == "/api/actions":
                self._json(c.actions())
            elif method == "POST" and p == "/api/act":
                b = self._body()
                self._json(c.act(str(b.get("name") or ""), b.get("args") or {}))
            elif method == "GET" and p == "/api/status":
                self._json(c.status())
            elif method == "GET" and p == "/api/info":
                self._json(c.info())
            elif method == "GET" and p == "/api/data":
                self._json(c.data())
            elif method == "GET" and p == "/api/frame.png":
                self._send(200, c.frame_png(q.get("f", ""), q.get("ch", "z"), q.get("dir", "forward")), "image/png")
            elif method == "GET" and p == "/api/frame_values":
                self._json(c.frame_values(q.get("f", ""), q.get("ch", "z"), q.get("dir", "forward"),
                                          int(q.get("max_size", 64))))
            elif method == "GET" and p == "/api/spectrum":
                self._json(c.spectrum(q.get("s", "")))
            elif method == "GET" and p == "/api/pixel":
                self._json(c.pixel(q.get("f", ""), float(q.get("col", 0)), float(q.get("row", 0))))
            elif method == "POST" and p == "/api/report":
                self._json(c.report(self._body()))
            elif method == "POST" and p == "/api/tip_state":
                self._json(c.tip_state(self._body()))
            elif method == "POST" and p == "/api/finish":
                self._json(c.finish())
            else:
                self._json({"error": f"no such endpoint: {method} {p}"}, 404)
        except (ActionError, TypeError, ValueError, KeyError) as exc:     # refused, as the MCP server refuses
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 — the page shows it; the episode goes on
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")


def make_server(console: Console, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("ConsoleHandler", (_Handler,), {"console": console})
    return ThreadingHTTPServer((host, int(port)), handler)


def serve(out: str | Path, *, scenario: str | None = None, seed: int | None = None, host: str = "127.0.0.1",
          port: int = 8765, open_browser: bool = True) -> None:
    """Serve the console until Ctrl-C; an episode still open is then finished and judged as it stands."""
    console = Console(out, scenario, seed)
    srv = make_server(console, host, port)
    shown = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    url = f"http://{shown}:{srv.server_address[1]}/"
    print(f"STM console on {url} (run directory: {Path(out)}); Ctrl-C to stop", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if console.inst is not None and not console.inst.finished:
            console.inst.finish()
