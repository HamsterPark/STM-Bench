"""The local web GUI for mode H: one process, one browser tab, one episode at a time.

    python -m stmbench.cli gui [--port 8765] [--out <runs dir>] [--time-scale 20]

The server is deliberately plain (``http.server``, no framework): a page, a handful of
JSON endpoints, PNGs of the saved frames. The episode itself runs in a worker thread
through the ordinary ``run_episode`` with ``mode="H"`` and a :class:`HumanPort` as the
model, so the ledger it writes (``episode.json`` under ``<out>/<scenario>/seed<N>/H_human_<policy>/``)
has the same shape as any LLM episode and ``stmbench.report`` reads it unchanged.

Endpoints
---------
``GET  /``                      the page
``GET  /api/scenarios``         the scenario YAMLs (paper families first)
``POST /api/start``             ``{scenario, seed, time_scale?, policy?}`` → starts an episode
``GET  /api/state``             snapshot; ``?v=<version>&wait=<s>`` long-polls until it changes,
                                ``?since=<n>`` returns only the events after ``n``
``GET  /api/transcript``        the conversation exactly as the model would see it
``POST /api/action``            ``{kind: "tool", name, args, note?}`` | ``{kind: "text", text}``
``POST /api/cancel``            end the episode with ``[ABORT]`` at the next reply
``GET  /api/files``             the ``.sxm`` / ``.dat`` files of the running episode
``GET  /api/frame``             PNG of a frame: ``?f=<name>&ch=Z&dir=forward&flatten=plane&cmap=gray&max=800``
``GET  /api/frame_meta``        the display meta of that rendering (colour limits, unit, geometry)
``GET  /api/dat``               PNG plot of a spectrum: ``?f=<name>&x=<col>&y=<col>``
``GET  /api/dat_json``          its columns

Only files inside the running episode's session directory are served, by bare name.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from stmsim.scenario import Scenario

from ..harness.episode import _budget_caps, resolve_policy, run_episode
from . import render
from .port import Action, HumanPort, HumanSession, strip_markers

STATIC_DIR = Path(__file__).resolve().parent / "static"
SCENARIO_DIR = Path(__file__).resolve().parents[1] / "trackB" / "scenarios"
MODE = "H"
MODEL_ID = "human"
MAX_WAIT_S = 30.0


def scenario_summary(sc: Scenario) -> dict:
    """What the page shows before and during an episode. The task text and the claim ids
    are what a model gets; tolerances and the scenario's notes stay hidden, as they do
    from a model."""
    claims = [{"id": str(c.get("id")), "kind": c.get("kind"), "unit": c.get("unit"),
               "position": c.get("position") == "required"} for c in (sc.claims or [])]
    return {"id": sc.id, "family": sc.family, "variant": sc.variant, "material": sc.material,
            "rig": sc.rig, "paper": dict(sc.paper) if isinstance(sc.paper, dict) else None,
            "task": sc.task, "budget": dict(sc.budget or {}), "time_scale": sc.time_scale,
            "claims": claims, "faults": len(sc.faults or []), "path": sc.path}


class HumanApp:
    """Owns the scenario list, the current session and the episode thread."""

    def __init__(self, out: str | Path, *, scenario_dir: str | Path = SCENARIO_DIR,
                 time_scale: float | None = None, max_calls: int = 1000, model_id: str = MODEL_ID):
        self.out = str(out)
        self.scenario_dir = Path(scenario_dir)
        self.time_scale = time_scale
        self.max_calls = int(max_calls)
        # who sits in the model's seat: "human" by default; a label such as
        # "claude-opus-agent" when an agent drives the page's API, so the ledger says so
        self.model_id = str(model_id or MODEL_ID)
        self.session = HumanSession()
        self.seq = 0                      # bumps per episode so a client can tell a fresh session
        self.history: list[dict] = []
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cache: dict[tuple, tuple[bytes, dict]] = {}

    # ── scenarios ──
    def scenario_paths(self) -> list[Path]:
        paths = sorted(self.scenario_dir.glob("*.yaml"))
        return sorted(paths, key=lambda p: (0 if p.stem.startswith("P") else 1, p.stem))

    def scenarios(self) -> list[dict]:
        out = []
        for p in self.scenario_paths():
            try:
                out.append(scenario_summary(Scenario.load(p)))
            except Exception as exc:  # noqa: BLE001 — one broken YAML must not hide the rest
                out.append({"id": p.stem, "error": f"{type(exc).__name__}: {exc}", "path": str(p)})
        return out

    def _find(self, scenario_id: str) -> Path:
        for p in self.scenario_paths():
            if p.stem == scenario_id:
                return p
        raise KeyError(f"no scenario {scenario_id!r} under {self.scenario_dir}")

    # ── episodes ──
    def busy(self) -> bool:
        return self.session.phase in ("starting", "awaiting_action", "running")

    def start(self, scenario_id: str, seed: int, *, time_scale: float | None = None,
              policy: str | None = None) -> dict:
        with self._lock:
            if self.busy():
                raise RuntimeError("an episode is already running")
            path = self._find(scenario_id)
            sc = Scenario.load(path)
            ts = time_scale if time_scale is not None else self.time_scale
            pol = resolve_policy(sc.family, MODE, policy)
            max_sim_s, max_cmds = _budget_caps(sc.budget)
            session = HumanSession()
            session.begin(scenario_summary(sc), seed=int(seed), time_scale=(ts if ts is not None else sc.time_scale),
                          policy=pol, out=self.out, max_sim_s=max_sim_s, max_cmds=max_cmds)
            self.session = session
            self.seq += 1
            self._cache.clear()
            t = threading.Thread(target=self._run, args=(sc, int(seed), ts, pol, session),
                                 name=f"stmbench-H-{sc.id}-seed{seed}", daemon=True)
            self._thread = t
            t.start()
            return {"ok": True, "seq": self.seq, "scenario": sc.id, "seed": int(seed)}

    def _run(self, sc: Scenario, seed: int, time_scale: float | None, policy: str | None,
             session: HumanSession) -> None:
        port = HumanPort(session)
        try:
            res = run_episode(sc, seed=seed, mode=MODE, out=self.out, time_scale=time_scale,
                              model_id=self.model_id, policy=policy, max_model_calls=self.max_calls,
                              max_tool_calls=self.max_calls, model=port,
                              on_host=session.attach_host, on_event=session.push_event)
            session.finish(res.as_dict())
            self.history.append({"scenario": sc.id, "seed": seed, "success": res.success,
                                 "partial": res.partial, "run_dir": res.out_dir})
        except Exception as exc:  # noqa: BLE001 — the page shows it; the process stays up
            session.fail(f"{type(exc).__name__}: {exc}", traceback.format_exc())

    def cancel(self) -> dict:
        self.session.cancel()
        return {"ok": True}

    def shutdown(self, timeout_s: float = 10.0) -> None:
        if self.busy():
            self.session.cancel()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout_s)

    # ── files of the running episode ──
    def session_dir(self) -> Path | None:
        d = self.session.session_dir
        return Path(d) if d else None

    def file(self, name: str) -> Path:
        d = self.session_dir()
        if d is None:
            raise FileNotFoundError("no episode session directory yet")
        if not name or Path(name).name != name or name.startswith("."):
            raise PermissionError(f"bad file name {name!r}")
        p = d / name
        if not p.is_file():
            raise FileNotFoundError(name)
        return p

    def frame(self, name: str, **opts: Any) -> tuple[bytes, dict]:
        p = self.file(name)
        key = (str(p), p.stat().st_mtime_ns, tuple(sorted(opts.items())))
        hit = self._cache.get(key)
        if hit is None:
            hit = render.render_frame(p, **opts)
            if len(self._cache) > 200:
                self._cache.clear()
            self._cache[key] = hit
        return hit


# ── HTTP ────────────────────────────────────────────────────────────────────
class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, app: HumanApp):
        super().__init__(addr, handler)
        self.app = app


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    # quiet by default; the page is the log
    def log_message(self, fmt, *args):  # noqa: D401
        return

    @property
    def app(self) -> HumanApp:
        return self.server.app

    # ── helpers ──
    def _send(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def _error(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise ValueError(f"bad JSON body: {exc}") from exc
        return obj if isinstance(obj, dict) else {"value": obj}

    # ── routing ──
    def do_GET(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                body = (STATIC_DIR / "index.html").read_bytes()
                return self._send(200, body, "text/html; charset=utf-8")
            if u.path == "/api/scenarios":
                return self._json({"scenarios": self.app.scenarios(), "out": self.app.out,
                                   "time_scale": self.app.time_scale})
            if u.path == "/api/state":
                return self._state(q)
            if u.path == "/api/transcript":
                return self._json(self.app.session.transcript())
            if u.path == "/api/files":
                d = self.app.session_dir()
                return self._json({"files": render.list_files(d) if d else [], "session_dir": str(d) if d else None})
            if u.path == "/api/frame":
                png, _meta = self.app.frame(q.get("f", ""), **_frame_opts(q))
                return self._send(200, png, "image/png")
            if u.path == "/api/frame_meta":
                _png, meta = self.app.frame(q.get("f", ""), **_frame_opts(q))
                return self._json({**meta, **render.frame_meta(self.app.file(q.get("f", "")))})
            if u.path == "/api/dat":
                png = render.render_dat(self.app.file(q.get("f", "")), x=q.get("x"), y=q.get("y"))
                return self._send(200, png, "image/png")
            if u.path == "/api/dat_json":
                return self._json(render.dat_columns(self.app.file(q.get("f", ""))))
            if u.path == "/api/history":
                return self._json({"history": self.app.history})
            return self._error(404, f"no route {u.path}")
        except (FileNotFoundError, PermissionError, KeyError) as exc:
            return self._error(404, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/start":
                seed = int(body.get("seed", 0))
                ts = body.get("time_scale")
                ts = float(ts) if ts not in (None, "") else None
                pol = body.get("policy") or None
                try:
                    return self._json(self.app.start(str(body.get("scenario", "")), seed, time_scale=ts, policy=pol))
                except RuntimeError as exc:
                    return self._error(409, str(exc))
                except KeyError as exc:
                    return self._error(404, str(exc))
            if u.path == "/api/action":
                return self._action(body)
            if u.path == "/api/cancel":
                return self._json(self.app.cancel())
            return self._error(404, f"no route {u.path}")
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error(500, f"{type(exc).__name__}: {exc}")

    # ── endpoints ──
    def _state(self, q: dict) -> None:
        app = self.app
        since = int(q.get("since", 0) or 0)
        wait = float(q.get("wait", 0) or 0)
        v = q.get("v")
        seq = q.get("seq")
        session = app.session
        if wait > 0 and v is not None and int(v) == session.version and (seq is None or int(seq) == app.seq):
            session.wait_change(int(v), timeout_s=min(wait, MAX_WAIT_S))
        session = app.session            # a new episode may have replaced it meanwhile
        snap = session.snapshot(since=since if (seq is None or int(seq) == app.seq) else 0)
        snap["seq"] = app.seq
        snap["busy"] = app.busy()
        snap["history"] = app.history[-20:]
        self._json(snap)

    def _action(self, body: dict) -> None:
        kind = str(body.get("kind", "tool"))
        if kind == "tool":
            name = str(body.get("name", "") or "").strip()
            if not name:
                return self._error(400, "tool action needs a name")
            args = body.get("args")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return self._error(400, "args must be an object")
            action = Action(kind="tool", name=name, args=args, text=strip_markers(body.get("note", "")))
        elif kind == "text":
            text = str(body.get("text", "") or "")
            if not text.strip():
                return self._error(400, "text action needs text")
            action = Action(kind="text", text=text)
        else:
            return self._error(400, f"unknown action kind {kind!r}")
        try:
            self.app.session.submit(action)
        except RuntimeError as exc:
            return self._error(409, str(exc))
        return self._json({"ok": True, "version": self.app.session.version})


def _frame_opts(q: dict) -> dict:
    opts: dict = {"channel": q.get("ch", "Z") or "Z",
                  "direction": "backward" if q.get("dir", "forward") == "backward" else "forward",
                  "flatten_mode": q.get("flatten", "plane") or "plane",
                  "cmap": q.get("cmap", render.DEFAULT_CMAP) or render.DEFAULT_CMAP}
    mx = q.get("max")
    if mx:
        opts["max_px"] = max(16, min(4096, int(mx)))
    return opts


def serve(out: str | Path, *, host: str = "127.0.0.1", port: int = 8765, time_scale: float | None = None,
          max_calls: int = 1000, open_browser: bool = True, scenario_dir: str | Path = SCENARIO_DIR,
          model_id: str = MODEL_ID, ready=None) -> None:
    """Run the GUI until Ctrl+C. ``ready(url, app)`` is called once the socket is bound
    (tests use it; ``port=0`` picks a free port)."""
    app = HumanApp(out, scenario_dir=scenario_dir, time_scale=time_scale, max_calls=max_calls, model_id=model_id)
    httpd = _Server((host, int(port)), _Handler, app)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"STM-Bench mode H · {url}  (runs → {app.out}; seat: {app.model_id}; Ctrl+C to stop)", flush=True)
    if callable(ready):
        ready(url, app)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nstopping…", flush=True)
    finally:
        app.shutdown()
        httpd.server_close()


class BackgroundServer:
    """The same server on a thread (tests, notebooks): ``with BackgroundServer(out) as s: s.url``."""

    def __init__(self, out: str | Path, *, host: str = "127.0.0.1", port: int = 0, **kw: Any):
        self.app = HumanApp(out, **kw)
        self.httpd = _Server((host, int(port)), _Handler, self.app)
        self.url = f"http://{host}:{self.httpd.server_address[1]}/"
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.2},
                                        name="stmbench-H-http", daemon=True)

    def __enter__(self) -> "BackgroundServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.app.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5.0)
        time.sleep(0.05)
