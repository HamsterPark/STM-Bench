"""The benchmark instrument over MCP (stdio): one episode per server process.

Any agent that speaks the Model Context Protocol can take the instrument's seat: the tools
are the instrument's actions (one tool each, with its JSON schema), its read-only
observations, and the result channel. The protocol layer is the small subset a tool server
needs - ``initialize``, ``tools/list``, ``tools/call``, ``ping`` - written against the
specification with the standard library only.

What the agent gets back: actions return the instrument's facts as JSON; ``scan`` and
``show_frame`` also return the frame as a PNG (:mod:`stmbench.instrument.render`, one fixed
rendering for every agent). ``finish`` ends the episode and writes the ledger; it does not
tell the agent its score. If the client goes away without finishing, the episode is finished
for it, so an unreported claim counts as not reproduced.

Run: ``python -m stmbench.cli episode --scenario P5_sader_jarvis_qplus --seed 0 --out <dir>``.
"""
from __future__ import annotations

import base64
import json
import sys
from typing import Any, Callable

import numpy as np

from .core import ACTIONS, API_VERSION, ActionError, Instrument
from .render import frame_png

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_NAME = "stm-bench-instrument"


def _text(obj: Any) -> dict:
    return {"type": "text", "text": obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)}


def _image(png: bytes) -> dict:
    return {"type": "image", "data": base64.b64encode(png).decode("ascii"), "mimeType": "image/png"}


def _block_means(a: np.ndarray, max_size: int) -> np.ndarray:
    """Down-sample by whole-block means to at most ``max_size`` per side."""
    k = max(1, int(np.ceil(max(a.shape) / max(1, max_size))))
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


class EpisodeServer:
    """MCP tool server around one :class:`Instrument` episode."""

    def __init__(self, instrument: Instrument):
        self.inst = instrument
        self.tools: dict[str, tuple[dict, Callable[[dict], list[dict]]]] = {}
        self._register()

    # ── tool table ──
    def _add(self, name: str, description: str, schema: dict, fn: Callable[[dict], list[dict]]) -> None:
        self.tools[name] = ({"name": name, "description": description, "inputSchema": schema}, fn)

    def _register(self) -> None:
        inst = self.inst
        none = {"type": "object", "properties": {}, "additionalProperties": False}
        self._add("get_task", "The task, its scoring rules, and the instrument manual. Read this first.", none,
                  lambda a: [_text(inst.task() + "\n\n" + inst.manual())])
        for spec in ACTIONS.values():
            self._add(spec.name, spec.doc, spec.schema(), self._action(spec.name))
        self._add("status", "Instrument readings now (no instrument time).", none, lambda a: [_text(inst.status())])
        self._add("info", "Fixed instrument facts and registered sensor constants.", none,
                  lambda a: [_text(inst.info())])
        self._add("list_data", "The frames and spectra saved so far.", none, lambda a: [_text(inst.list_data())])
        frame_schema = {"type": "object", "properties": {
            "frame": {"type": "string", "description": "frame file name"},
            "channel": {"type": "string", "enum": ["z", "current"], "default": "z"},
            "direction": {"type": "string", "enum": ["forward", "backward"], "default": "forward"}},
            "required": ["frame"], "additionalProperties": False}
        self._add("show_frame", "A saved frame as an image (row 0 = top, column 0 = left; line-flattened, "
                                "0.5–99.5 % contrast, enlarged by an integer factor) with its geometry.",
                  frame_schema, self._show_frame)
        values_schema = {"type": "object", "properties": {
            **frame_schema["properties"],
            "max_size": {"type": "integer", "minimum": 8, "maximum": 256, "default": 64,
                         "description": "at most this many values per side (block means)"}},
            "required": ["frame"], "additionalProperties": False}
        self._add("frame_values", "A saved frame's values as numbers (z in pm, current in pA), "
                                  "down-sampled by block means; row 0 = top.", values_schema, self._frame_values)
        self._add("get_spectrum", "A saved spectrum's data (V, A, dI/dV in A/V; or z in m, A, Δf in Hz, m).",
                  {"type": "object", "properties": {"spectrum": {"type": "string"}}, "required": ["spectrum"],
                   "additionalProperties": False}, self._spectrum)
        ids = list(inst.scenario.claim_ids)
        self._add("report_result", "Report one result of the task (see the scoring rules in get_task).", {
            "type": "object", "properties": {
                "claim_id": {"type": "string", "enum": ids},
                "value": {"type": "number"},
                "x_nm": {"type": "number", "description": "position results: x (scan coordinates, nm)"},
                "y_nm": {"type": "number", "description": "position results: y (scan coordinates, nm)"},
                "frame": {"type": "string", "description": "the frame a position was read in"},
                "evidence": {"type": "array", "items": {"type": "string"},
                             "description": "file names of the frames and spectra the result came from"},
                "unit": {"type": "string"}, "sigma": {"type": "number"}, "note": {"type": "string"}},
            "required": ["claim_id", "value"], "additionalProperties": False}, self._report)
        self._add("report_tip_state", "State your diagnosis of the tip.", {
            "type": "object", "properties": {"tip_state": {"type": "string"}, "junction": {"type": "string"},
                                             "note": {"type": "string"}},
            "required": ["tip_state"], "additionalProperties": False},
                  lambda a: [_text(inst.report_tip_state(**a))])
        self._add("finish", "End the episode: no further actions or reports.", none, self._finish)

    def _action(self, name: str) -> Callable[[dict], list[dict]]:
        def call(args: dict) -> list[dict]:
            out = self.inst.act(name, **args)
            blocks = [_text(out)]
            if name == "scan":
                f = self.inst.frame(out["frame"])
                png, k = frame_png(f["array"], "z")
                blocks = [_text({**out, "image": {"channel": "z", "direction": "forward", "display_scale": k}}),
                          _image(png)]
            return blocks
        return call

    def _show_frame(self, a: dict) -> list[dict]:
        f = self.inst.frame(a["frame"], a.get("channel", "z"), a.get("direction", "forward"))
        png, k = frame_png(f["array"], f["channel"])
        meta = {k2: v for k2, v in f.items() if k2 != "array"}
        meta["display_scale"] = k
        meta["pixel_to_nm"] = ("x_nm = center_x + (col + 0.5 - pixels/2) * nm_per_px, "
                               "y_nm = center_y - (row + 0.5 - pixels/2) * nm_per_px (rotate by angle_deg about "
                               "the centre); display pixels: divide by display_scale first")
        return [_text(meta), _image(png)]

    def _frame_values(self, a: dict) -> list[dict]:
        f = self.inst.frame(a["frame"], a.get("channel", "z"), a.get("direction", "forward"))
        scale, unit = (1e12, "pm") if f["channel"] == "z" else (1e12, "pA")
        v = _block_means(np.asarray(f["array"], float), int(a.get("max_size", 64))) * scale
        meta = {k2: x for k2, x in f.items() if k2 != "array"}
        meta.update(unit=unit, block_px=int(np.ceil(f["pixels"] / v.shape[0])), shape=list(v.shape),
                    values=np.round(v, 2).tolist())
        return [_text(meta)]

    def _spectrum(self, a: dict) -> list[dict]:
        s = self.inst.spectrum(a["spectrum"])
        out = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        return [_text(out)]

    def _report(self, a: dict) -> list[dict]:
        a = dict(a)
        return [_text(self.inst.report(a.pop("claim_id"), a.pop("value"), **a))]

    def _finish(self, a: dict) -> list[dict]:
        self.inst.finish()
        return [_text("The episode is over; the record is written.")]

    # ── JSON-RPC ──
    def handle(self, msg: dict) -> dict | None:
        """One JSON-RPC message in, its response out (``None`` for a notification)."""
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if mid is None:                                   # a notification: nothing to answer
            return None
        try:
            if method == "initialize":
                want = params.get("protocolVersion")
                version = want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
                result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": SERVER_NAME, "version": f"api-{API_VERSION}"},
                          "instructions": "Call get_task first. Actions change the instrument and spend "
                                          "instrument time; observations do not."}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [t for t, _ in self.tools.values()]}
            elif method == "tools/call":
                name = params.get("name")
                if name not in self.tools:
                    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}}
                try:
                    content = self.tools[name][1](dict(params.get("arguments") or {}))
                    result = {"content": content, "isError": False}
                except (ActionError, TypeError, ValueError, KeyError) as exc:
                    result = {"content": [_text(f"refused: {exc}")], "isError": True}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as exc:  # noqa: BLE001 - a server error must reach the client, not kill the episode
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve_stdio(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                             "error": {"code": -32700, "message": "parse error"}}) + "\n")
                    stdout.flush()
                    continue
                resp = self.handle(msg)
                if resp is not None:
                    stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                    stdout.flush()
        finally:
            if not self.inst.finished:
                self.inst.finish()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="stmbench-episode", description="serve one benchmark episode over MCP stdio")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="run directory for the ledger and saved files")
    a = ap.parse_args(argv)
    EpisodeServer(Instrument(a.scenario, a.seed, out_dir=a.out)).serve_stdio()
    return 0


if __name__ == "__main__":
    sys.exit(main())
