"""Run directory → replay: a folder the page opens straight from disk, or one HTML file.

    python -m stmbench.cli replay <run dir> [--out DIR] [--single-file] [--bare] [--no-truth] [--open]

Folder layout: ``index.html`` (the player), ``replay.js`` (``window.REPLAY = {...}``, so the
page works from ``file://`` without a server), ``frames/NNN.png`` (the model's frames),
``frames/NNN_ideal.png`` (the same pixels imaged by a perfect tip), ``overview.png``.
``--single-file`` inlines all of it (images as data URIs) into one ``.html``; ``--bare``
leaves out the html/head/body skeleton, for hosts that wrap pages in their own.

The replay reads the ledger and never writes into the run directory.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import math
import re
from pathlib import Path

import numpy as np

from ..human import render
from . import lay
from .timeline import Ledger, build_timeline
from .truth_view import SCENARIO_DIR, AtomsAt, TruthView, frame_extent_nm

STATIC = Path(__file__).resolve().parent / "static"
PLAYER = STATIC / "player.html"
DATA_TAG = '<script src="replay.js"></script>'
BODY_START = '<div class="wrap" id="app">'
HEAD = ('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
CMAP = "afmhot"
OVERVIEW_PX = 480
OVERVIEW_MIN_NM, OVERVIEW_MAX_NM = 12.0, 1500.0


def replay_name(ep: dict) -> str:
    raw = f"{ep.get('scenario_id')}_seed{ep.get('seed')}_{ep.get('mode')}_{ep.get('model_id') or 'script'}_{ep.get('run_id')}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw)


def mode_text(mode: str, model_id: str | None) -> tuple[str, str]:
    """(who acts, one line on how) — the page is honest about who sat at the instrument."""
    m = str(mode or "")
    if m == "C":
        return "脚本", "模式 C：写死的脚本基线，没有 AI 参与（只用来验证题目可解）。"
    if m == "H":
        if str(model_id or "human") == "human":
            return "操作员", "模式 H：一个人在网页上操作，和 AI 用的是同一套工具与预算；开发用，不上榜。"
        return "AI", f"模式 H：{model_id} 通过网页接口操作仪器，和正式测试用的是同一套工具与预算；开发用，不上榜。"
    return "AI", f"模式 {m}：{model_id} 独立操作仪器。"


def _scenario(ep: dict, scenario_dir: Path):
    from stmsim.scenario import Scenario

    p = scenario_dir / f"{ep.get('scenario_id')}.yaml"
    return Scenario.load(p) if p.is_file() else None


def _asset(assets: dict[str, bytes], name: str, data: bytes) -> str:
    assets[name] = data
    return name


def _shared_limits(z: np.ndarray, z_ai: np.ndarray, lo: float, hi: float) -> tuple[float, float] | None:
    """Colour limits for the perfect-tip frame. On the model's scale (background matched to
    background) when the two span about the same heights — then what still differs is the
    tip's or the instrument's contribution, not the colour map's. When the model's frame is
    severely distorted (a crashed tip's streaks span nanometres), its scale would wash the
    truth out: then the
    truth gets its own."""
    ok = np.isfinite(z_ai)
    if not ok.any() or not np.isfinite(z).any():
        return None
    a, b = np.percentile(z[np.isfinite(z)], [0.5, 99.5])
    if (b - a) < (hi - lo) / 3.0:
        return None
    shift = float(np.nanmedian(z_ai[ok])) - float(np.nanmedian(z))
    return lo - shift, hi - shift


def _overview_bbox(corners: list[list[float]], atoms: dict | None) -> tuple[float, float, float, float] | None:
    """Where the model worked: its frames, the atoms that moved, the target. Not every atom —
    a deposit covers far more of the sample than anyone looks at."""
    pts = [p for c in corners for p in c]
    if atoms:
        pts += [[m["x"], m["y"]] for m in atoms.get("moves") or [] if "x" in m]
        tgt = atoms.get("target") or {}
        pts += [p for p in (tgt.get("site"), tgt.get("start")) if p]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) * 1.12
    half = min(max(half, OVERVIEW_MIN_NM / 2), OVERVIEW_MAX_NM / 2)
    return cx - half, cy - half, cx + half, cy + half


def build_replay(run_dir: str | Path, out: str | Path | None = None, *, single_file: bool = False,
                 bare: bool = False, truth: bool = True, max_px: int = 400,
                 scenario_dir: str | Path = SCENARIO_DIR, log=print) -> Path:
    """Build the replay of one episode; returns the page to open."""
    ledger = Ledger.load(run_dir)
    ep = ledger.episode
    sc = _scenario(ep, Path(scenario_dir))
    specs = {str(c.get("id")): dict(c) for c in (sc.claims or [])} if sc is not None else {}
    tl = build_timeline(ledger, claim_specs=specs)
    if out is None:
        from ..paths import data_path
        out = data_path("replays", replay_name(ep))
    out_dir = Path(out)
    assets: dict[str, bytes] = {}

    view, why = (None, "构建时关掉了") if not truth else TruthView.rebuild(ledger, tl["frames"], scenario_dir)
    if why:
        log(f"truth view: {why}")
    atoms_at = AtomsAt(tl["atoms"])
    corners: list[list[list[float]]] = []
    for fr in tl["frames"]:
        path = ledger.session_dir / fr["file"] if fr.get("file") else None
        if path is None or not path.is_file():
            fr["png"] = None
            continue
        idx = int(fr.get("idx") or len(corners) + 1)
        z_ai, _ = render.frame_array(path, channel="Z", direction="forward", flatten_mode="plane")
        png, lo, hi = render.array_png(z_ai, cmap=CMAP, max_px=max_px)
        geom = {k: v for k, v in render.frame_meta(path).items()
                if k in ("cx_nm", "cy_nm", "w_nm", "h_nm", "angle_deg", "nx", "ny", "scan_dir")}
        if not geom.get("nx"):
            geom = dict(fr.get("geom") or {})
        fr["scan_dir"] = geom.pop("scan_dir", None) or fr.get("scan_dir") or "down"
        fr["geom"] = geom
        fr["png"] = _asset(assets, f"frames/{idx:03d}.png", png)
        fr["z_span_pm"] = (hi - lo) * 1e12
        fr["corners"] = frame_extent_nm(fr, geom, view)
        corners.append(fr["corners"])
        if view is not None:
            try:
                z = render.flatten(view.frame(fr, geom, atoms_at), "plane")
                ipng, _, _ = render.array_png(z, cmap=CMAP, max_px=max_px, limits=_shared_limits(z, z_ai, lo, hi))
                fr["ideal"] = _asset(assets, f"frames/{idx:03d}_ideal.png", ipng)
            except Exception as exc:  # noqa: BLE001 — one frame's truth must not sink the page
                log(f"truth view, frame {idx}: {type(exc).__name__}: {exc}")
        log(f"frame {idx}: {fr['file']}")
    overview = None
    if view is not None:
        bbox = _overview_bbox(corners, tl["atoms"])
        if bbox is not None:
            z, edge = view.overview(bbox, OVERVIEW_PX)
            # a muted map: the atoms and the model's frames are drawn over it in colour
            opng, _, _ = render.array_png(z, cmap="gray", min_span=60e-12, mask=edge, mask_rgb=(235, 235, 240))
            overview = {"png": _asset(assets, "overview.png", opng), "bbox": list(bbox)}
    fam = sc.family if sc is not None else ep.get("paper_id")
    actor, how = mode_text(ep.get("mode"), ep.get("model_id"))
    paper = dict(sc.paper) if sc is not None and isinstance(sc.paper, dict) else {}
    lay_paper = lay.paper_for(str(ep.get("scenario_id")), fam)
    page_title = ((lay_paper or {}).get("page") if actor == "AI" else None) \
        or ((lay_paper or {}).get("title") and f"{lay_paper['title']}回放") or "STM 实验回放"
    data = {
        "v": 1,
        "meta": {"scenario_id": ep.get("scenario_id"), "seed": ep.get("seed"), "mode": ep.get("mode"),
                 "model_id": ep.get("model_id"), "run_id": ep.get("run_id"), "success": bool(ep.get("success")),
                 "partial": ep.get("partial"), "family": fam, "material": sc.material if sc is not None else None,
                 "actor": actor, "how": how, "page_title": page_title,
                 "paper": {"title": paper.get("title"), "ref": paper.get("ref")},
                 "built": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        "lay": {"intro": lay.INTRO, "paper": lay_paper, "judging": lay.JUDGING},
        "task": (sc.task.strip() if sc is not None and sc.task else None),
        "timeline": tl,
        "truth": {"available": view is not None, "reason": why, "overview": overview},
    }
    return write_bundle(data, assets, out_dir, single_file=single_file or bare, bare=bare, name=replay_name(ep))


def page_html(data_script: str, title: str, *, bare: bool = False) -> str:
    """The player with its data script in place. Wrapped in a standards-mode skeleton (head
    = everything before the app's first element) unless ``bare``."""
    page = PLAYER.read_text(encoding="utf-8")
    if DATA_TAG not in page or BODY_START not in page:
        raise RuntimeError(f"{PLAYER} lacks {DATA_TAG!r} or {BODY_START!r}")
    page = re.sub(r"<title>.*?</title>", f"<title>{_html_text(title)}</title>", page, count=1, flags=re.S)
    page = page.replace(DATA_TAG, data_script)
    if bare:
        return page
    head, body = page.split(BODY_START, 1)
    return f"{HEAD}{head}</head>\n<body>\n{BODY_START}{body}</body>\n</html>\n"


def _html_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_bundle(data: dict, assets: dict[str, bytes], out_dir: Path, *, single_file: bool, name: str,
                 bare: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    title = (data.get("meta") or {}).get("page_title") or "STM 实验回放"
    if single_file:
        inline = {k: "data:image/png;base64," + base64.b64encode(v).decode("ascii") for k, v in assets.items()}
        target = out_dir / f"{name}.html"
        target.write_text(page_html(f"<script>{_js(_swap(data, inline))}</script>", title, bare=bare),
                          encoding="utf-8")
        return target
    for rel, blob in assets.items():
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    (out_dir / "replay.js").write_text(_js(data), encoding="utf-8")
    target = out_dir / "index.html"
    target.write_text(page_html(DATA_TAG, title), encoding="utf-8")
    return target


def _swap(obj, table: dict[str, str]):
    """Asset names → data URIs, anywhere in the data."""
    if isinstance(obj, dict):
        return {k: _swap(v, table) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap(v, table) for v in obj]
    if isinstance(obj, str) and obj in table:
        return table[obj]
    return obj


def _plain(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    return o


def _js(data: dict) -> str:
    txt = json.dumps(_plain(data), ensure_ascii=False, separators=(",", ":"))
    # inside <script>: no "</script>" and no HTML comment openers can come out of the data
    return "window.REPLAY = " + txt.replace("</", "<\\/").replace("<!--", "<\\!--") + ";"
