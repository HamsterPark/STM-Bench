"""What only the audience sees: the sample as it really is.

The world is rebuilt offline from the scenario and the seed — the same draw the episode
made — and checked against the ledger's ``truth_before`` before anything is drawn from it:
if the scenario file changed since the run, the rebuilt sample would be a different
sample, and the page says so instead of showing it.

Two pictures come out of it:

* the **perfect-tip frame** — every frame the model saved, recomputed over the same pixels
  (same geometry, the drift each row was taken under, the atoms where they were) with one
  sharp apex and no feedback lag or noise. What differs between it and the model's frame
  is the tip's doing (and the instrument's);
* the **overview** — the true sample around everything the model looked at, without
  adatoms (the page draws those, since they move) and with step edges outlined.

Surface damage made during the run (poke clusters, crash pits, scratches) is not in the
ledger with its geometry and is left out of both.
"""
from __future__ import annotations

import bisect
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from .timeline import Ledger

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "trackB" / "scenarios"
#: a perfect apex: the simulator's own floor for the terminating atom's smearing
IDEAL_APEX_SIGMA_M = 0.05e-9
#: sections of ``World.truth()`` that must agree before the rebuilt sample is trusted
CHECKED = ("hidden", "herringbone", "surface_state", "adatoms", "corral")


def _same(a, b, rel: float = 1e-6, abs_: float = 1e-9) -> bool:
    """Ledger value ``a`` vs rebuilt ``b``: numbers within tolerance, keys of ``a`` only (a
    newer simulator may report more)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and all(k in b and _same(v, b[k], rel, abs_) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y, rel, abs_) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_) or (math.isnan(a) and math.isnan(b))
    return a == b


class TruthView:
    def __init__(self, world, drift_samples: list[tuple[float, float, float]]):
        self.world = world
        self.surface = world.surface
        self._drift_t = [t for t, _, _ in drift_samples]
        self._drift_xy = [(x, y) for _, x, y in drift_samples]

    # ── building ──
    @classmethod
    def rebuild(cls, ledger: Ledger, frames: list[dict], scenario_dir: str | Path = SCENARIO_DIR
                ) -> tuple["TruthView | None", str]:
        """(view, "") — or (None, why not). The scenario file as it is now is tried first;
        if the sample it builds is not the one in the ledger, the file as it was at the
        commit the ledger records (``stmbench_sha``)."""
        ep = ledger.episode
        name = f"{ep.get('scenario_id')}.yaml"
        tb = ep.get("truth_before") or {}
        texts: list[tuple[str, str]] = []
        path = Path(scenario_dir) / name
        if path.is_file():
            texts.append(("now", path.read_text(encoding="utf-8")))
        old = _scenario_at_commit(ep.get("stmbench_sha"), name)
        if old is not None and all(old != t for _, t in texts):
            texts.append(("then", old))
        if not texts:
            return None, f"找不到场景文件 {name}"
        why = ""
        for _, text in texts:
            world, why = _build_and_check(text, int(ep.get("seed", 0)), tb)
            if world is not None:
                samples = sorted((float(f["sim0"]), float(f["drift_nm"][0]), float(f["drift_nm"][1]))
                                 for f in frames if f.get("drift_nm") is not None)
                return cls(world, samples), ""
        return None, why

    # ── drift ──
    def drift_nm(self, t: float) -> tuple[float, float]:
        """Sample offset (nm) at instrument time ``t``, interpolated between the frame starts
        the ledger recorded (and continued at the nearest segment's rate past either end)."""
        ts, xy = self._drift_t, self._drift_xy
        if not ts:
            return 0.0, 0.0
        if len(ts) == 1:
            return xy[0]
        k = min(max(bisect.bisect_right(ts, t) - 1, 0), len(ts) - 2)
        t0, t1 = ts[k], ts[k + 1]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (xy[k][0] + f * (xy[k + 1][0] - xy[k][0]), xy[k][1] + f * (xy[k + 1][1] - xy[k][1]))

    # ── atoms ──
    def set_atoms(self, state: dict) -> None:
        """Put the registry's atoms where ``state`` (id → {status, x, y} nm) says."""
        reg = self.surface.site.adatoms
        if reg is None:
            return
        reg.occ.clear()
        for aid, atom in reg.atoms.items():
            st = state.get(aid) or state.get(str(aid))
            if st is not None:
                atom.status = st.get("status", atom.status)
                if st.get("x") is not None and st.get("y") is not None:
                    atom.i, atom.j = reg.site_of(float(st["x"]) * 1e-9, float(st["y"]) * 1e-9, atom.sub)
            if atom.status == "on_surface":
                reg.occ[(atom.i, atom.j, atom.sub)] = aid
        reg.epoch += 1
        self.world.refresh_surface_state()

    # ── pictures ──
    def heights(self, X: np.ndarray, Y: np.ndarray, *, px_nm: float, bias_v: float | None,
                atomic: bool = True) -> np.ndarray:
        """Apparent height (m) a perfect apex would image at sample points ``X, Y`` (m)."""
        s = self.surface
        z = s.height_smooth(X, Y)
        a_nm = float(s.material.nn_m) * 1e9
        # the lattice only where a pixel grid resolves it cleanly (≥ 5 px per atom spacing);
        # coarser, it comes out as a moiré that reads as noise
        if atomic and px_nm <= a_nm / 5.0:
            period = float(s.material.first_order_period_m)
            z = z + s.atomic_height(X, Y) * math.exp(-2.0 * math.pi ** 2 * (IDEAL_APEX_SIGMA_M / period) ** 2)
        if bias_v and s.surface_state is not None:
            from stmsim.physics.junction import kappa_per_m
            s.prepare_frame(float(X.min()), float(X.max()), float(Y.min()), float(Y.max()), float(bias_v))
            z = z + s.electronic_height(X, Y, float(bias_v), kappa_per_m(s.material.phi_ev), IDEAL_APEX_SIGMA_M)
        return z

    def frame(self, fr: dict, geom: dict, atoms_at) -> np.ndarray:
        """The perfect-tip image of one saved frame, on its oriented pixel grid (row 0 = the
        high-v edge, like the model's frame). ``atoms_at(t)`` gives the atom state at ``t``;
        rows between two atom moves are drawn with the atoms where they were."""
        nx, ny = int(geom["nx"]), int(geom["ny"])
        w, h = float(geom["w_nm"]), float(geom["h_nm"])
        a = math.radians(float(geom.get("angle_deg") or 0.0))
        c, s = math.cos(a), math.sin(a)
        u = -w / 2 + (np.arange(nx) + 0.5) * w / nx
        v = h / 2 - (np.arange(ny) + 0.5) * h / ny
        U, V = np.meshgrid(u, v)
        X = float(geom["cx_nm"]) + c * U - s * V
        Y = float(geom["cy_nm"]) + s * U + c * V
        per_row = float(fr.get("per_row_s") or 0.0)
        up = str(fr.get("scan_dir") or "down") == "up"
        acq = (ny - 1 - np.arange(ny)) if up else np.arange(ny)
        t_row = float(fr["sim0"]) + acq * per_row
        for r in range(ny):
            dx, dy = self.drift_nm(float(t_row[r]))
            X[r] += dx
            Y[r] += dy
        out = np.empty((ny, nx))
        px_nm = w / max(nx, 1)
        bias = fr.get("bias_v")
        # rows grouped by the atom state they were taken under
        order = np.argsort(t_row)
        start = 0
        while start < ny:
            t0 = float(t_row[order[start]])
            key = atoms_at.key(t0)
            end = start + 1
            while end < ny and atoms_at.key(float(t_row[order[end]])) == key:
                end += 1
            rows = order[start:end]
            self.set_atoms(atoms_at(t0))
            out[rows] = self.heights(X[rows] * 1e-9, Y[rows] * 1e-9, px_nm=px_nm, bias_v=bias)
            start = end
        return out

    def overview(self, bbox_nm: tuple[float, float, float, float], px: int) -> tuple[np.ndarray, np.ndarray]:
        """(heights, step-edge mask) over ``(x0, y0, x1, y1)`` nm in the sample frame, row 0
        at the top. No adatoms and no atomic lattice: the page draws the atoms."""
        x0, y0, x1, y1 = bbox_nm
        xs = np.linspace(x0, x1, px) * 1e-9
        ys = np.linspace(y1, y0, px) * 1e-9
        X, Y = np.meshgrid(xs, ys)
        s = self.surface
        lv = s.terrace_level(X, Y)
        # terraces as faint shades (a real step would swamp a 10 pm reconstruction)
        z = s.herringbone_height(X, Y) + s.feature_height(X, Y) + 0.08 * float(s.material.step_m) * lv
        edge = np.zeros(lv.shape, dtype=bool)
        edge[:, 1:] |= lv[:, 1:] != lv[:, :-1]
        edge[1:, :] |= lv[1:, :] != lv[:-1, :]
        return z, edge


REPO = Path(__file__).resolve().parents[2]


def _scenario_at_commit(sha: str | None, name: str) -> str | None:
    """The scenario file at a commit, from git; None when there is no such commit or no git."""
    import subprocess

    if not sha or not all(c in "0123456789abcdef" for c in str(sha).lower()):
        return None
    try:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{sha}:stmbench/trackB/scenarios/{name}"],
                             capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.decode("utf-8") if out.returncode == 0 else None


def _build_and_check(text: str, seed: int, truth_before: dict):
    """(world, "") when the scenario text rebuilds the ledger's sample, else (None, why)."""
    from stmsim.scenario import Scenario

    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scenario.yaml"
            p.write_text(text, encoding="utf-8")
            sc = Scenario.load(p)
            world = sc.build_world(seed, session_dir=Path(td) / "session")
        truth = json.loads(json.dumps(world.truth(), default=float))
    except Exception as exc:  # noqa: BLE001 — the page still works without the truth view
        return None, f"重建样品失败：{type(exc).__name__}: {exc}"
    bad = [k for k in CHECKED if truth_before.get(k) not in (None, {}) and not _same(truth_before[k], truth.get(k))]
    if bad:
        return None, "场景文件在这次运行之后改过（" + "、".join(bad) + " 对不上），无法重建当时的样品"
    return world, ""


class AtomsAt:
    """Atom state (id → {status, x, y}) at any instant, from the timeline's atoms block."""

    def __init__(self, atoms: dict | None):
        atoms = atoms or {}
        self.initial = {a["id"]: {"status": a.get("status", "on_surface"), "x": a["x"], "y": a["y"]}
                        for a in atoms.get("initial") or []}
        self.moves = sorted(atoms.get("moves") or [], key=lambda m: m["sim"])
        self._times = [float(m["sim"]) for m in self.moves]

    def key(self, t: float) -> int:
        """How many moves have happened by ``t`` — equal keys, equal states."""
        return bisect.bisect_right(self._times, t)

    def __call__(self, t: float) -> dict:
        st = {k: dict(v) for k, v in self.initial.items()}
        for m in self.moves[:self.key(t)]:
            cur = st.setdefault(m["id"], {"status": "on_surface", "x": None, "y": None})
            cur["status"] = m.get("status", cur["status"])
            if "x" in m:
                cur["x"], cur["y"] = m["x"], m["y"]
        return st


def frame_extent_nm(fr: dict, geom: dict, view: TruthView | None) -> list[list[float]]:
    """The frame's four corners in the sample frame (nm), at the drift of its first row."""
    w, h = float(geom["w_nm"]), float(geom["h_nm"])
    a = math.radians(float(geom.get("angle_deg") or 0.0))
    c, s = math.cos(a), math.sin(a)
    dx, dy = view.drift_nm(float(fr["sim0"])) if view is not None else tuple(fr.get("drift_nm") or (0.0, 0.0))
    out = []
    for u, v in ((-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2), (-w / 2, -h / 2)):
        out.append([float(geom["cx_nm"]) + c * u - s * v + dx, float(geom["cy_nm"]) + s * u + c * v + dy])
    return out
