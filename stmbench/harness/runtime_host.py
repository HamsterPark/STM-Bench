"""Host MAST's real ``CoreRuntime`` on top of a simulator instance (docs/DESIGN.md §5.3).

Why the full runtime and not a bare ``ExecutionContext``: MAST binds process-level holders
in ``CoreRuntime.setup()`` — tip facts (``tip_state``), scan-map scope, operating mode,
monitoring/alert store, watchdog, experiment storage. Skills read those holders; without
the runtime a composite like ``ForgeAuTip`` runs "blind" (no tip registry → treats a qPlus
as a wire tip → pokes at 50 mV → rings the fork up; no map → every spot looks clean).

The host also **syncs the world's facts into MAST** the way an operator would: registers
the tip (material / form / qPlus params) and fills the instrument profile keys that come
from the rig profile (Z range, au_step_pm, preamp). Everything MAST writes goes under
``MAST2_PROJECT_ROOT = <out>/mast_root``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import fields as _dc_fields
import time
from pathlib import Path
from typing import Any

from stmsim.modules import build_dispatcher
from stmsim.physics.world import World
from stmsim.wire.server import WireServer

log = logging.getLogger("stmbench.host")

DEFAULT_SETTINGS = {
    "current_monitor.cm_enabled": 1,
    "current_monitor.cm_use_osci2t": 1,
    "hardware_modules.osci_2t": 1,
    "tool_refine_enabled": 0,
}


class RuntimeHost:
    def __init__(self, world: World, out_dir: str | Path, *, settings: dict | None = None,
                 vision_backend: str | None = None):
        self.world = world
        self.out_dir = Path(out_dir)
        self.mast_root = self.out_dir / "mast_root"
        self.mast_root.mkdir(parents=True, exist_ok=True)
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.vision_backend = vision_backend
        self.dispatcher = build_dispatcher(world)
        self.dispatcher.log_calls = True
        self.server: WireServer | None = None
        self.app: Any = None
        self.t_setup_s = 0.0

    # ── lifecycle ──
    def start(self) -> "RuntimeHost":
        self.server = WireServer(self.dispatcher, ports=[0, 0, 0, 0]).start()
        os.environ["MAST2_PROJECT_ROOT"] = str(self.mast_root)
        os.environ["MAST2_USER_ROOT"] = str(self.mast_root)
        for role, port in zip(("MAIN", "MONITOR", "DATA", "EMERGENCY"), self.server.bound_ports):
            os.environ[f"MAST_NANONIS_PORT_{role}"] = str(port)
        if self.vision_backend:
            os.environ["MAST_VISION_BACKEND"] = self.vision_backend
        self._write_settings()
        from mast.config import MASTConfig
        from mast.core.runtime import CoreRuntime

        # The isolated MAST2_PROJECT_ROOT has no provider key files — export configured keys as
        # env vars (MAST reads env before file, per provider). Nothing is written to disk.
        self.facts_keys = _export_api_keys_from_repo()

        t0 = time.perf_counter()
        self.app = CoreRuntime(MASTConfig())
        self.app.setup()
        self.t_setup_s = time.perf_counter() - t0
        self.sync_facts()
        return self

    def _write_settings(self) -> None:
        """Seed MAST's persistent UI settings before setup() hydrates them."""
        import json
        cfg_dir = self.mast_root / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        path = cfg_dir / "ui_settings.json"
        try:
            cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:  # noqa: BLE001
            cur = {}
        cur.update(self.settings)
        path.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")

    def sync_facts(self) -> dict:
        """Register the world's tip in MAST and push rig facts into the instrument profile."""
        out: dict = {}
        storage = getattr(self.app, "_storage", None)
        tip = self.world.tip
        try:
            from mast.logging import tip_registry
            res = tip_registry.register_tip(
                storage, material=tip.material, fabrication="etched" if tip.form != "qplus" else "unknown",
                form=tip.form, name=f"sim-{self.world.seed}", installed_by="stmbench",
                qplus_q=tip.qplus_q if tip.form == "qplus" else None,
                # the spring constant is a registration fact, not a controller readback: without
                # it a force inversion can only guess the nominal 1800 N/m.
                qplus_k_n_per_m=tip.qplus_k_n_per_m if tip.form == "qplus" else None,
                qplus_f0_hz=tip.qplus_f0_hz if tip.form == "qplus" else None,
                note="registered by stmbench harness")
            out["tip"] = {k: res.get(k) for k in ("ok", "tip_id", "warnings")}
        except Exception as exc:  # noqa: BLE001
            out["tip"] = {"ok": False, "error": repr(exc)}
        try:
            from mast.core import tip_state
            out["is_qplus"] = tip_state.is_qplus()
        except Exception as exc:  # noqa: BLE001
            out["is_qplus"] = repr(exc)
        try:
            from mast.core import instrument_profile as ip
            rig = self.world.rig
            vals = {
                # what an operator fills in on the initialization page for this rig
                "au_step_pm": float(rig.get("sample_reference.au_step_pm_measured", 235.4)),
                "z_range_m": float(rig.z_range_m),
                "z_extend_sign": "+1" if rig.extend_sign > 0 else "-1",
                # MAST clamps ``setpoint_max_a`` to this fact ("this preamp is ±10 nA" on a
                # rig whose gain is fixed). This preamp's gain is switchable (1E6…1E11 V/A,
                # ``Current.GainSet`` moves the real range), so the fact an operator would
                # register is the WIDEST range it reaches — 10 µA at index 0 — and the
                # hazard the clamp guards against (a setpoint above the range saturates the
                # preamp and drives Z into the surface) is left to the physics and to the
                # operator switching the gain first, as MoveAtomTo does. Declaring the
                # imaging range (10 nA) here would clamp manipulation setpoints such as 57 nA
                # in every mode and prevent the intended atom movement.
                "preamp_full_scale_a": float(self.world.preamp.max_full_scale_a),
                "preamp_gain_v_per_a": float(rig.get("preamp.gain_v_per_a", 1e9)),
                "approach_setpoint_a": float(rig.get("feedback.approach_setpoint_a", 120e-12)),
                "approach_p_gain_m": float(rig.get("feedback.approach_p_m", 3e-12)),
                "approach_i_gain_m_per_s": float(rig.get("feedback.approach_i_m_per_s", 180e-9)),
                "xy_coarse_motion": "yes",     # enum ("yes"|"no"), not a bool
                "qplus_amplitude_signal_index": 16,
                "z_noise_floor_m": float(self.world.noise.z_floor_m),
            }
            # read-modify-write: set_profile() REPLACES the snapshot, so carry the rest along
            prof = ip.get_profile()
            prof.update(vals)
            stored = ip.set_profile(prof)
            out["instrument_profile"] = {k: stored.get(k) for k in vals}
        except Exception as exc:  # noqa: BLE001
            out["instrument_profile"] = repr(exc)
        # Piezo-tilt response matrix G: AutoTilt refuses to run without it (the axis
        # mapping / sign of Piezo_TiltSet is wiring-dependent on a real rig, so MAST makes
        # an operator run TiltCalibrate once). The harness declares the simulator's own
        # response — World.TILT_RESPONSE_G, derived from the convention documented on
        # World.set_piezo_tilt — the way that operator's one-off calibration would have.
        # tests/test_tilt.py runs the real TiltCalibrate against the sim and checks it
        # solves the same matrix, so this declaration cannot silently drift from the physics.
        try:
            import numpy as _np
            from mast.core import instrument_profile as ip
            g = [list(row) for row in self.world.TILT_RESPONSE_G]
            cond = float(_np.linalg.cond(_np.asarray(g, dtype=float)))
            stored_cal = ip.set_tilt_calibration(g, cond=cond)
            out["tilt_calibration"] = ({"ok": True, **stored_cal} if stored_cal
                                       else {"ok": False, "error": "set_tilt_calibration refused", "g": g, "cond": cond})
        except Exception as exc:  # noqa: BLE001
            out["tilt_calibration"] = {"ok": False, "error": repr(exc)}
        # coarse-drive four-lock: the piezo-stack voltage ceiling must be DECLARED (nothing
        # on the instrument reports it); the harness declares the rig profile's value
        try:
            from mast.core import coarse_drive
            rig = self.world.rig
            decl = coarse_drive.declare(float(rig.get("motor.max_amplitude_v", 250.0)),
                                        float(rig.get("motor.freq_hz", 300.0)),
                                        declared_by="stmbench", notes="from stmsim rig profile")
            out["coarse_drive"] = {k: decl.get(k) for k in ("max_amplitude_v", "expected_frequency_hz")}
        except Exception as exc:  # noqa: BLE001
            out["coarse_drive"] = {"ok": False, "error": repr(exc)}
        # dI/dV needs a stabilisation point and a sweep window. MAST's condition table ships
        # with the structure and no sample facts, so the benchmark supplies these values per
        # sample; without them spectroscopy skills refuse to start.
        cond = getattr(self.world, "sts_condition", None)
        if cond:
            try:
                from dataclasses import replace as _replace

                from mast.core import sts_workflow as sw
                base = sw.CONDITIONS.get(sw.DEFAULT_CONDITION)
                fields = {f.name for f in _dc_fields(base)}
                use = {k: v for k, v in cond.items() if k in fields}
                sw.CONDITIONS[sw.DEFAULT_CONDITION] = _replace(base, **use)
                out["sts_condition"] = {"ok": True, **use,
                                        "ignored": sorted(set(cond) - set(use))}
            except Exception as exc:  # noqa: BLE001
                out["sts_condition"] = {"ok": False, "error": repr(exc)}
        out["experiment"] = self.ensure_experiment()
        # The rig's vacuum gauge is a placeholder in MAST (reads 0 = "no data"); an operator
        # signs "pressure is safe" before coarse motion. The harness signs on the scenario's behalf.
        try:
            from mast.core import vacuum_interlock
            # reason must be one of vacuum_interlock.ATTESTATION_REASONS; the sim chamber is
            # pumped and its gauge is MAST's placeholder → "high vacuum, gauge unavailable"
            att = vacuum_interlock.attest(
                "high_vacuum_gauge_unavailable", signed_by="stmbench", ttl_s=24 * 3600.0,
                note=f"simulated chamber at {self.world.pressure_pa:.1e} Pa (rig profile {self.world.rig.name})")
            out["vacuum_attest"] = {"ok": True, "signed_by": getattr(att, "signed_by", "stmbench")}
        except Exception as exc:  # noqa: BLE001
            out["vacuum_attest"] = {"ok": False, "error": repr(exc)}
        self.facts = out
        return out

    def ensure_experiment(self, name: str = "stmbench-episode", sample: str | None = None) -> dict:
        """Open an experiment + sample so the sample_gate lets instrument skills run
        (an operator does this in the right-hand panel before touching the tip)."""
        el = getattr(self.app, "_experiment_log", None)
        if el is None:
            return {"ok": False, "error": "no experiment log"}
        try:
            exp_id = getattr(el, "current_experiment_id", None)
            if not exp_id:
                exp_id = el.start_experiment(name, "benchmark episode on stmsim")
            sample_name = sample or f"{self.world.surface.material.name} seed{self.world.seed}"
            sid = el.start_sample(sample_name)
            return {"ok": True, "experiment_id": exp_id, "sample_id": sid}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": repr(exc)}

    # ── access ──
    @property
    def pool(self):
        return self.app._pool

    @property
    def state(self):
        return self.app._state

    @property
    def registry(self):
        return self.app._registry

    def context(self, approval_source: str = "llm", run_id: str = "stmbench"):
        """An ExecutionContext wired the way the runtime wires its own.

        A bare ``ExecutionContext`` runs skills fine but lacks two attachments the
        runtime adds to every context it hands out: the ``marker_sink`` (the ONLY
        path by which a nested ``RelocateCoarseXY`` writes the ``coarse_move`` marker
        that advances ``coord_epoch``) and the non-latching tip-quality halt check.
        Omitting these attachments leaves the coordinate epoch unchanged after coarse
        relocations and can cause ``FindCleanSpot`` to reuse stale site markers. The stand-in
        therefore carries the same attachments as the runtime context.
        """
        from mast.core.execution_context import ExecutionContext
        from mast.core import runtime as _rt

        ctx = ExecutionContext(pool=self.pool, state=self.state, registry=self.registry,
                               approval_source=approval_source, run_id=run_id,
                               owner="stmbench harness")
        attach_halt = getattr(_rt, "_attach_halt_check", None)
        attach_sink = getattr(_rt, "_attach_marker_sink", None)
        if callable(attach_halt):
            attach_halt(ctx, self.app, run_id)
        if callable(attach_sink):
            attach_sink(ctx, self.app)
        if not callable(getattr(ctx, "marker_sink", None)):
            raise RuntimeError("ExecutionContext has no marker_sink — coord_epoch would never advance")
        return ctx

    def run_skill(self, name: str, params: dict | None = None):
        return self.context().run(name, params or {})

    def stop(self) -> None:
        try:
            if self.app is not None:
                for meth in ("shutdown", "stop", "close"):
                    fn = getattr(self.app, meth, None)
                    if callable(fn):
                        fn()
                        break
        except Exception:  # noqa: BLE001
            log.exception("runtime shutdown failed")
        if self.server is not None:
            self.server.stop()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def _export_api_keys_from_repo() -> dict:
    """Make the MAST repo's provider keys visible to a runtime whose project root is
    the episode's isolated directory.

    ``mast.config`` fixes the key-file map at import time from ``MAST2_PROJECT_ROOT``
    (which the host points at ``<out>/mast_root``), but reads env vars per provider
    first. Read each provider's key from the configured source (``MAST2_API_KEY_DIR`` if
    set, else the provider package's configured key directory) and export it under the
    provider's first env-var name — ``setdefault``, never overriding what the user set.
    Keys are never copied to disk; the returned dict only says which providers got one.
    """
    import importlib.util

    try:
        from mast import config as mcfg
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}
    key_dir = os.environ.get("MAST2_API_KEY_DIR", "").strip()
    if not key_dir:
        spec = importlib.util.find_spec("mast")
        origin = Path(getattr(spec, "origin", "") or "")
        # <repo>/MASTv2/mast/__init__.py → <repo>/api key
        key_dir = str(origin.resolve().parents[2] / "api key") if origin.name else ""
    out: dict = {"ok": True, "key_dir": key_dir, "exported": [], "missing": []}
    files = getattr(mcfg, "_API_KEY_FILES", {}) or {}
    env_map = getattr(mcfg, "_API_KEY_ENV", {}) or {}
    for provider, path in files.items():
        vars_ = tuple(env_map.get(provider, ()) or ())
        if not vars_:
            continue
        if any(os.environ.get(v, "").strip() for v in vars_):
            out["exported"].append(f"{provider}(env)")
            continue
        src = Path(key_dir) / Path(path).name if key_dir else None
        key = ""
        try:
            if src is not None and src.exists():
                for line in src.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key = line
                        break
        except Exception:  # noqa: BLE001
            key = ""
        if key:
            os.environ.setdefault(vars_[0], key)
            out["exported"].append(provider)
        else:
            out["missing"].append(provider)
    return out
