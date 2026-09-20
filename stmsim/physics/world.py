"""World: the whole instrument in one object (surface + tip + junction + loop + scanner +
coarse stage + clocks + noise), with the controller-facing verbs as methods.

All controller modules call into this class under ``self.lock``. Every method that MAST
polls at kHz rate (``current_now``, ``z_now``) is O(1): equilibrium values are cached and
only re-derived when something changed; transients (poke, pulse, setpoint change) are
pre-computed time series looked up by wall time.

Z convention: ``z_tip = coarse_gap − extend_sign · Z_ctrl``. With
``extend_sign = −1``, a larger controller Z means a higher tip; withdrawn = Z at +range.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .clock import Clock
from .coarse import AutoApproach, CoarseStage
from .feedback import Loop, LoopParams
from .junction import Preamp, current_metal, gap_for_current, iv_curve, kappa_per_m
from .noise import NoiseModel
from .paths import default_session_dir
from .rig import RigProfile
from .scanner import SIG_CURRENT, SIG_Z, Frame, Renderer, ScanSettings
from .surface import Feature, Surface
from .tip import Tip

ZCTRL_OFF, ZCTRL_ON, ZCTRL_HOLD, ZCTRL_SWITCHING_OFF, ZCTRL_SAFETIP, ZCTRL_WITHDRAWING = 1, 2, 3, 4, 5, 6


@dataclass
class ZController:
    on: bool = False
    setpoint_a: float = 50e-12
    p_m: float = 3e-12
    i_m_per_s: float = 50e-9
    time_const_s: float = 6e-5
    z_n: float = 0.0                 # controller Z (m); when off this is the held value
    limit_high_m: float = 169.5e-9
    limit_low_m: float = -169.5e-9
    limits_enabled: bool = True
    switch_off_delay_s: float = 0.0
    tip_lift_m: float = 0.0
    status: int = ZCTRL_OFF
    controllers: list[str] = field(default_factory=lambda: ["Current", "log Current", "df"])
    active_index: int = 1


@dataclass
class Transient:
    """A pre-computed (t, z_tip, i) trajectory overriding the equilibrium for a window."""
    t0_wall: float
    t: np.ndarray                 # seconds from t0
    z_tip: np.ndarray
    i: np.ndarray
    fb_restore_at: float | None   # seconds from t0 when the loop takes over again
    label: str = ""

    def active(self, wall: float) -> bool:
        return self.t0_wall <= wall < self.t0_wall + float(self.t[-1])

    def sample(self, wall: float) -> tuple[float, float]:
        tt = wall - self.t0_wall
        return float(np.interp(tt, self.t, self.z_tip)), float(np.interp(tt, self.t, self.i))


class World:
    def __init__(self, *, rig: RigProfile | None = None, seed: int = 0,
                 material: str = "Au(111)", time_scale: float = 1.0,
                 session_dir: str | Path | None = None, tip: Tip | None = None,
                 contamination: float = 0.0, adsorbate_density_per_um2: float = 3.0,
                 herringbone=None, surface_state=None, terrace_median_m: float | None = None):
        self.lock = threading.RLock()
        self.rig = rig or RigProfile.load("reference-stm")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # drift and creep accrue per SIM second: they are functions of how long the experiment
        # runs, and the scenario budgets are written in sim hours (see Clock.__init__)
        self.clock = Clock(time_scale=time_scale, slow_scale=time_scale)
        self.surface = Surface(material, seed=seed, contamination=contamination,
                               adsorbate_density_per_um2=adsorbate_density_per_um2,
                               herringbone=herringbone, surface_state=surface_state,
                               terrace_median_m=terrace_median_m)
        self.tip = tip or Tip(material=self.rig.get("tip.material", "W"),
                              form=self.rig.get("tip.form", "qplus"),
                              rng=np.random.default_rng(seed + 1))
        if self.tip.apex_radius_ref_m < 0:      # apex smearing never sampled: draw it now
            self.tip._refresh_apex(self.tip.rng)
        self.preamp = Preamp(full_scale_a=self.rig.preamp_full_scale_a,
                             gain_index=Preamp.index_for(self.rig.preamp_full_scale_a),
                             desat_tau_s=self.rig.desat_tau_s)
        self.noise = NoiseModel(np.random.default_rng(seed + 2), floor_a=self.rig.noise_floor_a,
                                mains_amp_a=0.02e-12, pink_amp_a=0.02e-12,
                                spectral_lines={787.6: 0.01e-12},
                                z_floor_m=float(self.rig.get("z.noise_floor_m", 2e-12)))
        self.zctrl = ZController(limit_high_m=self.rig.z_range_m, limit_low_m=-self.rig.z_range_m,
                                 setpoint_a=float(self.rig.get("feedback.imaging_setpoint_a", 50e-12)),
                                 p_m=float(self.rig.get("feedback.approach_p_m", 3e-12)),
                                 i_m_per_s=float(self.rig.get("feedback.approach_i_m_per_s", 50e-9)) / 3.6)
        self.bias_v = 0.1
        self.bias_range_index = 0
        self.coarse = CoarseStage()
        self.coarse.freq_hz = float(self.rig.get("motor.freq_hz", 300))
        self.coarse.amp_v = float(self.rig.get("motor.drive_v", 180))
        self.approach = AutoApproach()
        self.approach.pulses_per_cycle = int(self.rig.get("motor.approach_pulses_per_cycle", 3))
        # Safe pulsed approach only works if one cycle's motor travel is smaller than the
        # fine-Z sweep, else the sample can jump from "out of reach" to "too close" in one cycle
        # (a crash). The rig's per-cycle travel is uncertain (107 µm / ~128 cycles estimate);
        # cap it at 80 % of the full fine range.
        per_cycle = float(self.rig.get("motor.z_step_m_lhe" if self.rig.temperature_k < 20 else "motor.z_step_m_rt",
                                       0.66e-6) or 0.66e-6)
        cap = 0.8 * 2.0 * self.rig.z_range_m
        self.coarse.z_step_m = min(per_cycle, cap) / max(self.approach.pulses_per_cycle, 1)
        self.scan = ScanSettings()
        self.frame: Frame | None = None
        self.frames_saved: list[str] = []
        self.session_dir = Path(session_dir) if session_dir else default_session_dir()
        self.tip_x = 0.0            # FolMe position (stage frame, m)
        self.tip_y = 0.0
        self._extend_sign = self.rig.extend_sign
        self.temperature_k = self.rig.temperature_k
        self.pressure_pa = float(self.rig.get("environment.pressure_pa", 1e-8))
        # drift / creep
        cold = self.temperature_k < 20
        # calibrated on 983 intervals (stmsim/calibrate/fit_creep.py, <data-root>/calib/creep.json):
        # lateral |v| ≈ 11 nm / (512 s + t) after a move → ~1 nm/min steady drift on the cold rig,
        # release step Δx 0.20 nm / Δz 26 pm vs 0.39 nm / 94 pm earlier
        dv = self.rig.get("drift", {}) or {}
        v_xy = float(dv.get("v_xy_m_per_s", 1.5e-11 if cold else 5e-11))
        v_z = float(dv.get("v_z_m_per_s", 1.0e-12 if cold else 8e-12))
        ang = float(self.rng.uniform(0, 2 * math.pi))
        self.drift_v_m_per_s = np.array([v_xy * math.cos(ang), v_xy * math.sin(ang), v_z]) * self.rng.normal(1.0, 0.25, 3)
        # Piezo drift compensation. The instrument offers it, MAST drives it
        # (SetDriftCompensation), and the papers need it: a spectroscopy line or a corral
        # repair runs for an hour, and at ~1 nm/min the sample walks several nanometres under
        # the tip while it does. See set_drift_comp for the sign convention.
        self.drift_comp_on = False
        self.drift_comp_v = np.zeros(3)
        self.drift_comp_t0 = 0.0                          # slow-clock instant it came on
        self._comp_frozen = np.zeros(3)                   # what it had already taken up
        self.drift_comp_sat = float(dv.get("comp_saturation", 0.9))
        self._creep: list[tuple[float, np.ndarray]] = []   # (t_sim, delta vector) after moves
        self.creep_gamma = float(dv.get("creep_gamma", 0.05))
        self.creep_tau_s = float(dv.get("creep_tau_s", 500.0))
        # fast-axis hysteresis, as the fraction of the axis by which the retrace is offset from
        # the trace on average: 0.0117 median on 40 80–120 nm Au frames (overlap-NCC peak
        # shift, <data-root>/calib/hysteresis.json direct_measurement; 6 px at 512 px), so
        # 3.1 px at 256 px. Shape (Renderer.hysteresis_profile_px): the offset is NOT constant
        # along the line — measured in thirds on the same 39 frames (2026-08-28); the left /
        # centre / right thirds read 0.98 / 1.37 / 1.17 % of the axis, i.e. a constant part
        # plus a mid-line bulge (a piezo loop whose branches only partly close at the
        # turnarounds): δ(u) = h·(e + (1−e)·1.5·4u(1−u)) with edge fraction e = 0.62 gives
        # thirds of 1.07 / 1.37 / 1.07 % for h = 1.17 % (the two real edge thirds pooled).
        # "shift" keeps a constant offset.
        self.hyst_frac = 0.012
        self.hyst_shape = "loop"
        self.hyst_edge_frac = 0.62
        # approach: crosstalk-spike scale (1.0 = high-noise condition; 0.3 = quiet) and the
        # per-cycle chance that a spike above the setpoint stops the module ("false landing")
        self.approach_noise = 0.3
        self.false_landing_p = 0.15
        # slow thermal creep of the sample toward the tip (fault: thermal_drift_to_limit), m/s of sim time
        self._coarse_creep_v = 0.0
        self._last_slow_tick_sim = 0.0
        # transients & events
        self.transients: list[Transient] = []
        self.events: list[dict] = []
        self.envelope_log: list[dict] = []
        self.withdrawn = True
        self.zctrl.z_n = self.zctrl.limit_high_m
        self._eq_cache: dict = {}
        self.lockin_on = False
        self.lockin_freq_hz = 973.0
        self.lockin_amp_v = 0.01
        self.renderer = Renderer(self)
        self.scan_series_counter = 0
        self.sts_last: dict | None = None
        self.piezo_tilt_deg: tuple[float, float] = (0.0, 0.0)   # Piezo.TiltSet state
        # ── paper-scenario bookkeeping (all inert unless a scenario switches them on) ──
        self.hidden: dict = {}                  # what resolve_hidden drew for this seed
        self.sts_counter = 0
        self.sts_records: list[dict] = []       # compact per-sweep record (arrays kept in memory)
        self.zspec_counter = 0
        self.zspec_records: list[dict] = []
        self.zspec_last: dict | None = None
        self.dats_saved: list[str] = []
        self._dat_counters: dict[str, int] = {}
        self.motion_log: list[dict] = []        # FolMe path segments (lateral manipulation)
        self.force = None                       # ForceParams (P5) or None
        self.pll = None                         # PLLState when the rig loads the PLL module
        self.z_sweep_m = float(self.rig.get("spectroscopy.z_sweep_m", 0.2e-9))
        if "PLL" in (self.rig.modules_loaded or ()):
            from .qplus import PLLState
            self.pll = PLLState.from_rig(self.rig, self.tip)
        self.df_rng = np.random.default_rng([seed, 0xDF])   # Δf noise: its own stream

    # ────────────────────────── derived physics ──────────────────────────
    def phi_junction(self) -> float:
        surf = self.surface.material.phi_ev * self.surface.phi_factor(*self.sample_xy())
        return 0.5 * (surf + self.tip.phi_ev)

    def kappa_m(self) -> float:
        return kappa_per_m(self.phi_junction())

    def loop_params(self) -> LoopParams:
        return LoopParams(p_m=self.zctrl.p_m, i_m_per_s=self.zctrl.i_m_per_s, kappa_m=self.kappa_m())

    def osc_factor(self) -> float:
        """⟨I⟩ / I(z_mean) when the qPlus sensor is oscillating — the feedback holds the
        *averaged* current, so an oscillating tip sits further out for the same setpoint."""
        if self.pll is None or not self.pll.output_on:
            return 1.0
        amp = self.pll.amplitude_now(self.clock.wall())
        if amp <= 0:
            return 1.0
        from .forces import averaged_current_factor
        return averaged_current_factor(self.kappa_m(), amp)

    def equilibrium_gap(self) -> float:
        return gap_for_current(self.zctrl.setpoint_a / self.osc_factor(), self.bias_v,
                               self.phi_junction())

    def hysteresis_px(self, st: ScanSettings) -> float:
        return self.hyst_frac * st.nx * (1.0 if st.w > 20e-9 else 0.5)

    def set_drift_comp(self, on: bool, vx: float, vy: float, vz: float,
                       sat: float | None = None) -> None:
        """``Piezo.DriftCompSet``: ramp the piezo to follow the drifting sample.

        **Sign.** ``v`` is the velocity at which features are seen to move in the scan frame —
        which is exactly what a drift measurement from two frames reports (MAST's
        ``ComputeDriftVector`` returns the cross-correlation shift of the current frame against
        a reference). The controller applies the measured vector and the motion stops. A reversed
        sign doubles the drift instead of cancelling it, which is why the
        documented workflow ends in "measure again": the mistake is visible and recoverable.

        Compensation freezes the offset where it stands; it does not undo the drift that has
        already happened. Switching it off leaves the piezo where the ramp had taken it.
        """
        now = self.clock.slow()
        self._comp_frozen = self._comp_accum(now)         # keep what the ramp already took up
        self.drift_comp_on = bool(on)
        self.drift_comp_v = np.array([float(vx), float(vy), float(vz)])
        self.drift_comp_t0 = now
        if sat is not None:
            self.drift_comp_sat = float(sat)

    def _comp_accum(self, t_slow: float) -> np.ndarray:
        """How far the compensation ramp has moved the piezo by ``t_slow`` (m)."""
        d = np.array(self._comp_frozen, dtype=float)
        if self.drift_comp_on:
            d = d + self.drift_comp_v * max(0.0, t_slow - self.drift_comp_t0)
        # the controller runs out of piezo eventually: the ramp saturates at a fraction of the
        # half range, and past that the sample walks away again
        rx, ry = self.rig.xy_range_m
        lim = float(self.drift_comp_sat) * min(rx, ry) / 2.0
        n = float(np.hypot(d[0], d[1]))
        if lim > 0 and n > lim:
            d = np.array([d[0] * lim / n, d[1] * lim / n, d[2]])
        return d

    def drift_offset(self, t_sim: float) -> tuple[float, float, float]:
        """Drift/creep offset (m) at sim instant ``t_sim``; evaluated on the slow clock."""
        t = self.clock.slow_from_sim(t_sim)
        d = self.drift_v_m_per_s * t + self._comp_accum(t)
        for t0, delta in self._creep:
            if t > t0:
                # creep continues in the direction of the move: x(t) = D·γ·ln(1 + t/τ)
                d = d + delta * self.creep_gamma * math.log1p((t - t0) / self.creep_tau_s)
        return float(d[0]), float(d[1]), float(d[2])

    def note_move(self, delta: np.ndarray) -> None:
        self._creep.append((self.clock.slow(), np.asarray(delta, float)))
        self._creep = self._creep[-20:]

    # Z conversions
    def z_tip_from_ctrl(self, z_n) -> float | np.ndarray:
        return self.coarse.coarse_gap_m - self._extend_sign * np.asarray(z_n, float)

    def z_ctrl_from_tip(self, z_tip) -> float | np.ndarray:
        return (self.coarse.coarse_gap_m - np.asarray(z_tip, float)) / self._extend_sign

    def sample_xy(self) -> tuple[float, float]:
        """Sample-frame coordinates under the tip now (tip position + accumulated drift/creep).
        Anything left on the surface must be stored in THESE coordinates, or drift walks the
        scan window away from it."""
        ox, oy, _ = self.drift_offset(self.clock.sim())
        return self.tip_x + ox, self.tip_y + oy

    def surface_height_here(self) -> float:
        ox, oy, oz = self.drift_offset(self.clock.sim())
        h = self.tip.effective_height(self.surface, np.array([self.tip_x + ox]),
                                      np.array([self.tip_y + oy]), self.kappa_m(),
                                      bias_v=self.bias_v)
        return float(h[0]) + oz

    def tip_height_m(self) -> float:
        return float(self.z_tip_from_ctrl(self.zctrl.z_n))

    def _equilibrium_z_tip(self) -> float:
        return self.surface_height_here() + self.equilibrium_gap()

    def _clamp_z_n(self, z_n: float) -> float:
        """Hardware range always; software limits only when enabled."""
        hw = self.rig.z_range_m
        z_n = min(max(z_n, -hw), hw)
        if self.zctrl.limits_enabled:
            z_n = min(max(z_n, self.zctrl.limit_low_m), self.zctrl.limit_high_m)
        return z_n

    def current_z_tip(self) -> float:
        """Tip height right now: mid-transient value if one is active, else the loop's."""
        tr = self._active_transient()
        if tr is not None:
            return tr.sample(self.clock.wall())[0]
        return self.achievable_z_tip()

    def _tick_slow(self) -> None:
        """Advance slow physics that is not a function of sim time alone (sample creep)."""
        now = self.clock.slow()
        dt = now - self._last_slow_tick_sim
        self._last_slow_tick_sim = now
        if self._coarse_creep_v and dt > 0:
            self.coarse.coarse_gap_m = self.coarse.coarse_gap_m - self._coarse_creep_v * dt

    def achievable_z_tip(self) -> float:
        """Where the tip actually is: the loop's target, pinned by the Z range if unreachable."""
        self._tick_slow()
        if self.zctrl.on and not self.withdrawn:
            z_n = self._clamp_z_n(float(self.z_ctrl_from_tip(self._equilibrium_z_tip())))
            self.zctrl.z_n = z_n
            return float(self.z_tip_from_ctrl(z_n))
        return self.tip_height_m()

    def _fine_range_ok(self, z_tip: float) -> bool:
        z_n = float(self.z_ctrl_from_tip(z_tip))
        return self.zctrl.limit_low_m * 0.98 < z_n < self.zctrl.limit_high_m * 0.98

    # ────────────────────────── live values (O(1)) ──────────────────────────
    def _active_transient(self) -> Transient | None:
        wall = self.clock.wall()
        for tr in self.transients:
            if tr.active(wall):
                return tr
        # drop finished ones
        self.transients = [tr for tr in self.transients if wall < tr.t0_wall + float(tr.t[-1])]
        return None

    def _approach_z_waveform(self) -> float | None:
        """Fine-Z during an auto-approach cycle: extend (+lim → −lim) then retract."""
        ap = self.approach
        if not ap.running:
            return None
        cyc = ap.cycle_time_s()
        phase = (self.clock.wall() - ap.t_started_wall) % cyc
        hi, lo = self.zctrl.limit_high_m, self.zctrl.limit_low_m
        if phase < ap.extend_time_s:
            return hi + (lo - hi) * (phase / ap.extend_time_s)
        if phase < ap.extend_time_s + ap.retract_time_s:
            return lo + (hi - lo) * ((phase - ap.extend_time_s) / ap.retract_time_s)
        return hi

    def z_now(self) -> float:
        with self.lock:
            self.auto_approach_poll()
            zw = self._approach_z_waveform()
            if zw is not None:
                self.zctrl.z_n = zw
                return zw + self.noise.z_sample()
            tr = self._active_transient()
            if tr is not None:
                z_tip, _ = tr.sample(self.clock.wall())
                return float(self.z_ctrl_from_tip(z_tip)) + self.noise.z_sample()
            if self.zctrl.on and not self.withdrawn:
                self.achievable_z_tip()
            z_n = self.zctrl.z_n
            if self.withdrawn or z_n in (self.zctrl.limit_high_m, self.zctrl.limit_low_m):
                return z_n                       # at the rail the reading is exactly the limit
            return z_n + self.noise.z_sample()

    def current_now(self) -> float:
        with self.lock:
            wall = self.clock.wall()
            self.auto_approach_poll()
            if self.approach.running:
                return float(self.preamp.clamp(self.noise.current_sample(0.0, wall)))
            tr = self._active_transient()
            if tr is not None:
                _, i = tr.sample(wall)
                return float(self.preamp.clamp(self.noise.current_sample(i, wall)))
            i_dc = self._dc_current()
            return float(self.preamp.clamp(self.noise.current_sample(i_dc, wall)))

    def _dc_current(self) -> float:
        if self.withdrawn:
            return 0.0
        if self.zctrl.on:
            z_tip = self.achievable_z_tip()
            gap = z_tip - self.surface_height_here()
            want = self.equilibrium_gap()
            if abs(gap - want) < 1e-12:
                return self.zctrl.setpoint_a * math.copysign(1.0, self.bias_v if self.bias_v else 1.0)
            # loop pinned at a Z limit: too far (no current) or too close (sample drifted in)
            if gap <= 0.02e-9:
                return float(self.preamp.clamp_a * math.copysign(1.0, self.bias_v or 1.0))
            return float(current_metal(gap, self.bias_v, self.phi_junction()))
        gap = self.tip_height_m() - self.surface_height_here()
        if gap <= 0.02e-9:
            return float(self.preamp.clamp_a * math.copysign(1.0, self.bias_v or 1.0))
        return float(current_metal(gap, self.bias_v, self.phi_junction()))

    # ────────────────────────── verbs ──────────────────────────
    def set_bias(self, v: float) -> None:
        with self.lock:
            self.bias_v = float(v)

    def set_setpoint(self, i: float) -> None:
        with self.lock:
            old = self.zctrl.setpoint_a
            self.zctrl.setpoint_a = float(i)
            if self.zctrl.on and not self.withdrawn and old != i:
                self._schedule_step_transient("setpoint")

    def zctrl_set(self, on: bool) -> None:
        with self.lock:
            if on and not self.zctrl.on:
                self.zctrl.on = True
                self.zctrl.status = ZCTRL_ON
                if self.withdrawn:
                    # controller: closing the loop from the withdraw rail extends Z at the I-gain
                    # speed until the setpoint is found (or the far rail if the sample is out
                    # of fine range) — this is how MAST's settle routine re-engages after a
                    # Withdraw (skills/composite/_z_settle.py: Withdraw → OnOffSet(1) → poll)
                    self.withdrawn = False
                    z0 = self.tip_height_m()
                    z_eq = self._equilibrium_z_tip()
                    if not self._fine_range_ok(z_eq):
                        z_eq = float(self.z_tip_from_ctrl(self._clamp_z_n(
                            float(self.z_ctrl_from_tip(z_eq)))))
                    travel = abs(z0 - z_eq)
                    speed = max(self.zctrl.i_m_per_s * 3.6, 10e-9)     # panel I-gain ≈ nm/s of travel
                    self._schedule_ramp_transient("fb_on_from_rail", z0, z_eq, travel / speed + 0.5)
                else:
                    self._schedule_step_transient("fb_on")
            elif not on and self.zctrl.on:
                # hold the current position
                self.zctrl.z_n = float(self.z_now())
                self.zctrl.on = False
                self.zctrl.status = ZCTRL_OFF

    def withdraw(self) -> None:
        with self.lock:
            self.zctrl.on = False
            self.zctrl.status = ZCTRL_OFF
            self.zctrl.z_n = self.zctrl.limit_high_m
            self.withdrawn = True
            self.transients.clear()
            if self.frame is not None and not self.frame.finished:
                self.frame.stopped = True
            self._event("withdraw")

    def home(self) -> None:
        with self.lock:
            self.zctrl.z_n = 0.0
            self.withdrawn = False

    def set_z_position(self, z_n: float) -> None:
        with self.lock:
            if self.zctrl.on:
                return
            self.zctrl.z_n = self._clamp_z_n(float(z_n))
            self.withdrawn = abs(self.zctrl.z_n - self.zctrl.limit_high_m) < 1e-12
            self._check_crash()

    def _check_crash(self) -> None:
        gap = self.tip_height_m() - self.surface_height_here()
        if gap < 0.0:
            self._crash("z_set_into_surface", severity=min(3.0, 1.0 + abs(gap) / 1e-9))

    def _crash(self, reason: str, severity: float = 1.0) -> None:
        out = self.tip.crash(self.clock.sim(), severity)
        sx, sy = self.sample_xy()
        self.surface.add_feature(Feature(sx, sy, -0.3e-9 * severity, 3e-9 * severity,
                                         kind="pit", born_sim_s=self.clock.sim()))
        self._lose_adatoms(sx, sy, 3e-9 * severity, cause="crash")
        self._event("crash", reason=reason, **out)

    def _lose_adatoms(self, x: float, y: float, radius: float, *, cause: str) -> None:
        """A crash / poke / pulse takes the adatoms it lands on with it."""
        reg = self.surface.site.adatoms
        if reg is None:
            return
        gone = reg.remove_within(x, y, radius)
        if reg.carried is not None and cause == "crash":
            reg.carried.status = "lost"
            gone.append(reg.carried.id)
            reg.carried = None
            self.tip.drop(self.clock.sim())
        if gone:
            self.refresh_surface_state()
            self._event("adatom_lost", atom_ids=gone, cause=cause,
                        xy_nm=[x * 1e9, y * 1e9], radius_nm=radius * 1e9)

    def _schedule_step_transient(self, label: str, duration_s: float = 0.6) -> None:
        """Feedback moves from the current z to the new equilibrium with loop dynamics."""
        z0 = self.tip_height_m()
        z_eq = self._equilibrium_z_tip()
        loop = Loop(self.loop_params())
        fs = 2000.0
        t = np.arange(0, duration_s, 1 / fs)
        z = loop.transient(np.full(t.size, z_eq), t, z0)
        h = self.surface_height_here()
        i = current_metal(np.clip(z - h, 0.02e-9, None), self.bias_v, self.phi_junction())
        self.transients.append(Transient(self.clock.wall(), t, z, self.preamp.clamp(i), None, label))

    def _schedule_ramp_transient(self, label: str, z0: float, z_eq: float, duration_s: float) -> None:
        """Linear Z ramp (integrator slew) from z0 to z_eq, then the loop holds."""
        fs = 500.0
        duration_s = max(float(duration_s), 0.2)
        t = np.arange(0, duration_s + 0.5, 1 / fs)
        ramp_end = duration_s
        z = np.where(t < ramp_end, z0 + (z_eq - z0) * (t / ramp_end), z_eq)
        h = self.surface_height_here()
        i = current_metal(np.clip(z - h, 0.02e-9, None), self.bias_v, self.phi_junction())
        self.transients.append(Transient(self.clock.wall(), t, z, self.preamp.clamp(i), None, label))

    def move_xy(self, x: float, y: float) -> None:
        with self.lock:
            delta = np.array([x - self.tip_x, y - self.tip_y, 0.0])
            self.tip_x, self.tip_y = float(x), float(y)
            if np.hypot(delta[0], delta[1]) > 5e-9:
                self.note_move(delta)
            self._eq_cache.clear()

    # ── lateral manipulation ──
    def junction_resistance_ohm(self) -> float:
        """The panel R = |V| / I is the control value used to select imaging or
        atom-manipulation conditions; it is not the true junction resistance."""
        return abs(self.bias_v) / max(self.zctrl.setpoint_a, 1e-15)

    def grip(self):
        """How the tip as it is now grabs adatoms (:func:`~stmsim.physics.adatoms.grip_of`)."""
        from .adatoms import grip_of
        return grip_of(self.tip, self.kappa_m())

    def site_blocked(self, x: float, y: float) -> bool:
        """A lattice site under a deposit — debris from a poke, a pulse, a crash, or a native
        adsorbate: an adatom cannot be dragged onto it."""
        for f in self.surface.features_near(x, y, 6e-9):
            r = 0.5 * max(f.sigma_x, f.sigma_y or f.sigma_x)
            if (f.x - x) ** 2 + (f.y - y) ** 2 <= r * r:
                return True
        return False

    #: a drag is only integrated below this multiple of the nominal pull threshold (atoms
    #: differ by a lognormal factor; beyond it no atom follows) — the rest is a direct move
    DRAG_CEILING = 5.0

    def move_xy_path(self, x: float, y: float, speed_m_s: float = 293e-9) -> list[dict]:
        """A FolMe move, integrated along its path so the tip can carry an atom with it.

        Far above the pull threshold this uses a direct displacement with a distance record; that
        early exit is what keeps ordinary navigation free. Below it the move also doses the
        tip with the manipulation current, which can change tip state."""
        with self.lock:
            x0, y0 = self.tip_x, self.tip_y
            reg = self.surface.site.adatoms
            r_ohm = self.junction_resistance_ohm()
            events: list[dict] = []
            active = (reg is not None and reg.n_on_surface and self.zctrl.on and not self.withdrawn
                      and r_ohm < self.DRAG_CEILING * reg.params.r_threshold_ohm)
            if active:
                ox, oy, _ = self.drift_offset(self.clock.sim())
                events = reg.drag((x0 + ox, y0 + oy), (x + ox, y + oy), r_ohm=r_ohm,
                                  v_m_s=speed_m_s, sim_s=self.clock.sim(),
                                  terrace=self.surface.terrace_level, grip=self.grip(),
                                  blocked=self.site_blocked)
                dt = math.hypot(x - x0, y - y0) / max(speed_m_s, 1e-12)
                self.tip.maybe_change(dt, self.clock.sim(), current_a=self.zctrl.setpoint_a,
                                      bias_v=self.bias_v)
            self.move_xy(x, y)
            if events:
                self.refresh_surface_state()
                n_hops = sum(int(e.get("n_hops", 0)) for e in events)
                for k in range(n_hops):
                    t = self.clock.wall() + (k + 1) * 0.02
                    self.noise.add_spike(t, 0.5 * self.zctrl.setpoint_a, 0.02)
                for ev in events:
                    detail = {k: v for k, v in ev.items() if k != "kind"}
                    if ev["kind"] == "adatom_picked":
                        self.tip.pick_up(self.clock.sim(), reg.params.species)
                    self._event(ev["kind"], **detail)
            if active:
                self.motion_log.append({"sim_s": self.clock.sim(),
                                        "from_nm": [x0 * 1e9, y0 * 1e9],
                                        "to_nm": [x * 1e9, y * 1e9],
                                        "speed_nm_s": speed_m_s * 1e9,
                                        "r_kohm": r_ohm / 1e3, "n_events": len(events)})
                if len(self.motion_log) > 2000:
                    del self.motion_log[:-2000]
            return events

    def registry_scan_row(self, st, row: int, ox: float, oy: float, t_sim: float,
                          speed_m_s: float) -> None:
        """Per-row hook from the renderer: imaging at a low junction resistance moves atoms."""
        reg = self.surface.site.adatoms
        if reg is None or not reg.n_on_surface or not self.zctrl.on or self.withdrawn:
            return
        xs, ys = st.pixel_xy(np.array([0.0, float(st.nx - 1)]), row)
        events = reg.scan_row((float(xs[0]) + ox, float(ys[0]) + oy),
                              (float(xs[1]) + ox, float(ys[1]) + oy),
                              r_ohm=self.junction_resistance_ohm(), v_m_s=speed_m_s, sim_s=t_sim,
                              grip=self.grip(), blocked=self.site_blocked,
                              terrace=self.surface.terrace_level)
        if not events:
            return
        self.refresh_surface_state()
        for ev in events:
            detail = {k: v for k, v in ev.items() if k != "kind"}
            if ev["kind"] == "adatom_picked":
                self.tip.pick_up(t_sim, reg.params.species)
            # the frame is rendered at scan start, so ``sim_s`` stamps the start; the row the
            # tip was on when the atom moved is when a replay shows it move
            self._event(ev["kind"], **detail, row_sim_s=float(t_sim))

    def drop_carried(self, cause: str = "approach") -> bool:
        """Put a carried atom back on the surface under the tip."""
        with self.lock:
            reg = self.surface.site.adatoms
            if reg is None or reg.carried is None:
                return False
            sx, sy = self.sample_xy()
            atom = reg.drop(sx, sy, sim_s=self.clock.sim(), cause=cause)
            if atom is None:
                return False
            self.tip.drop(self.clock.sim())
            self.refresh_surface_state()
            ax, ay = reg.site_xy(atom.i, atom.j, atom.sub)
            self._event("adatom_dropped", atom_id=atom.id, cause=cause,
                        site=[atom.i, atom.j], xy_nm=[ax * 1e9, ay * 1e9])
            return True

    def nearest_adatom(self, x: float, y: float):
        reg = self.surface.site.adatoms
        if reg is None:
            return None, float("inf")
        return reg.nearest(x, y)

    # ── qPlus force spectroscopy ──
    def force_law_here(self):
        """The force law over the current lateral position: the bond fades as a Gaussian of
        width ``sigma_lat`` away from the atom, so being 0.1 nm off costs 20 % of the well."""
        from .forces import ForceLaw
        if self.force is None:
            return None
        sx, sy = self.sample_xy()
        atom, d = self.nearest_adatom(sx, sy)
        w_lat = 0.0
        if atom is not None and math.isfinite(d):
            w_lat = math.exp(-0.5 * (d / self.force.sigma_lat_m) ** 2)
        return ForceLaw(self.force, radius_m=self.tip.radius_m, w_lat=w_lat)

    def df_physical(self, gap_mean_m: float, amplitude_m: float) -> float:
        """Δf (Hz) from the force law at a given mean tip height."""
        from .forces import delta_f
        law = self.force_law_here()
        if law is None:
            return 0.0
        z_close = max(gap_mean_m - amplitude_m, 0.02e-9)
        out = delta_f(law.F, np.array([z_close]), amplitude_m,
                      float(self.tip.qplus_f0_hz), float(self.tip.qplus_k_n_per_m))
        return float(np.asarray(out).ravel()[0])

    def df_now(self) -> float:
        """What ``PLL.FreqShiftGet`` and signal 17 report right now."""
        with self.lock:
            if self.pll is None:
                return float(self.df_rng.normal(0, 0.05))
            sigma = float(self.force.sigma_df_hz) if self.force is not None else 0.05
            if not self.pll.output_on or self.withdrawn:
                return float(self.df_rng.normal(0, 0.05))
            amp = self.pll.amplitude_now(self.clock.wall())
            gap = self.current_z_tip() - self.surface_height_here()
            phys = self.df_physical(gap, amp)
            return float(self.tip.qplus_f0_hz + phys - self.pll.center_freq_hz
                         + self.df_rng.normal(0, sigma))

    def amplitude_now(self) -> float:
        if self.pll is None:
            return float(50e-12 + self.df_rng.normal(0, 8e-12))
        if not self.pll.output_on:
            return float(self.pll.undriven_floor_m + abs(self.df_rng.normal(0, 1e-12)))
        gap = self.current_z_tip() - self.surface_height_here()
        amp = self.pll.amplitude_now(self.clock.wall(), gap_close_m=gap - self.pll.amp_setpoint_m)
        return float(amp + self.df_rng.normal(0, 1e-12))

    def zspec_curve(self, n: int, sweep_m: float, z_offset_m: float = 0.0, *, bwd: bool = True,
                    sweeps: int = 1, settle_s: float = 0.005, integ_s: float = 0.005,
                    retract: tuple | None = None) -> dict:
        """A Z sweep toward the surface: Δf(z), the averaged current, and the amplitude.

        ``z_rel`` follows the controller convention used by the simulator: it starts at 0 and
        goes **negative** as the tip approaches, and the header records the sweep end as
        ``−sweep``."""
        from .forces import averaged_current_factor
        with self.lock:
            n = max(int(n), 2)
            es = self._extend_sign
            h = self.surface_height_here()
            z_tip0 = self.tip_height_m() if not self.zctrl.on else h + self.equilibrium_gap()
            z_rel = np.linspace(0.0, -abs(sweep_m), n)
            z_tip = z_tip0 - es * z_offset_m - es * z_rel
            gap_mean = z_tip - h
            amp_set = self.pll.amp_setpoint_m if self.pll is not None else 0.0
            amp_free = self.pll.amplitude_now(self.clock.wall()) if self.pll is not None else 0.0
            z_close = gap_mean - amp_free
            contact = z_close < 0.02e-9
            law = self.force_law_here()
            if law is not None and amp_free > 0:
                from .forces import delta_f
                df = delta_f(law.F, np.clip(z_close, 0.02e-9, None), amp_free,
                             float(self.tip.qplus_f0_hz), float(self.tip.qplus_k_n_per_m))
            else:
                df = np.zeros(n)
            factor = averaged_current_factor(self.kappa_m(), amp_free) if amp_free > 0 else 1.0
            i = current_metal(np.clip(gap_mean, 0.02e-9, None), self.bias_v, self.phi_junction())
            i = i * factor
            t = np.arange(n) * (settle_s + integ_s)
            i = self.preamp.clamp(self.noise.current_series(i, self.clock.wall() + t))
            sigma = float(self.force.sigma_df_hz) if self.force is not None else 0.1
            df = np.asarray(df, float) + self.df_rng.normal(0, sigma, n)
            jump = False
            if self.tip.lambda_per_s > 2e-3 and self.df_rng.random() < 0.5:
                k = int(self.df_rng.integers(n // 4, 3 * n // 4))
                df[k:] += float(self.df_rng.uniform(0.2, 1.0)) * (1 if self.df_rng.random() < 0.5 else -1)
                jump = True
            amp = np.full(n, amp_free)
            if amp_free > 0:
                amp = np.where(contact, amp_free * 0.05, amp_free)
            retract_abort = False
            if retract and int(retract[0]):
                thr = float(retract[1])
                over = np.flatnonzero(np.abs(i) > thr)
                if over.size:
                    cut = int(over[0])
                    df[cut:] = np.nan
                    i[cut:] = np.nan
                    amp[cut:] = np.nan
                    contact = contact & (np.arange(n) < cut)
                    retract_abort = True
            if bool(np.any(contact)) and not retract_abort:
                depth = float(-np.min(z_close))
                outcome = self.tip.poke(self.clock.sim(), max(depth, 0.0), self.bias_v)
                sx0, sy0 = self.sample_xy()
                self._apply_poke_outcome(outcome, sx0, sy0)
                df = np.where(contact, np.maximum(50.0, np.abs(df) + 50.0), df)
                if self.zctrl.on:
                    self._schedule_step_transient("zspec_contact")
            sx, sy = self.sample_xy()
            near_kind, near_m = self.surface.nearest_scatterer(sx, sy)
            i_min = int(np.nanargmin(df)) if np.any(np.isfinite(df)) else 0
            span_beyond = abs(float(z_rel[-1] - z_rel[i_min])) * 1e12
            self.zspec_counter += 1
            rec = {"idx": self.zspec_counter, "n": n, "sweep_m": abs(sweep_m),
                   "z_offset_m": z_offset_m, "z0_n_m": self.zctrl.z_n,
                   "x_m": self.tip_x, "y_m": self.tip_y, "sx_m": sx, "sy_m": sy,
                   "near_kind": near_kind, "near_nm": near_m * 1e9,
                   "bwd": bool(bwd), "sweeps": int(sweeps),
                   "amplitude_m": amp_free, "f0_hz": float(self.tip.qplus_f0_hz),
                   "k_n_per_m": float(self.tip.qplus_k_n_per_m),
                   "center_freq_hz": float(self.pll.center_freq_hz) if self.pll else 0.0,
                   "bias_v": self.bias_v, "setpoint_a": self.zctrl.setpoint_a,
                   "df_min_hz": float(np.nanmin(df)) if np.any(np.isfinite(df)) else 0.0,
                   "z_rel_at_df_min_m": float(z_rel[i_min]),
                   "df_span_beyond_min_pm": span_beyond,
                   "i_max_a": float(np.nanmax(np.abs(i))) if np.any(np.isfinite(i)) else 0.0,
                   "contact": bool(np.any(contact)), "retract_abort": retract_abort,
                   "jump": jump, "path": None}
            self.zspec_last = {**rec, "z_rel": z_rel, "df": df, "i": i, "amp": amp,
                               "z_n": self.z_ctrl_from_tip(z_tip)}
            self.zspec_records.append(dict(rec))
            self._event("zspec", **rec)
            return self.zspec_last

    def _apply_poke_outcome(self, outcome: dict, sx: float, sy: float) -> None:
        """Surface memory of a poke: clusters/pits per apex, spatter for a qPlus ring-up."""
        kind = outcome.get("outcome")
        sim_s = self.clock.sim()
        if kind in ("cluster", "pit"):
            axis = float(outcome.get("cluster_axis_ratio", 1.0) or 1.0)
            hgt = float(outcome.get("cluster_height_m", 0.2e-9))
            sig = float(outcome.get("cluster_sigma_m", 1.5e-9))
            for a in self.tip.apexes:
                self.surface.add_feature(Feature(sx + a.dx, sy + a.dy,
                                                 hgt * a.w * (1 if kind == "cluster" else -1),
                                                 sig, sig * axis,
                                                 float(self.rng.uniform(0, math.pi)),
                                                 kind=kind, born_sim_s=sim_s))
            self._lose_adatoms(sx, sy, 2 * sig, cause="poke")
        elif kind == "ring_up":
            for _ in range(int(outcome.get("n_clusters", 8))):
                self.surface.add_feature(Feature(sx + float(self.rng.normal(0, 15e-9)),
                                                 sy + float(self.rng.normal(0, 15e-9)),
                                                 0.15e-9, 1.5e-9, kind="spatter", born_sim_s=sim_s))
            self._lose_adatoms(sx, sy, 15e-9, cause="poke")

    # ── piezo tilt correction ──
    #: Response of the *reported* Z slope (per axis, deg per deg) to a Piezo.TiltSet step is
    #: −1 on both axes under the convention in :meth:`set_piezo_tilt`, so the compensating
    #: matrix MAST's TiltCalibrate solves (G = −M⁻¹) is the identity: Δtilt = +measured slope.
    #: Declared by the harness as this rig's one-off calibration (RuntimeHost.sync_facts).
    TILT_RESPONSE_G: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))

    def tilt_plane_z(self, x, y):
        """The correction plane controller adds to the Z output (controller-Z metres, stage frame)."""
        tx, ty = self.piezo_tilt_deg
        return math.tan(math.radians(tx)) * np.asarray(x, float) + math.tan(math.radians(ty)) * np.asarray(y, float)

    def set_piezo_tilt(self, tilt_x_deg: float, tilt_y_deg: float) -> None:
        """Piezo.TiltSet: rotate the scan plane by the two tilt angles.

        Convention (controller Piezo Configuration → Tilt): the controller adds the plane
        ``tan(tilt_x)·x + tan(tilt_y)·y`` to the Z *output*, and the Z it reports (the
        Z-controller signal, what every frame and ``ZCtrl.ZPosGet`` carry) is the output
        minus that plane. With ``z_tip = coarse_gap − extend_sign·Z`` the physical tip is
        unaffected; what the loop has to follow — the surface the sim renders — becomes
        ``h_eff = h + extend_sign·plane``. On this rig (``extend_sign = −1``) that is the
        intuitive ``h − plane``: a sample rising at slope ``s`` along x is flattened by
        ``tilt_x = atan(s)``. Reported-Z slope per axis is ``−sample_slope/extend_sign −
        tan(tilt)``, so the response to a tilt step is −1 whichever way Z is wired, and the
        residual the instrument sees goes to zero at ``tilt = atan(−extend_sign·slope)``.

        A tilt step under closed loop is a step in the surface the loop follows (Z jumps
        by the plane's change at the tip position): a real Z transient, the reason MAST
        applies tilt in ≤1° sub-steps with the feedback on.
        """
        with self.lock:
            tx, ty = float(tilt_x_deg), float(tilt_y_deg)
            h_old = self.surface_height_here()
            self.piezo_tilt_deg = (tx, ty)
            es = float(self._extend_sign)
            self.surface.tilt_comp = (-es * math.tan(math.radians(tx)), -es * math.tan(math.radians(ty)))
            self._eq_cache.clear()
            if self.zctrl.on and not self.withdrawn and not self.approach.running:
                if abs(self.surface_height_here() - h_old) > 1e-13:
                    self._schedule_step_transient("tilt")
            self._event("tilt_set", tilt_x_deg=tx, tilt_y_deg=ty,
                        residual_mrad=[v * 1e3 for v in self.surface.residual_tilt()])

    # ── scanning ──
    def scan_start(self, direction_up: bool = False) -> None:
        with self.lock:
            st = self.scan.copy()
            st.scan_dir = "up" if direction_up else "down"
            if self.withdrawn or not self.zctrl.on:
                # controller scans anyway (feedback off → flat Z at the held value)
                pass
            n_tip0 = len(self.tip.events)
            fr = self.renderer.render(st, self.clock.sim())
            # rendering is instantaneous in sim time: the frame starts when the call returns
            now = self.clock.sim()
            fr.t_start_sim = now
            fr.rec_sim_s = now
            fr.row_times_sim = now + (st.line_time_fwd_s + st.line_time_bwd_s) * np.arange(st.ny)
            self.frame = fr
            self.tip_x, self.tip_y = st.cx, st.cy
            # tip_events: the slice of ``tip.events`` this render drew. Those changes carry
            # the time of the row they hit, which can lie past a stop — a replay needs to
            # know they belong to this frame to place them.
            self._event("scan_start", w_nm=st.w * 1e9, px=st.nx, line_s=st.line_time_fwd_s,
                        per_row_s=st.line_time_fwd_s + st.line_time_bwd_s, scan_dir=st.scan_dir,
                        tip_events=[n_tip0, len(self.tip.events)])

    def scan_stop(self) -> None:
        with self.lock:
            if self.frame is not None and not self.frame.finished:
                self.frame.progress(self.clock.sim())
                self.frame.stopped = True
                self._event("scan_stop", rows=self.frame.rows_done)

    def scan_running(self) -> bool:
        with self.lock:
            if self.frame is None:
                return False
            self.frame.progress(self.clock.sim())
            if self.frame.finished and not self.frame.stopped:
                self._on_frame_finished()
            return not (self.frame.finished or self.frame.stopped)

    def _on_frame_finished(self) -> None:
        fr = self.frame
        if fr is None or fr.saved_path is not None or getattr(fr, "_handled", False):
            return
        fr._handled = True  # type: ignore[attr-defined]
        if fr.settings.autosave:
            self.save_frame()
        if fr.settings.continuous:
            self.scan_start(direction_up=(fr.settings.scan_dir == "down") if fr.settings.bouncy else False)

    def save_frame(self) -> str | None:
        from ..io.sxm_writer import write_sxm
        with self.lock:
            fr = self.frame
            if fr is None:
                return None
            fr.progress(self.clock.sim())
            self.scan_series_counter += 1
            self.session_dir.mkdir(parents=True, exist_ok=True)
            name = f"{fr.settings.series_name or 'sim'}{self.scan_series_counter:03d}.sxm"
            path = self.session_dir / name
            write_sxm(path, fr, self)
            fr.saved_path = str(path)
            self.frames_saved.append(str(path))
            st = fr.settings
            ox, oy, _ = self.drift_offset(fr.t_start_sim)
            # geometry + the drift the frame was taken under: a judge maps a reported scan-frame
            # position onto the sample frame with these (see stmbench claims evidence rules)
            self._event("scan_saved", path=str(path), rows=fr.rows_done,
                        idx=self.scan_series_counter, complete=bool(fr.rows_done >= st.ny),
                        cx_m=st.cx, cy_m=st.cy, w_m=st.w, h_m=st.h, angle_deg=st.angle_deg,
                        nx=st.nx, ny=st.ny, bias_v=self.bias_v, setpoint_a=self.zctrl.setpoint_a,
                        t_start_sim=fr.t_start_sim, drift_m=[float(ox), float(oy)])
            return str(path)

    # ── tip shaper (poke) ──
    def tip_shaper_start(self, props: dict) -> None:
        """Schedule the four-segment Z/I trajectory and apply the tip/surface outcome."""
        with self.lock:
            switch_off_delay = float(props.get("switch_off_delay", 0.05))
            lift1 = float(props.get("tip_lift_m", -0.5e-9))          # negative = toward surface
            t1 = max(float(props.get("lift_time_1_s", 0.1)), 0.01)
            settle = float(props.get("bias_settling_s", 0.1))
            lift2 = float(props.get("lift_height_m", 0.5e-9))
            t2 = max(float(props.get("lift_time_2_s", 0.1)), 0.01)
            end_wait = float(props.get("end_wait_s", 0.5))
            restore = bool(props.get("restore_feedback", True))
            bias = float(props.get("bias_v", self.bias_v)) if props.get("change_bias") else self.bias_v
            hw_lag = 0.22 + float(self.rng.uniform(0, 0.11))     # measured: hardware starts 0.22–0.33 s late
            z0 = self.current_z_tip() if not self.withdrawn else self.surface_height_here() + 0.5e-9
            self.transients.clear()                               # the shaper takes over Z
            h_old = self.surface_height_here()
            depth = -lift1 if lift1 < 0 else 0.0
            outcome = self.tip.poke(self.clock.sim(), depth, bias)
            # surface memory
            sx, sy = self.sample_xy()
            if outcome["outcome"] in ("cluster", "pit"):
                sig = outcome.get("cluster_sigma_m", 1e-9)
                hgt = outcome.get("cluster_height_m", 0.4e-9)
                ratio = outcome.get("cluster_axis_ratio", 0.7)
                for a in self.tip.apexes:
                    self.surface.add_feature(Feature(sx + a.dx, sy + a.dy,
                                                     (-hgt if outcome["outcome"] == "pit" else hgt) * a.w,
                                                     sig, sig * ratio, float(self.rng.uniform(0, math.pi)),
                                                     kind=outcome["outcome"], born_sim_s=self.clock.sim()))
                # the same rule as a poke made by a Z sweep (_apply_poke_outcome): whatever
                # sat where the tip went in is gone
                self._lose_adatoms(sx, sy, 2 * sig, cause="poke")
            elif outcome["outcome"] == "ring_up":
                for _ in range(int(outcome.get("n_clusters", 8))):
                    self.surface.add_feature(Feature(sx + self.rng.normal(0, 15e-9), sy + self.rng.normal(0, 15e-9),
                                                     float(self.rng.uniform(0.5e-9, 2e-9)), float(self.rng.uniform(1e-9, 3e-9)),
                                                     kind="spatter", born_sim_s=self.clock.sim()))
                self._lose_adatoms(sx, sy, 15e-9, cause="poke")
            # trajectory (wall time)
            fs = 2000.0
            seg1 = hw_lag + switch_off_delay
            t_press_end = seg1 + t1
            t_settle_end = t_press_end + settle
            t_back_end = t_settle_end + t2
            t_fb = t_back_end + end_wait
            total = t_fb + 3.0
            t = np.arange(0, total, 1 / fs)
            z = np.empty_like(t)
            z_target = z0 + lift1
            for k, tt in enumerate(t):
                if tt < seg1:
                    z[k] = z0
                elif tt < t_press_end:
                    z[k] = z0 + lift1 * (tt - seg1) / t1
                elif tt < t_settle_end:
                    z[k] = z_target
                elif tt < t_back_end:
                    z[k] = z_target + lift2 * (tt - t_settle_end) / t2
                elif tt < t_fb:
                    z[k] = z_target + lift2
                else:
                    z[k] = np.nan
            # segment 4: feedback restores to the NEW equilibrium (surface changed under the tip)
            h_new = self.surface_height_here()
            z_eq_new = h_new + self.equilibrium_gap()
            m = np.isnan(z)
            if restore and m.any():
                loop = Loop(self.loop_params())
                tt = t[m] - t_fb
                z[m] = loop.transient(np.full(tt.size, z_eq_new), tt, float(z_target + lift2))
                self.zctrl.on = True
                self.zctrl.status = ZCTRL_ON
            else:
                z[m] = z_target + lift2
                self.zctrl.on = False
                self.zctrl.status = ZCTRL_OFF
            # segments 1-2 see the surface as it was; from the retract on, the surface has changed
            h_of_t = np.where(t < t_settle_end, h_old, h_new)
            gap = z - h_of_t
            i = current_metal(np.clip(gap, 0.02e-9, None), bias, self.phi_junction())
            i = np.where(gap <= 0.02e-9, self.preamp.clamp_a, i)
            i = self._desaturate(i, t)
            self.transients.append(Transient(self.clock.wall(), t, z, self.preamp.clamp(i), t_fb, "tip_shaper"))
            self.withdrawn = False
            self._event("poke", **{k: v for k, v in outcome.items() if k != "depth_m"}, depth_pm=depth * 1e12, bias_v=bias)

    def _desaturate(self, i: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Preamp exits saturation exponentially (τ from the rig profile)."""
        clamp = self.preamp.clamp_a
        out = i.copy()
        tau = self.preamp.desat_tau_s
        dt = float(t[1] - t[0]) if t.size > 1 else 1e-3
        sat = False
        level = 0.0
        for k in range(out.size):
            if abs(out[k]) >= clamp * 0.999:
                sat = True
                level = math.copysign(clamp, out[k])
            elif sat:
                level = level * math.exp(-dt / tau)
                if abs(level) > abs(out[k]):
                    out[k] = level
                else:
                    sat = False
        return out

    # ── bias pulse ──
    def bias_pulse(self, width_s: float, v: float, z_hold: bool, rel_abs: bool) -> None:
        with self.lock:
            vv = self.bias_v + v if rel_abs == 0 and False else v
            envelope = float(self.rig.get("tip.pulse_v_operator", 10.0))
            outcome = self.tip.pulse(self.clock.sim(), vv, width_s, envelope_v=max(envelope, 10.0))
            # spatter / crater on the surface — and the adatoms they land on
            sx, sy = self.sample_xy()
            if outcome["outcome"] != "no_effect":
                self._lose_adatoms(sx, sy, 1.5e-9, cause="pulse")        # the field under the apex
                for _ in range(int(self.rng.integers(2, 7))):
                    f = self.surface.add_feature(Feature(sx + self.rng.normal(0, 40e-9), sy + self.rng.normal(0, 40e-9),
                                                         float(self.rng.uniform(0.3e-9, 1.5e-9)), float(self.rng.uniform(1e-9, 4e-9)),
                                                         kind="spatter", born_sim_s=self.clock.sim()))
                    self._lose_adatoms(f.x, f.y, 0.5 * f.sigma_x, cause="spatter")
                if abs(vv) > 6:
                    f = self.surface.add_feature(Feature(sx, sy, -float(self.rng.uniform(0.2e-9, 1e-9)),
                                                         float(self.rng.uniform(3e-9, 10e-9)), kind="crater", born_sim_s=self.clock.sim()))
                    self._lose_adatoms(sx, sy, f.sigma_x, cause="pulse")
            # Z/I trajectory: during the pulse (z hold) current spikes to saturation, then the
            # loop finds the new equilibrium; the Z jump is the observable outcome
            fs = 2000.0
            pre = 0.05
            t = np.arange(0, pre + width_s + 2.5, 1 / fs)
            z0 = self.current_z_tip() if not self.withdrawn else self.surface_height_here() + 0.5e-9
            self.transients.clear()
            z_eq_new = self.surface_height_here() + self.equilibrium_gap()   # includes the apex-length jump
            z = np.full(t.size, z0)
            m_after = t >= pre + width_s
            # the observed Z jump (20-50 nm) is mostly a transient of the reshaped apex;
            # only a fraction persists as apex length (the tip model already added that fraction)
            dz_full = float(outcome.get("dz_m", 0.0))
            dz_transient = dz_full - float(self.tip.last_persistent_dz_m)
            if self.zctrl.on or not z_hold:
                loop = Loop(self.loop_params())
                tt = t[m_after] - (pre + width_s)
                target = z_eq_new + dz_transient * np.exp(-tt / 1.2)
                z[m_after] = loop.transient(target, tt, z0)
            i = current_metal(np.clip(z - self.surface_height_here(), 0.02e-9, None), self.bias_v, self.phi_junction())
            m_pulse = (t >= pre) & (t < pre + width_s)
            i[m_pulse] = self.preamp.clamp_a * math.copysign(1.0, vv)
            i = self._desaturate(i, t)
            self.transients.append(Transient(self.clock.wall(), t, z, self.preamp.clamp(i), pre + width_s, "bias_pulse"))
            # the sample locally rises under the tip if dz_m > 0 (material piled up)
            dz = outcome.get("dz_m", 0.0)
            if abs(dz) > 5e-9:
                self.surface.add_feature(Feature(sx, sy, dz * 0.02, 20e-9, kind="crater", born_sim_s=self.clock.sim()))
            self._event("pulse", v=vv, width_s=width_s, **outcome)

    # ── spectroscopy ──
    def sts_curve(self, v_start: float, v_end: float, n: int, z_offset_m: float = 0.0,
                  settle_s: float = 0.0, integ_s: float = 0.0) -> dict:
        with self.lock:
            vs = np.linspace(v_start, v_end, n)
            h = self.surface_height_here()
            z_tip = self.achievable_z_tip()
            gap = max(z_tip + z_offset_m - h, 0.02e-9)
            sx, sy = self.sample_xy()
            # position-dependent LDOS when the scenario switched the surface state on (standing
            # waves near steps / adatoms); otherwise the material template, exactly as before
            rho_s = self.surface.ldos_at(sx, sy)
            i = iv_curve(vs, gap, self.phi_junction(), rho_s, self.tip.ldos)
            t = np.arange(n) * (settle_s + integ_s)
            i = self.preamp.clamp(self.noise.current_series(i, self.clock.wall() + t))
            # unstable tip: a jump somewhere in the sweep
            jump = False
            if self.tip.lambda_per_s > 2e-3 and self.rng.random() < 0.5:
                k = int(self.rng.integers(n // 4, 3 * n // 4))
                i[k:] *= float(self.rng.uniform(0.5, 1.6))
                jump = True
            if n > 2 and abs(v_end - v_start) > 1e-9:
                didv = np.gradient(i, vs)
                didv = didv + self.rng.normal(0, 0.03 * float(np.nanmax(np.abs(didv))) + 1e-14, n)
            else:
                didv = np.zeros_like(i)
            self.sts_counter += 1
            near_kind, near_m = self.surface.nearest_scatterer(sx, sy)
            centre_nm = self.surface.corral_centre_distance_nm(sx, sy)
            self.sts_last = {"v": vs, "i": i, "didv": didv, "gap_m": gap, "jump": jump,
                             "phi_ev": self.phi_junction(), "sim_s": self.clock.sim(),
                             "idx": self.sts_counter,
                             "x_nm": self.tip_x * 1e9, "y_nm": self.tip_y * 1e9,
                             "sx_nm": sx * 1e9, "sy_nm": sy * 1e9,
                             "near_kind": near_kind, "near_nm": near_m * 1e9,
                             "corral_centre_dist_nm": centre_nm}
            self.sts_records.append({k: v for k, v in self.sts_last.items()
                                     if k not in ("v", "i", "didv")} |
                                    {"v": vs, "i": i, "didv": didv})
            self._event("sts", n=n, v0=v_start, v1=v_end, z_off_m=z_offset_m, jump=jump,
                        phi_ev=self.phi_junction(), idx=self.sts_counter,
                        x_m=self.tip_x, y_m=self.tip_y, sx_m=sx, sy_m=sy,
                        near_kind=near_kind, near_nm=near_m * 1e9,
                        corral_centre_dist_nm=centre_nm,
                        bias_v=self.bias_v, setpoint_a=self.zctrl.setpoint_a,
                        lockin_on=self.lockin_on, lockin_amp_v=self.lockin_amp_v)
            return self.sts_last

    # ── coarse motion / approach ──
    def motor_move(self, direction: str, steps: int) -> None:
        with self.lock:
            if direction in ("Z-", "Z+"):
                if direction == "Z-" and not self.withdrawn and self.zctrl.on:
                    # stepping the sample into a tip in tunnelling = crash
                    self._crash("coarse_z_while_in_contact", severity=3.0)
                self.coarse.step_z(steps, self.rng, direction=-1 if direction == "Z-" else +1)
                for k in range(steps):
                    self.noise.add_spike(self.clock.wall() + k / self.coarse.freq_hz,
                                         float(self.rng.uniform(82e-12, 375e-12)) * (1 if self.rng.random() < 0.5 else -1), 0.01)
            else:
                axis = direction[0]
                sign = 1 if direction.endswith("+") else -1
                self.coarse.xy_steps[direction] = self.coarse.xy_steps.get(direction, 0) + steps
                if steps >= 3:
                    self.surface.move_site(sign * (1 if axis == "X" else 0), sign * (1 if axis == "Y" else 0))
                    self.tip_x = float(self.rng.normal(0, 50e-9))
                    self.tip_y = float(self.rng.normal(0, 50e-9))
                    self._creep.clear()
                    self.note_move(np.array([steps * self.coarse.xy_step_m * sign, 0.0, 0.0]))
                if not self.withdrawn and self.zctrl.on and self.tip_height_m() - self.surface_height_here() < 200e-9:
                    self._crash("lateral_coarse_move_in_tunnelling", severity=2.0)
            self._event("motor_move", direction=direction, steps=steps, coarse_gap_um=self.coarse.coarse_gap_m * 1e6)

    def auto_approach_start(self) -> None:
        with self.lock:
            ap = self.approach
            ap.running = True
            ap.landed = False
            ap.aborted = False
            ap.cycles = 0
            ap.t_started_wall = self.clock.wall()
            ap.extend_time_s = float(2 * self.zctrl.limit_high_m / max(self.zctrl.i_m_per_s * 3.6, 1e-9))
            ap.extend_time_s = float(min(max(ap.extend_time_s, 0.5), 3.0))
            ap.t_next_cycle_wall = ap.t_started_wall + ap.cycle_time_s()
            self.withdrawn = False
            self.zctrl.on = True
            self.zctrl.status = ZCTRL_ON
            self._event("approach_start", coarse_gap_um=self.coarse.coarse_gap_m * 1e6)

    def auto_approach_poll(self) -> bool:
        """Advance the pulsed approach in wall time. Returns the running flag."""
        with self.lock:
            ap = self.approach
            if not ap.running:
                return False
            wall = self.clock.wall()
            while ap.running and wall >= ap.t_next_cycle_wall:
                # can the fine Z reach tunnelling from here?
                z_eq = self._equilibrium_z_tip()
                if self._fine_range_ok(z_eq):
                    ap.running = False
                    ap.landed = True
                    self.zctrl.on = True
                    self._schedule_step_transient("landing", duration_s=1.0)
                    self._event("approach_landed", cycles=ap.cycles, coarse_gap_um=self.coarse.coarse_gap_m * 1e6)
                    break
                # false landing: crosstalk spikes exceed the setpoint (scenario knob approach_noise)
                spike = float(self.rng.uniform(82e-12, 375e-12)) * self.approach_noise
                if spike > self.zctrl.setpoint_a * 1.2 and self.rng.random() < self.false_landing_p:
                    ap.running = False
                    ap.landed = False
                    self.zctrl.on = True
                    self._event("approach_false_landing", cycles=ap.cycles, spike_pa=spike * 1e12)
                    break
                self.coarse.step_z(ap.pulses_per_cycle, self.rng, direction=-1)
                # overshoot: even fully retracted the tip would be inside the sample
                if float(self.z_ctrl_from_tip(self._equilibrium_z_tip())) > self.zctrl.limit_high_m:
                    ap.running = False
                    self._crash("approach_overshoot", severity=4.0)
                    break
                ap.cycles += 1
                ap.t_next_cycle_wall += ap.cycle_time_s()
            if not ap.running and not ap.landed and not self.zctrl.on:
                pass
            return ap.running

    def auto_approach_stop(self) -> None:
        with self.lock:
            self.approach.running = False
            self.approach.aborted = True

    # ── scratches ──
    def maybe_scratch(self, st: ScanSettings, row: int, speed_m_per_s: float, t_sim: float) -> None:
        if speed_m_per_s > 450e-9 and self.tip.radius_m > 1.5e-9 and self.rng.random() < 0.02:
            xs, ys = st.pixel_xy(np.array([st.nx / 2]), row)
            ox, oy, _ = self.drift_offset(t_sim)
            self.surface.add_feature(Feature(float(xs[0]) + ox, float(ys[0]) + oy, -0.15e-9, st.w / 2, 1.5e-9,
                                             math.radians(st.angle_deg), kind="scratch", born_sim_s=t_sim))
            self.tip.lambda_per_s = min(0.05, self.tip.lambda_per_s * 1.5)

    # ── bookkeeping ──
    def _event(self, kind: str, **detail) -> None:
        self.events.append({"sim_s": self.clock.sim(), "wall_s": self.clock.wall(), "kind": kind, **detail})

    FLAT_WINDOW_LADDER_NM = (10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0)

    def flat_window_nm(self) -> float:
        """Largest ladder window (nm) centred on the sample point under the tip that crosses no
        step edge and holds no instrument-made damage; 0 when even 10 nm fails (B3 truth)."""
        sx, sy = self.sample_xy()
        best = 0.0
        for w_nm in self.FLAT_WINDOW_LADDER_NM:
            w = w_nm * 1e-9
            if not self.surface.step_free_window(sx, sy, w) or self.surface.damage_in_window(sx, sy, w):
                break
            best = w_nm
        return best

    def z_at_limit(self) -> bool:
        """Fine Z pinned within 2 % of a software limit while nominally in tunnelling — the
        'Z at +169.5 nm, I = 10 nA' signature of the range being eaten (not a crash)."""
        if self.withdrawn or not self.zctrl.on:
            return False
        z_n = float(self.zctrl.z_n)
        return z_n >= 0.98 * self.zctrl.limit_high_m or z_n <= 0.98 * self.zctrl.limit_low_m

    def truth(self) -> dict:
        with self.lock:
            self.achievable_z_tip()          # refresh z_n from the loop before reading it
            sts = self.sts_last
            sts_summary = None
            if sts is not None:
                v = np.asarray(sts["v"], float)
                sts_summary = {"n": int(v.size), "v0": float(v[0]) if v.size else 0.0,
                               "v1": float(v[-1]) if v.size else 0.0,
                               "jump": bool(sts.get("jump", False)),
                               "phi_ev": float(sts.get("phi_ev", self.phi_junction())),
                               "sim_s": float(sts.get("sim_s", 0.0))}
            # residual = what a frame shows after the piezo correction (B3 levelling truth);
            # the physical sample tilt is reported alongside and never changes
            resid = self.surface.residual_tilt()
            tilt = self.surface.site.tilt
            return {
                "sim_s": self.clock.sim(), "wall_s": self.clock.wall(),
                "material": self.surface.material.name,
                "ldos_onset_ev": self.surface.material.ldos_onset_ev,
                "tip": self.tip.snapshot(),
                "phi_junction_ev": self.phi_junction(),
                "ghost_contrast": self.tip.ghost_contrast(self.surface.material.corrugation_m, self.kappa_m()),
                "bias_v": self.bias_v, "setpoint_a": self.zctrl.setpoint_a,
                "zctrl_on": self.zctrl.on, "withdrawn": self.withdrawn,
                "z_n": self.zctrl.z_n, "z_at_limit": self.z_at_limit(),
                "coarse_gap_um": self.coarse.coarse_gap_m * 1e6,
                "tip_xy_nm": (self.tip_x * 1e9, self.tip_y * 1e9), "site": self.surface.site_index,
                "flat_window_nm": self.flat_window_nm(),
                # what the run was up against, and what it did about it. A ledger that records
                # the drift but not the compensation cannot distinguish an intrinsically stable
                # sample from one stabilized by active compensation.
                "drift": {
                    "v_xy_nm_per_min": float(np.hypot(*self.drift_v_m_per_s[:2])) * 6e10,
                    "v_z_pm_per_min": float(self.drift_v_m_per_s[2]) * 6e13,
                    "compensation_on": bool(self.drift_comp_on),
                    "compensation_nm_per_min": [float(v) * 6e10 for v in self.drift_comp_v],
                    "residual_nm_per_min": float(np.hypot(
                        *(self.drift_v_m_per_s[:2] + self.drift_comp_v[:2]))) * 6e10},
                "tilt_residual_mrad": (math.atan(resid[0]) * 1e3, math.atan(resid[1]) * 1e3),
                "sample_tilt_mrad": (math.atan(tilt[0]) * 1e3, math.atan(tilt[1]) * 1e3),
                "piezo_tilt_deg": tuple(self.piezo_tilt_deg),
                "damage_area_nm2": self.surface.damage_area_nm2(),
                "n_events": len(self.events), "frames_saved": len(self.frames_saved),
                "n_sts": sum(1 for e in self.events if e["kind"] == "sts"),
                "n_crash": sum(1 for e in self.events if e["kind"] == "crash"),
                "sts_last": sts_summary,
                # ── paper scenarios: what was drawn and what the generator actually holds ──
                "hidden": dict(self.hidden) if self.hidden else {},
                "herringbone": self.surface.herringbone_snapshot(),
                "surface_state": self.surface.surface_state_snapshot(),
                "adatoms": self.surface.adatoms_snapshot(),
                "corral": self.surface.corral_snapshot(),
                "n_zspec": len(self.zspec_records),
                "n_dats": len(self.dats_saved),
                **self._qplus_truth(),
            }

    def _qplus_truth(self) -> dict:
        """qPlus / force-law truth (P5). Empty on a rig without the sensor."""
        if self.force is None and self.pll is None:
            return {}
        out: dict = {}
        if self.pll is not None:
            out["qplus"] = {"f0_hz": float(getattr(self.tip, "qplus_f0_hz", 0.0)),
                            "k_n_per_m": float(getattr(self.tip, "qplus_k_n_per_m", 0.0)),
                            "q": float(getattr(self.tip, "qplus_q", 0.0)),
                            "amplitude_m": float(self.pll.amp_setpoint_m),
                            "output_on": bool(self.pll.output_on),
                            "center_freq_hz": float(self.pll.center_freq_hz)}
        if self.force is not None:
            from .forces import force_truth
            amp = float(self.pll.amp_setpoint_m) if self.pll is not None else 50e-12
            best = None
            for rec in self.zspec_records:
                if rec.get("near_kind") == "adatom" and (best is None or rec["near_nm"] < best["near_nm"]):
                    best = rec
            if best is not None and best.get("amplitude_m"):
                amp = float(best["amplitude_m"])
            out["force"] = force_truth(self.force, radius_m=self.tip.radius_m, amplitude_m=amp,
                                       f0_hz=float(getattr(self.tip, "qplus_f0_hz", 30e3)),
                                       k_n_per_m=float(getattr(self.tip, "qplus_k_n_per_m", 1800.0)))
            out["force"]["amplitude_used_m"] = amp
            out["zspec_best_adatom"] = ({k: v for k, v in best.items() if not hasattr(v, "shape")}
                                        if best is not None else None)
        return out

    def refresh_surface_state(self) -> None:
        """Drop cached LDOS / frame maps after the scatterer set changed (adatom moved)."""
        self.surface.invalidate_electronic_cache()
