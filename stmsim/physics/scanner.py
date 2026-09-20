"""Raster scanner: frame geometry, rendering (both directions, all channels), row reveal.

Geometry: pixel ``(i, j)`` (column, row) of an ``nx × ny`` frame with centre ``(cx, cy)``,
size ``(w, h)`` and angle ``θ`` (deg, counter-clockwise) maps to stage coordinates::

    u = (i + 0.5) / nx · w − w/2,   v = h/2 − (j + 0.5) / ny · h     (row 0 = top edge)
    x = cx + u cosθ − v sinθ,       y = cy + u sinθ + v cosθ

Rows are stored in **acquisition order**: for ``scan_dir == "down"`` row 0 is the top edge,
for ``"up"`` row 0 is the bottom edge (controller convention; the writer/reader flip).

Per line: forward pass samples ``x(u)`` left→right, backward pass right→left offset along
the fast axis by the piezo hysteresis loop (:meth:`Renderer.hysteresis_profile_px`: a
constant part plus a mid-line bulge, both measured on calibration frames); both go through the
feedback :class:`Loop` at the pixel rate; the Z channel is the loop output, the Current
channel the tracking error mapped back through the junction. Slow drift/creep offsets are
applied per row from the world's drift model. Tip-change hazards are drawn per row and
applied to the tip *in place* so all following rows see the changed tip (a row DC jump).
A crashed (flickering) tip additionally gets its own apex arrangement and its own mid-line
apex-length hops on *each* pass (``Tip.pass_apexes`` / ``Tip.flicker_series``), which is
what makes its trace and retrace disagree.

Live buffer semantics (``Scan.FrameDataGrab``): rows not yet acquired are **zeros**;
saved ``.sxm`` files put NaN there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .feedback import Loop
from .junction import current_metal

# controller global signal indices used for the scan buffer (reference-stm layout)
SIG_CURRENT = 0
SIG_Z = 14
SIG_BIAS = 24


@dataclass
class ScanSettings:
    cx: float = 0.0
    cy: float = 0.0
    w: float = 100e-9
    h: float = 100e-9
    angle_deg: float = 0.0
    nx: int = 256
    ny: int = 256
    line_time_fwd_s: float = 0.586
    line_time_bwd_s: float = 0.586
    scan_dir: str = "down"                 # down | up
    continuous: bool = False
    bouncy: bool = False
    autosave: bool = False
    autopaste: bool = False
    series_name: str = "sim"
    comment: str = ""
    modules_names: list[str] = field(default_factory=list)
    channels: list[int] = field(default_factory=lambda: [SIG_CURRENT, SIG_Z])
    speed_fwd_m_per_s: float | None = None
    speed_bwd_m_per_s: float | None = None

    def copy(self) -> "ScanSettings":
        return ScanSettings(**{**self.__dict__, "modules_names": list(self.modules_names),
                               "channels": list(self.channels)})

    @property
    def frame_time_s(self) -> float:
        return self.ny * (self.line_time_fwd_s + self.line_time_bwd_s)

    @property
    def nm_per_px(self) -> float:
        return self.w / self.nx * 1e9

    def pixel_xy(self, i, j):
        """Stage coordinates of pixel centres (arrays ok). Row ``j`` counts from the TOP."""
        i = np.asarray(i, float)
        j = np.asarray(j, float)
        u = (i + 0.5) / self.nx * self.w - self.w / 2
        v = self.h / 2 - (j + 0.5) / self.ny * self.h
        th = math.radians(self.angle_deg)
        c, s = math.cos(th), math.sin(th)
        return self.cx + u * c - v * s, self.cy + u * s + v * c

    def row_index_for_acq(self, k: int) -> int:
        """Acquisition row k → geometric row (from top)."""
        return k if self.scan_dir == "down" else self.ny - 1 - k


@dataclass
class Frame:
    settings: ScanSettings
    data: dict[int, tuple[np.ndarray, np.ndarray]]      # signal index → (fwd, bwd), acquisition row order
    t_start_wall: float
    t_start_sim: float
    rows_total: int
    rows_done: int = 0
    finished: bool = False
    stopped: bool = False
    saved_path: str | None = None
    row_times_sim: np.ndarray | None = None
    tip_change_rows: list[int] = field(default_factory=list)
    rec_sim_s: float = 0.0

    def progress(self, sim_now: float) -> int:
        if self.finished or self.stopped:
            return self.rows_done
        per_row = self.settings.line_time_fwd_s + self.settings.line_time_bwd_s
        rows = int((sim_now - self.t_start_sim) // per_row)
        self.rows_done = max(0, min(self.rows_total, rows))
        if self.rows_done >= self.rows_total:
            self.finished = True
        return self.rows_done

    def live_buffer(self, sig: int, direction: str) -> np.ndarray:
        """controller-style live buffer: acquired rows, zeros elsewhere."""
        fwd, bwd = self.data[sig]
        src = fwd if direction == "fwd" else bwd
        out = np.zeros_like(src)
        n = self.rows_done
        out[:n] = src[:n]
        return out

    def saved_arrays(self, sig: int) -> tuple[np.ndarray, np.ndarray]:
        """As stored in .sxm: NaN for unacquired rows (acquisition order kept)."""
        fwd, bwd = self.data[sig]
        f = fwd.copy()
        b = bwd.copy()
        f[self.rows_done:] = np.nan
        b[self.rows_done:] = np.nan
        return f, b


class Renderer:
    """Renders a whole frame from the world state at scan start."""

    OVERSAMPLE = 4
    MIN_SIGMA_PX = 1.0     # calibrated so a good tip gives MAST trace/retrace ≈0.95-0.98 at 100 nm (reference: 0.93-0.98)

    def __init__(self, world):
        self.w = world

    def _sampled_line(self, st: ScanSettings, col_pos: np.ndarray, j: int, ox: float, oy: float,
                      kappa: float, apexes=None, bias_v: float = 0.0) -> np.ndarray:
        """Effective height averaged over each pixel's dwell footprint along the fast axis."""
        w = self.w
        n = self.OVERSAMPLE
        sub = (np.arange(n) + 0.5) / n - 0.5                     # offsets within a pixel
        pos = (np.asarray(col_pos, float)[:, None] + sub[None, :]).ravel()
        xs, ys = st.pixel_xy(pos, j)
        smooth, atomic = w.tip.height_parts(w.surface, xs + ox, ys + oy, kappa, apexes=apexes,
                                            bias_v=bias_v)
        return smooth.reshape(-1, n).mean(axis=1), atomic.reshape(-1, n).mean(axis=1)

    @staticmethod
    def hysteresis_profile_px(cols: np.ndarray, nx: int, hyst_px: float, shape: str = "loop",
                              edge_frac: float = 0.62) -> np.ndarray:
        """Retrace offset (px, +x) at each backward pixel ``cols`` (geometric column index).

        ``shape == "loop"``: a piezo hysteresis loop whose branches only partly close at the
        turnarounds — a constant part plus a mid-line bulge,
        ``δ(u) = hyst_px·(e + (1−e)·1.5·4u(1−u))`` with ``u`` the position along the line and
        ``e = edge_frac`` the offset at the line ends as a fraction of the mean. The mean over
        the line is ``hyst_px`` whatever ``e`` (what the whole-frame overlap-NCC peak shift in
        ``calibrate/hysteresis.py`` reads back). ``e`` was measured on 39 80–120 nm Au
        frames (World.__init__): left / centre / right thirds 0.98 / 1.37 / 1.17 % of the axis
        ⇒ e ≈ 0.62 (model thirds 1.07 / 1.37 / 1.07); a fully closed loop (e = 0, edge thirds
        at 54 % of the centre) was evaluated first and rejected by the same measurement (reference
        edge thirds ≥ 71 % of the centre). ``"shift"``: the constant offset ``hyst_px``."""
        cols = np.asarray(cols, float)
        if shape == "shift" or nx <= 1:
            return np.full(cols.shape, float(hyst_px))
        e = float(min(max(edge_frac, 0.0), 1.0))
        u = (cols + 0.5) / nx
        return float(hyst_px) * (e + (1.0 - e) * 1.5 * 4.0 * u * (1.0 - u))

    def render(self, st: ScanSettings, t_start_sim: float) -> Frame:
        w = self.w
        nx, ny = st.nx, st.ny
        kappa = w.kappa_m()
        loop = Loop(w.loop_params())
        dt_px_f = st.line_time_fwd_s / nx
        dt_px_b = st.line_time_bwd_s / nx
        gap = w.equilibrium_gap()
        i_set = w.zctrl.setpoint_a
        sigma_m = w.tip.sigma_eff_m(w.phi_junction())
        # tip radius blur ⊕ loop/ADC bandwidth ⊕ dwell: 100 nm/256 px calibration frames show a
        # 10-90 % step-edge width of ~1.2 px (72 rig frames), i.e. σ ≈ 0.5-0.6 px minimum
        sigma_px = math.hypot(sigma_m / (st.w / nx), self.MIN_SIGMA_PX)
        hyst_px = w.hysteresis_px(st)
        cols_b = np.arange(nx)[::-1]                              # backward pass: right → left
        bwd_pos = cols_b + self.hysteresis_profile_px(cols_b, nx, hyst_px, getattr(w, "hyst_shape", "loop"),
                                                      getattr(w, "hyst_edge_frac", 0.62))
        z_fwd = np.zeros((ny, nx))
        z_bwd = np.zeros((ny, nx))
        i_fwd = np.zeros((ny, nx))
        i_bwd = np.zeros((ny, nx))
        per_row = st.line_time_fwd_s + st.line_time_bwd_s
        row_times = t_start_sim + per_row * np.arange(ny)
        cols = np.arange(nx)
        z_prev = None
        tip_rows: list[int] = []
        fb_on = w.zctrl.on
        z_frozen_tip = w.tip_height_m()
        speed = st.w / max(st.line_time_fwd_s, 1e-9)
        # standing waves cost Bessel/Struve evaluations per sample; computing them once for the
        # whole frame keeps Scan.Action inside the controller client's 5 s reply window (no-op without a
        # surface state)
        prepare = getattr(w.surface, "prepare_frame", None)
        if prepare is not None:
            cxs, cys = st.pixel_xy(np.array([0.0, st.nx - 1.0, 0.0, st.nx - 1.0]),
                                   np.array([0.0, 0.0, st.ny - 1.0, st.ny - 1.0]))
            ox0, oy0, _ = w.drift_offset(t_start_sim)
            ox1, oy1, _ = w.drift_offset(float(row_times[-1]))
            prepare(float(np.min(cxs)) + min(ox0, ox1), float(np.max(cxs)) + max(ox0, ox1),
                    float(np.min(cys)) + min(oy0, oy1), float(np.max(cys)) + max(oy0, oy1),
                    w.bias_v)
        for k in range(ny):
            j = st.row_index_for_acq(k)
            t_row = row_times[k]
            # slow drift / creep offset at this row (stage frame)
            ox, oy, oz = w.drift_offset(t_row)
            # tip-change hazard for this row (Poisson over one row time)
            if w.tip.maybe_change(per_row, t_row, current_a=i_set, bias_v=w.bias_v):
                tip_rows.append(k)
            # imaging at a manipulation resistance drags (or picks up) adatoms mid-frame:
            # later rows see the new positions, producing the corresponding row jump in the frame
            w.registry_scan_row(st, j, ox, oy, t_row, speed)
            # forward — pixel-dwell averaging: the current is integrated while the tip moves
            # one pixel, so sub-pixel structure (an aliased lattice at 0.8 nm/px) averages out
            # the mesoscopic blur applies to the smooth surface only; the atomic lattice is
            # already apex-transferred (Tip.height_parts) and must not be blurred along one
            # axis — a fast-axis-only blur leaves the slow-axis lattice components intact and
            # introduces radial fast-axis anisotropy.
            # a flickering (crashed) tip: each pass images its own apex arrangement
            s_f, a_f = self._sampled_line(st, cols, j, ox, oy, kappa, apexes=w.tip.pass_apexes(),
                                          bias_v=w.bias_v)
            if sigma_px > 0.3:
                s_f = ndimage.gaussian_filter1d(s_f, sigma_px, mode="nearest")
            h_f = s_f + a_f + oz
            fl = w.tip.flicker_series(nx, dt_px_f)          # crashed tip: mid-line apex-length hops
            if fl is not None:
                h_f = h_f + fl
            if fb_on:
                z_line_tip = loop.track(h_f + gap, dt_px_f, z0=z_prev)
            else:
                z_line_tip = np.full(nx, z_frozen_tip)
            # backward: acquired right→left, offset along the fast axis by the hysteresis loop
            s_b, a_b = self._sampled_line(st, bwd_pos, j, ox, oy, kappa, apexes=w.tip.pass_apexes(),
                                          bias_v=w.bias_v)
            if sigma_px > 0.3:
                s_b = ndimage.gaussian_filter1d(s_b, sigma_px, mode="nearest")
            h_b = s_b + a_b + oz
            fl = w.tip.flicker_series(nx, dt_px_b)          # drawn afresh: the passes disagree
            if fl is not None:
                h_b = h_b + fl
            if fb_on:
                z_line_tip_b = loop.track(h_b + gap, dt_px_b, z0=float(z_line_tip[-1]))
                z_prev = float(z_line_tip_b[-1])
            else:
                z_line_tip_b = np.full(nx, z_frozen_tip)
            # scratches: a fragile/blunt tip scanning too fast damages the surface
            w.maybe_scratch(st, j, speed, t_row)
            # channels
            z_fwd[k] = w.z_ctrl_from_tip(z_line_tip) + w.noise.z_series(nx) * (1 + 0.3 * w.tip.lambda_per_s * 1e2)
            z_bwd[k] = w.z_ctrl_from_tip(z_line_tip_b) + w.noise.z_series(nx)
            gf = np.clip(z_line_tip - h_f, 0.05e-9, None)
            gb = np.clip(z_line_tip_b - h_b, 0.05e-9, None)
            i_f = current_metal(gf, w.bias_v, w.phi_junction())
            i_b = current_metal(gb, w.bias_v, w.phi_junction())
            i_fwd[k] = w.preamp.clamp(w.noise.current_series(i_f, t_row + dt_px_f * cols))
            i_bwd[k] = w.preamp.clamp(w.noise.current_series(i_b, t_row + st.line_time_fwd_s + dt_px_b * cols))
        data = {SIG_Z: (z_fwd, z_bwd), SIG_CURRENT: (i_fwd, i_bwd),
                SIG_BIAS: (np.full((ny, nx), w.bias_v), np.full((ny, nx), w.bias_v))}
        for sig in st.channels:
            if sig not in data:
                data[sig] = (np.zeros((ny, nx)), np.zeros((ny, nx)))
        return Frame(settings=st.copy(), data=data, t_start_wall=w.clock.wall(),
                     t_start_sim=t_start_sim, rows_total=ny, row_times_sim=row_times,
                     tip_change_rows=tip_rows, rec_sim_s=t_start_sim)
