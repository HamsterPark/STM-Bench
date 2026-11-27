"""Stripe analysis of the Au(111) herringbone: what an operator reads off a frame of the zigzag.

Pure functions of a frame's arrays and geometry: no world, no truth, no seed. P1's reference
solution and its baseline both measure through this module, so the two cannot disagree about
what "the period" or "the direction" of a place is.

What a frame looks like (the field of ``stmsim.physics.herringbone`` with ``elbow: continuous``,
Barth et al. 1990). The lines of the reconstruction bend without a jump: along each stripe of the
mean direction the lines run at +alpha, then -alpha (alpha 28-32 degrees, so two neighbouring arms
are 56-64 degrees apart), every L/2 = 20-30 nm. Each stretch where the lines run one way is a
**rotational domain** in the paper's sense (and the judge's, for P1's orientation claims): the
direction that counts at a place is the direction of the lines *there*. Two readings follow.

* **Arms, from the spectrum of a window** (:func:`assess_window`). The spectrum is two *combs* of
  lines 1/L apart, one around the wavevector of each arm, with almost nothing on the mean direction
  (at most ~0.19 of the strongest line in 120 nm windows). The wavevector of an arm has
  ``|k| = 1 / (T cos alpha)``: the period between the lines of one arm, which is what the period
  claim asks for. The strongest single line of a comb is off the arm period by up to 0.2 nm, so
  the period of an arm is read from the power-weighted centroid of its comb, not from a peak. A
  window that holds three or more arm directions (a region wall) returns each of them.
* **The direction at a place** (:func:`local_direction`, :func:`orientation_map`). A real-space
  structure tensor of the band-passed frame gives the direction of the lines at every pixel; the
  reading at a point also says how uniform the field is around it (``spread_deg``), so that a
  caller can report in the middle of an arm, away from the elbow lines (20-30 nm apart), and
  whether a step is near. :func:`arm_places` finds such places, one per direction.

Conventions
-----------
* Frames are instrument frames: row 0 is the top (largest scan y), column 0 the left, one pixel is
  ``nm_per_px`` on both axes, ``angle_deg`` is the scan rotation (clockwise positive, as the
  ``scan`` action takes it). Heights are metres.
* Directions are the directions of the lines (the way the bright lines run) in **scan-frame**
  degrees, counter-clockwise from +x with y up, modulo 180. A frame taken at ``angle_deg`` has its
  image axes turned clockwise by that angle, so an image direction ``phi`` is ``phi - angle_deg`` in
  the scan frame. Positions are scan-frame nm, mapped with :func:`pixel_to_scan_nm`, the same
  mapping :func:`stmbench.render.pixel_to_scan_nm` uses (pinned by a test).

Round 1 of this module (2026-11-27, the "jump" zigzag: arms 16-36 degrees apart, a strong peak on
the mean direction, a mean-of-two-arms answer) is replaced: its pairing and "mixed" rules do not
describe this spectrum.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

#: band searched for the arm combs, as the period of a line (nm): the arm period is 6.0-6.6 nm and
#: a comb reaches a little beyond it on either side
BAND_PERIOD_NM = (4.5, 9.5)
#: zero padding of the spectrum (the centroid is continuous, the padding only smooths the lobes)
PAD_FACTOR = 2
#: Gaussian sigma (deg) of the angular power histogram in which the arm lobes are found, and the
#: least separation of two lobes. Arms of one zigzag are 56-64 degrees apart, a comb spreads over
#: about +-15 degrees
LOBE_SIGMA_DEG = 6.0
LOBE_MIN_SEP_DEG = 35.0
#: radius (cycles/nm) of the disc around an arm's wavevector over which its comb is summed, and
#: the estimator: the power-weighted mean of |k| in the disc. Chosen on 30 synthetic continuous
#: zigzags of the P1 ranges (120 nm windows, 0.4 nm/px, 3 pm noise), against 0.03 / 0.06 and the
#: length of the vector centroid: this one has p90 0.045 nm and max 0.052 nm error (a steady
#: +0.03 nm); 0.03 gives max 0.097, 0.06 a +0.05 bias, the vector centroid +0.05 (it is shortened
#: by the comb's angular spread)
COMB_RADIUS = 0.045
#: an arm counts when its comb holds this fraction of the strongest comb's power, and its mean
#: power density is this many times the median of the band
ARM_REL_POWER_MIN = 0.12
ARM_QUALITY_MIN = 8.0
#: Gaussian sigma of the step detector's low-pass, in stripe periods
STEP_SIGMA_PERIODS = 1.0
#: peak-to-peak (pm) of the quadratic-levelled low-pass above which a window is called stepped
#: (steps 100-250 pm, step-free up to 25 pm: seed-6 windows of the first calibration, 2026-11-27)
STEP_PP_PM = 45.0
#: structure-tensor scales (nm): the band-pass keeps the lines (period ~6.3 nm) and drops steps,
#: tilt and adsorbates; the integration scale sets how locally the direction is read (an arm is
#: 20-30 nm wide)
BANDPASS_NM = (1.0, 4.0)
TENSOR_SIGMA_NM = 4.0
#: a step (or a jump of whole rows) is near when the low-pass slope exceeds the frame's median by this
STEP_SLOPE_PM_NM = 4.0
STEP_SLOPE_SIGMA_NM = 6.0
#: reading at a point: radius of the disc over which uniformity is judged, and what counts as uniform
LOCAL_RADIUS_NM = 8.0
MAX_SPREAD_DEG = 4.0
MIN_COHERENCE = 0.5


@dataclass
class Arm:
    """One comb of the spectrum: the lines of one arm of the zigzag."""
    direction_deg: float     # direction of the lines, scan frame, mod 180
    period_nm: float         # between the lines, perpendicular to them (the arm period)
    power: float             # power of the comb, relative to the strongest comb of the window = 1
    quality: float           # mean power density of the comb / median of the band
    kx: float = 0.0          # centroid wavevector (cycles/nm), scan frame (x right, y up)
    ky: float = 0.0


@dataclass
class Assessment:
    """Reading of one square window. ``verdict`` is ``ok`` (at least one arm), ``step`` (a step edge
    or a jump of whole rows crosses the window), ``no_stripe`` (no comb above the background).
    ``arms`` are strongest first."""
    verdict: str
    reason: str = ""
    size_nm: float = 0.0
    nm_per_px: float = 0.0
    arms: list[Arm] = field(default_factory=list)
    quality: float = 0.0
    step_pp_pm: float = 0.0
    centre_nm: tuple[float, float] | None = None  # scan frame, set by the frame-level helpers

    @property
    def measurable(self) -> bool:
        return self.verdict == "ok"

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    @property
    def direction_deg(self) -> float | None:
        """Direction of the strongest arm's lines."""
        return self.arms[0].direction_deg if self.arms else None

    @property
    def period_along_arm_nm(self) -> float | None:
        """The arm period (what the period claim asks for): the power-weighted mean over the arms."""
        if not self.arms:
            return None
        w = np.array([a.power for a in self.arms])
        return float(np.sum(w * np.array([a.period_nm for a in self.arms])) / w.sum())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LocalReading:
    """The direction of the lines at a point and how uniform it is around it."""
    direction_deg: float          # scan frame, mod 180
    spread_deg: float             # rms deviation of the field from it within ``radius_nm`` (deg)
    coherence: float              # mean structure-tensor coherence within the disc, 0-1
    step_nearby: bool             # a step or row jump within the disc
    radius_nm: float
    x_nm: float
    y_nm: float

    @property
    def ok(self) -> bool:
        """Uniform, coherent and clear of steps: a place to report a direction at."""
        return self.spread_deg <= MAX_SPREAD_DEG and self.coherence >= MIN_COHERENCE and not self.step_nearby


@dataclass
class Place:
    """A place in the middle of a stretch of uniform direction (``arm_places``)."""
    x_nm: float
    y_nm: float
    direction_deg: float
    clearance_nm: float           # distance to the nearest pixel of another direction or a step
    spread_deg: float


# ── geometry ──────────────────────────────────────────────────────────────────
def pixel_to_scan_nm(geom: dict, col: float, row: float) -> tuple[float, float]:
    """Frame pixel (``col`` right, ``row`` down from the top edge; pixel centres at half-integers)
    to scan-frame nm. ``geom`` is the instrument's frame geometry (``center_nm``, ``size_nm``,
    ``pixels``, ``angle_deg``); a positive angle turns the offset from the centre clockwise."""
    n = int(geom["pixels"])
    size = float(geom["size_nm"])
    u = -size / 2 + (float(col) + 0.5) * size / n
    v = size / 2 - (float(row) + 0.5) * size / n
    a = math.radians(float(geom.get("angle_deg", 0.0) or 0.0))
    c, s = math.cos(a), math.sin(a)
    cx, cy = geom["center_nm"]
    return float(cx) + c * u + s * v, float(cy) - s * u + c * v


def scan_nm_to_pixel(geom: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Inverse of :func:`pixel_to_scan_nm`: ``(col, row)`` (fractional, centres at half-integers)."""
    n = int(geom["pixels"])
    size = float(geom["size_nm"])
    a = math.radians(float(geom.get("angle_deg", 0.0) or 0.0))
    c, s = math.cos(a), math.sin(a)
    cx, cy = geom["center_nm"]
    dx, dy = float(x_nm) - float(cx), float(y_nm) - float(cy)
    u = c * dx - s * dy
    v = s * dx + c * dy
    return (u + size / 2) * n / size - 0.5, (size / 2 - v) * n / size - 0.5


def _to_scan_deg(image_deg: float, angle_deg: float) -> float:
    return (image_deg - float(angle_deg)) % 180.0


def _fold180(d: float) -> float:
    """Angle difference of two directions modulo 180, in [0, 90]."""
    d = abs(d) % 180.0
    return min(d, 180.0 - d)


def _mean_direction(a: float, b: float) -> tuple[float, float]:
    """(bisector, half the separation) of two directions mod 180, the bisector of the short way."""
    d = ((b - a + 90.0) % 180.0) - 90.0
    return (a + d / 2.0) % 180.0, abs(d) / 2.0


# ── leveling and steps ────────────────────────────────────────────────────────
def _clean(z: np.ndarray) -> np.ndarray:
    a = np.asarray(z, float)
    bad = ~np.isfinite(a)
    if bad.any():
        a = np.where(bad, np.nanmedian(a) if (~bad).any() else 0.0, a)
    return a


def level_plane(z: np.ndarray) -> np.ndarray:
    """``z`` minus its least-squares plane (the sample's tilt and the scanner's bow are the
    first thing in a frame; a stripe grating has no component in a plane)."""
    a = _clean(z)
    ny, nx = a.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    X = np.column_stack([np.ones(a.size), xx.ravel() / nx, yy.ravel() / ny])
    coef, *_ = np.linalg.lstsq(X, a.ravel(), rcond=None)
    return a - (X @ coef).reshape(a.shape)


def step_score_pm(z: np.ndarray, nm_per_px: float, period_nm: float = 6.3) -> float:
    """Peak-to-peak (pm) of the quadratic-levelled low-pass of the window, edges excluded: ~0 for a
    flat, tilted or bowed terrace with stripes and adsorbates on it, ~100 pm and more for a step
    (or for a jump of whole rows, which a tip change leaves in a frame)."""
    a = _clean(z) * 1e12
    sigma_px = STEP_SIGMA_PERIODS * period_nm / nm_per_px
    low = ndimage.gaussian_filter(a, sigma_px, mode="nearest")
    m = int(round(1.5 * sigma_px))
    ny, nx = low.shape
    if ny - 2 * m < 8 or nx - 2 * m < 8:
        m = 0
    low = low[m:ny - m, m:nx - m]
    # a quadratic, not a plane: the Z loop's creep and the scanner's bow curve a window by tens of pm
    # (seed 55 of the P1 calibration: a step-free frame scored 50 pm after a plane), a step is not
    # a quadratic
    ny2, nx2 = low.shape
    yy, xx = np.mgrid[0:ny2, 0:nx2]
    u, v = xx.ravel() / nx2 - 0.5, yy.ravel() / ny2 - 0.5
    X = np.column_stack([np.ones(u.size), u, v, u * u, u * v, v * v])
    coef, *_ = np.linalg.lstsq(X, low.ravel(), rcond=None)
    return float(np.ptp(low.ravel() - X @ coef))


# ── spectrum: the arms of a window ─────────────────────────────────────────────
def _window2d(ny: int, nx: int) -> np.ndarray:
    return np.outer(np.hanning(ny + 2)[1:-1], np.hanning(nx + 2)[1:-1])


def find_arms(z: np.ndarray, nm_per_px: float, *, band_period_nm: tuple[float, float] = BAND_PERIOD_NM,
              comb_radius: float = COMB_RADIUS) -> tuple[list[Arm], float]:
    """The arm combs of one levelled, Hann-windowed square window, strongest first, and the median
    power of the band (the background). Wavevectors and directions are in the **image** frame here
    (x right, y up); :func:`assess_window` turns them into scan-frame ones.

    The lobes of the angular power histogram of the band (smoothed by ``LOBE_SIGMA_DEG``, at least
    ``LOBE_MIN_SEP_DEG`` apart) are the arms. For each, the disc of ``comb_radius`` around the
    lobe's mean wavevector is re-centred on its own power-weighted centroid until it settles, and
    the period is that of the power-weighted mean |k| of the disc."""
    a = level_plane(z)
    ny, nx = a.shape
    d = float(nm_per_px)
    zw = a * _window2d(ny, nx)
    N = int(2 ** math.ceil(math.log2(PAD_FACTOR * max(ny, nx))))
    P = np.abs(np.fft.fft2(zw, s=(N, N))) ** 2
    kax = np.fft.fftfreq(N, d)
    KX, KY = np.meshgrid(kax, -kax)                 # array rows run down, y runs up
    klo, khi = 1.0 / band_period_nm[1], 1.0 / band_period_nm[0]
    # only the part of the spectrum that can matter, as flat arrays
    sel = (np.abs(KX) <= khi + comb_radius) & (np.abs(KY) <= khi + comb_radius)
    P, KX, KY = P[sel], KX[sel], KY[sel]
    KR = np.hypot(KX, KY)
    band = (KR >= klo) & (KR <= khi)
    if not band.any():
        return [], 0.0
    background = float(np.median(P[band]))
    if background <= 0:
        background = float(np.finfo(float).tiny)
    phi = np.degrees(np.arctan2(KY, KX)) % 180.0     # direction of k, folded: P(k) = P(-k)
    hist = np.zeros(180)
    np.add.at(hist, np.floor(phi[band]).astype(int) % 180, P[band])
    kk = np.arange(-30, 31)
    ker = np.exp(-0.5 * (kk / LOBE_SIGMA_DEG) ** 2)
    smooth = sum(w * np.roll(hist, int(c)) for w, c in zip(ker, kk))
    peaks = [i for i in range(180) if smooth[i] > 0 and smooth[i] >= smooth[(i - 1) % 180]
             and smooth[i] >= smooth[(i + 1) % 180] and smooth[i] >= 0.1 * smooth.max()]
    peaks.sort(key=lambda i: -smooth[i])
    lobes: list[int] = []
    for i in peaks:
        if all(min(abs(i - j), 180 - abs(i - j)) >= LOBE_MIN_SEP_DEG for j in lobes):
            lobes.append(i)
    out: list[tuple[float, float, float, float, float]] = []      # (kx, ky, |k|, power, quality)
    for i in lobes:
        near = band & (np.minimum(np.abs(phi - i), 180 - np.abs(phi - i)) <= 15.0)
        if not near.any() or P[near].sum() <= 0:
            continue
        kmean = float(np.sum(P[near] * KR[near]) / np.sum(P[near]))
        ang = math.radians(i)
        K = np.array([math.cos(ang), math.sin(ang)]) * kmean
        w = None
        for _ in range(12):
            disc = (np.hypot(KX - K[0], KY - K[1]) < comb_radius) & band
            w = P * disc
            tot = w.sum()
            if tot <= 0:
                break
            Kn = np.array([(w * KX).sum(), (w * KY).sum()]) / tot
            done = math.hypot(*(Kn - K)) < 1e-5
            K = Kn
            if done:
                break
        if w is None or w.sum() <= 0:
            continue
        kabs = float((w * KR).sum() / w.sum())
        n_disc = max(int(np.count_nonzero(w)), 1)
        out.append((float(K[0]), float(K[1]), kabs, float(w.sum()), float(w.sum() / n_disc / background)))
    out.sort(key=lambda t: -t[3])
    arms: list[Arm] = []
    for kx, ky, kabs, pw, q in out:
        if any(math.hypot(kx - a_.kx, ky - a_.ky) < comb_radius for a_ in arms):
            continue                              # the same comb found from two lobes
        top = out[0][3]
        if pw < ARM_REL_POWER_MIN * top or q < ARM_QUALITY_MIN:
            continue
        arms.append(Arm(direction_deg=(math.degrees(math.atan2(ky, kx)) + 90.0) % 180.0,
                        period_nm=1.0 / kabs, power=pw / top, quality=q, kx=kx, ky=ky))
    return arms, background


# ── one window ────────────────────────────────────────────────────────────────
def assess_window(z: np.ndarray, nm_per_px: float, *, angle_deg: float = 0.0,
                  check_step: bool = True) -> Assessment:
    """Read one square window of a frame (a 2-D array of heights in metres, row 0 = top): the arms
    (direction, period, relative power) it holds, or why it cannot be read."""
    a = np.asarray(z, float)
    ny, nx = a.shape
    out = Assessment(verdict="no_stripe", size_nm=max(ny, nx) * float(nm_per_px), nm_per_px=float(nm_per_px))
    if np.mean(np.isfinite(a)) < 0.9:
        out.reason = "the window holds unacquired (NaN) pixels"
        return out
    out.step_pp_pm = step_score_pm(a, nm_per_px)
    if check_step and out.step_pp_pm > STEP_PP_PM:
        out.verdict = "step"
        out.reason = f"a step edge or a jump of rows crosses the window (levelled low-pass {out.step_pp_pm:.0f} pm)"
        return out
    arms, _ = find_arms(a, nm_per_px)
    if not arms:
        out.reason = "no stripe comb above the background in the band"
        return out
    th = math.radians(-float(angle_deg))
    for arm in arms:
        arm.direction_deg = _to_scan_deg(arm.direction_deg, angle_deg)
        arm.kx, arm.ky = (arm.kx * math.cos(th) - arm.ky * math.sin(th),
                          arm.kx * math.sin(th) + arm.ky * math.cos(th))
    out.arms = arms
    out.quality = arms[0].quality
    out.verdict = "ok"
    return out


# ── windows of a frame ────────────────────────────────────────────────────────
def crop_window(frame: dict, centre_nm: tuple[float, float], size_nm: float) -> np.ndarray | None:
    """The square window of ``size_nm`` centred on scan-frame ``centre_nm``, cut from the frame's
    image array (axis-aligned in the image), or ``None`` when it does not fit inside the frame."""
    n = int(frame["pixels"])
    d = float(frame["nm_per_px"])
    col, row = scan_nm_to_pixel(frame, *centre_nm)
    m = int(round(size_nm / d))
    c0 = int(round(col + 0.5 - m / 2))
    r0 = int(round(row + 0.5 - m / 2))
    if c0 < 0 or r0 < 0 or c0 + m > n or r0 + m > n or m < 8:
        return None
    return np.asarray(frame["array"])[r0:r0 + m, c0:c0 + m]


def assess_at(frame: dict, centre_nm: tuple[float, float], size_nm: float) -> Assessment | None:
    """:func:`assess_window` on the window of ``frame`` at a scan-frame position; ``centre_nm``
    on the result is the window's actual centre (rounded to a pixel)."""
    win = crop_window(frame, centre_nm, size_nm)
    if win is None:
        return None
    m = win.shape[0]
    n = int(frame["pixels"])
    col, row = scan_nm_to_pixel(frame, *centre_nm)
    c0 = int(round(col + 0.5 - m / 2))
    r0 = int(round(row + 0.5 - m / 2))
    a = assess_window(win, float(frame["nm_per_px"]), angle_deg=float(frame.get("angle_deg", 0.0) or 0.0))
    a.centre_nm = pixel_to_scan_nm(frame, c0 + m / 2 - 0.5, r0 + m / 2 - 0.5)
    return a


def survey(frame: dict, size_nm: float, stride_nm: float | None = None) -> list[Assessment]:
    """Windows of ``size_nm`` on a regular grid over the frame (stride defaults to the window size,
    so the windows tile it), each assessed; row-major from the top left."""
    n = int(frame["pixels"])
    d = float(frame["nm_per_px"])
    m = int(round(size_nm / d))
    stride = max(1, int(round((stride_nm if stride_nm else size_nm) / d)))
    arr = np.asarray(frame["array"])
    angle = float(frame.get("angle_deg", 0.0) or 0.0)
    out: list[Assessment] = []
    for r0 in range(0, n - m + 1, stride):
        for c0 in range(0, n - m + 1, stride):
            a = assess_window(arr[r0:r0 + m, c0:c0 + m], d, angle_deg=angle)
            a.centre_nm = pixel_to_scan_nm(frame, c0 + m / 2 - 0.5, r0 + m / 2 - 0.5)
            out.append(a)
    return out


def median_period_nm(assessments: list[Assessment]) -> float | None:
    """The along-arm period over the windows that measured one: their median. One window's
    reading can sit a fraction of a bin off; the median over windows is what is reported."""
    v = [a.period_along_arm_nm for a in assessments if a.measurable and a.period_along_arm_nm is not None]
    return float(np.median(v)) if v else None


# ── the direction of the lines at a place ──────────────────────────────────────
@dataclass
class OrientationMap:
    """The direction of the lines at every pixel of a frame (:func:`orientation_map`)."""
    direction_deg: np.ndarray     # scan frame, mod 180, same shape as the frame
    coherence: np.ndarray         # structure-tensor coherence, 0-1
    step_near: np.ndarray         # bool: a step or a jump of rows is here (low-pass slope)
    nm_per_px: float
    geom: dict


def orientation_map(frame: dict, *, sigma_nm: float = TENSOR_SIGMA_NM,
                    bandpass_nm: tuple[float, float] = BANDPASS_NM) -> OrientationMap:
    """Structure-tensor orientation of the whole frame. The frame is band-passed (a Gaussian of
    ``bandpass_nm[0]`` minus one of ``bandpass_nm[1]``: the 6 nm lines stay, steps, tilt and
    adsorbates go), its gradient tensor is averaged with a Gaussian of ``sigma_nm``, and the lines
    run perpendicular to the dominant gradient. ``step_near`` marks where the frame's low-pass slope
    exceeds its median by ``STEP_SLOPE_PM_NM`` (a step, a row jump or a tall cluster), dilated by
    the tensor's reach, so that a reading there can be refused."""
    z = np.asarray(frame["array"], float)
    d = float(frame["nm_per_px"])
    angle = float(frame.get("angle_deg", 0.0) or 0.0)
    a = _clean(z) * 1e12
    bp = ndimage.gaussian_filter(a, bandpass_nm[0] / d, mode="nearest") \
        - ndimage.gaussian_filter(a, bandpass_nm[1] / d, mode="nearest")
    gy, gx = np.gradient(bp)
    gy = -gy                                          # array rows run down, y runs up
    s = sigma_nm / d
    jxx = ndimage.gaussian_filter(gx * gx, s, mode="nearest")
    jyy = ndimage.gaussian_filter(gy * gy, s, mode="nearest")
    jxy = ndimage.gaussian_filter(gx * gy, s, mode="nearest")
    grad_dir = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    tot = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.where(tot > 0, tot, 1.0)
    direction = (grad_dir + 90.0 - angle) % 180.0
    low = ndimage.gaussian_filter(a, STEP_SLOPE_SIGMA_NM / d, mode="nearest")
    ly, lx = np.gradient(low, d)
    slope = np.hypot(lx, ly)
    near = slope > np.median(slope) + STEP_SLOPE_PM_NM
    near = ndimage.binary_dilation(near, iterations=max(1, int(round(sigma_nm / d))))
    return OrientationMap(direction_deg=direction, coherence=coh, step_near=near, nm_per_px=d, geom=dict(frame))


def _disc(omap: OrientationMap, x_nm: float, y_nm: float, radius_nm: float):
    """Row and column index arrays of the pixels of the disc around a scan-frame point."""
    n = omap.direction_deg.shape[0]
    col, row = scan_nm_to_pixel(omap.geom, x_nm, y_nm)
    r = radius_nm / omap.nm_per_px
    r0, r1 = max(0, int(math.floor(row - r))), min(n, int(math.ceil(row + r)) + 1)
    c0, c1 = max(0, int(math.floor(col - r))), min(n, int(math.ceil(col + r)) + 1)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    inside = (rr - row) ** 2 + (cc - col) ** 2 <= r * r
    return rr[inside], cc[inside]


def local_direction(frame: dict, x_nm: float, y_nm: float, *, radius_nm: float = LOCAL_RADIUS_NM,
                    omap: OrientationMap | None = None) -> LocalReading | None:
    """The direction of the lines at a scan-frame point and how uniform it is within ``radius_nm``.
    The direction is the coherence-weighted mean of the field over the disc (double-angle average);
    ``spread_deg`` is the rms deviation of the field from it. ``None`` if the disc is not inside the
    frame. Pass ``omap`` (from :func:`orientation_map`) to read several points of one frame."""
    om = omap if omap is not None else orientation_map(frame)
    n = om.direction_deg.shape[0]
    col, row = scan_nm_to_pixel(om.geom, x_nm, y_nm)
    r = radius_nm / om.nm_per_px
    if col - r < 0 or row - r < 0 or col + r > n - 1 or row + r > n - 1:
        return None
    rr, cc = _disc(om, x_nm, y_nm, radius_nm)
    th = np.radians(om.direction_deg[rr, cc])
    w = om.coherence[rr, cc]
    c2, s2 = np.sum(w * np.cos(2 * th)), np.sum(w * np.sin(2 * th))
    mean = (0.5 * math.degrees(math.atan2(s2, c2))) % 180.0
    dev = ((om.direction_deg[rr, cc] - mean + 90.0) % 180.0) - 90.0
    return LocalReading(direction_deg=mean, spread_deg=float(np.sqrt(np.mean(dev ** 2))),
                        coherence=float(np.mean(w)), step_nearby=bool(om.step_near[rr, cc].any()),
                        radius_nm=float(radius_nm), x_nm=float(x_nm), y_nm=float(y_nm))


def arm_places(frame: dict, *, tol_deg: float = 8.0, min_clearance_nm: float = 5.0,
               omap: OrientationMap | None = None) -> list[Place]:
    """The middle of each stretch of uniform line direction in a frame, one place per direction,
    best first. A pixel is *uniform* when the field is coherent, clear of steps and within
    ``tol_deg`` of the direction in question; for each direction found (the histogram of the
    coherent, step-free pixels, peaks at least 25 degrees apart) the place is the pixel farthest from
    anything not uniform in that direction (distance transform), so a stretch's middle, away from
    the elbow lines. Places with less than ``min_clearance_nm`` are dropped."""
    om = omap if omap is not None else orientation_map(frame)
    good = (om.coherence >= MIN_COHERENCE) & ~om.step_near
    if not good.any():
        return []
    dirs = om.direction_deg
    hist, _ = np.histogram(dirs[good], bins=180, range=(0.0, 180.0))
    kk = np.arange(-8, 9)
    ker = np.exp(-0.5 * (kk / 3.0) ** 2)
    sm = sum(w * np.roll(hist, int(c)) for w, c in zip(ker, kk)).astype(float)
    peaks = sorted((i for i in range(180) if sm[i] > 0 and sm[i] >= sm[(i - 1) % 180] and sm[i] >= sm[(i + 1) % 180]),
                   key=lambda i: -sm[i])
    chosen: list[int] = []
    for i in peaks:
        if sm[i] >= 0.05 * sm[peaks[0]] and all(_fold180(i - j) >= 25.0 for j in chosen):
            chosen.append(i)
    out: list[Place] = []
    for i in chosen:
        dev = np.abs(((dirs - (i + 0.5) + 90.0) % 180.0) - 90.0)
        mask = good & (dev <= tol_deg)
        mask = ndimage.binary_opening(mask, iterations=2)
        if not mask.any():
            continue
        dist = ndimage.distance_transform_edt(mask) * om.nm_per_px
        # a reading's disc has to fit in the frame: keep clear of its edge
        n = dist.shape[0]
        edge = int(round(max(min_clearance_nm, LOCAL_RADIUS_NM) / om.nm_per_px)) + 1
        dist[:edge, :] = 0
        dist[n - edge:, :] = 0
        dist[:, :edge] = 0
        dist[:, n - edge:] = 0
        r, c = np.unravel_index(int(np.argmax(dist)), dist.shape)
        if dist[r, c] < min_clearance_nm:
            continue
        x, y = pixel_to_scan_nm(om.geom, float(c), float(r))
        rd = local_direction(om.geom, x, y, radius_nm=min(LOCAL_RADIUS_NM, float(dist[r, c])), omap=om)
        if rd is None:
            continue
        out.append(Place(x_nm=x, y_nm=y, direction_deg=rd.direction_deg, clearance_nm=float(dist[r, c]),
                         spread_deg=rd.spread_deg))
    out.sort(key=lambda p: -p.clearance_nm)
    return out
