"""Stripe analysis of the Au(111) herringbone: what an operator reads off a frame's Fourier spectrum.

Pure functions of a frame's arrays and geometry: no world, no truth, no seed. P1's reference
solution and its baseline both measure through this module, so the two cannot disagree about
what "the period" or "the direction" of a window is.

What a herringbone window looks like. Inside one rotational domain the stripes zigzag: two arms
at +-alpha about the mean stripe direction (alpha 8-18 degrees), alternating every half chevron
period along the stripes. The spectrum of a window therefore has **two arm peaks**, both at the
same ``|k| = 1 / (T cos alpha)``, so the period read off an arm peak is the period along the
arm, ``T cos alpha``: the quantity the paper's claim is scored on
(``herringbone.period_along_arm_nm``). Measured on gratings built from the simulator's own
height expression, whose phase ``u + s zig(v) v`` jumps at every elbow: there is also a peak on
the *mean* direction, at the longer period ``T`` and about as strong as an arm, and weaker
sidebands (chevron modulation) that must not be taken for arms. A window that crosses a domain
wall shows the peaks of the other domain too, a window that crosses a step has its low
frequencies dominated by the step, and a blunt or noisy window shows no peak at all.

Conventions
-----------
* Frames are instrument frames: row 0 is the top (largest scan y), column 0 the left, one pixel is
  ``nm_per_px`` on both axes, ``angle_deg`` is the scan rotation (clockwise positive, as the
  ``scan`` action takes it). Heights are metres.
* Directions are stripe directions (the way the bright lines run) in **scan-frame** degrees,
  counter-clockwise from +x with y up, modulo 180. A frame taken at ``angle_deg`` has its image
  axes turned clockwise by that angle, so an image direction ``phi`` is ``phi - angle_deg`` in the
  scan frame. Positions are scan-frame nm, mapped with :func:`pixel_to_scan_nm`, the same
  mapping :func:`stmbench.render.pixel_to_scan_nm` uses (pinned by a test).

Method. The window is levelled by a plane, multiplied by a Hann window, zero-padded and
transformed; each local maximum of the power spectrum inside the stripe band is then refined to
a continuous wavevector by maximising the window's own Fourier sum there (no FFT-bin snapping:
a 120 nm window has one bin per third of a nanometre of period, and the claim's tolerance is
0.15 nm). A step is told from stripes by the low-pass of the levelled window, in which the 6 nm
stripes are averaged away and a 235 pm step is not.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage, optimize

#: stripe band searched, as the period of a peak (nm): the along-arm period is 6.0 cos(18 deg)
#: .. 6.6 (5.7-6.6 nm) in the scenario; a blunt tip shifts nothing, so the band is that, widened
BAND_PERIOD_NM = (4.8, 8.2)
#: zero padding of the coarse spectrum (the refinement removes the remaining bin limit)
PAD_FACTOR = 4
#: a peak counts when its power is this fraction of the strongest, and this many times the median
#: power of the band around it
REL_POWER_MIN = 0.12
QUALITY_MIN = 12.0
#: Gaussian sigma of the step detector's low-pass, in stripe periods (stripes are averaged to
#: exp(-2 pi^2 sigma^2 / T^2) of their height, 1e-9 at sigma = T): a 235 pm step survives
STEP_SIGMA_PERIODS = 1.0
#: peak-to-peak (pm) of the quadratic-levelled low-pass above which a window is called stepped. Seed 6's
#: 80-100 nm windows (measured 2026-11-27): the ones a step crosses score 100-250 pm, the step-free
#: ones up to 25 pm; Hann-windowed stripes do not care about a step in the outer 10 % of a window.
#: Native adsorbates (tens of pm, ~1 nm wide) stay below it.
STEP_PP_PM = 45.0
#: two peaks are the two arms of one domain when their stripe directions are within this angle
#: (2 alpha <= 36 deg) and their |k| within ARM_K_TOL
ARM_MAX_SEP_DEG = 40.0
#: 0.6 %: the mean-direction peak's period is that of the arms times 1 / cos(alpha), 1.0 % at the
#: smallest alpha (8 degrees), so a tolerance above half of that takes an arm and the mean peak for
#: the two arms (seed 24 of the P1 calibration: mean read 3.7 degrees off); two true arms agree to
#: about 0.2 % on synthetic windows
ARM_K_TOL = 0.006
#: peaks weaker than this fraction of the strongest are the zigzag's sidebands, not arms. Chosen on
#: 120 synthetic 120 nm windows (the simulator's height expression, 3 pm noise): at 0.55 nine percent
#: of two-arm windows are called mixed (a sideband counts), at 0.65 two percent, at 0.75 a weak arm
#: starts to be missed and readings go wrong (3 of 120 beyond 0.03 nm / 3 deg)
MAJOR_FRAC = 0.65
#: the mean-direction peak sits within this angle of the bisector of the two arms
MEAN_PEAK_TOL_DEG = 3.0
#: peaks below this fraction of the strongest are not candidates for the arm group
GROUP_FRAC = 0.3
#: a peak this close to a peak of the group is a sideband of it, not another domain: the sidebands
#: sit 10-16 degrees from an arm (chevron 22-34 nm, |k| 0.16/nm), the arms of an adjacent domain
#: at least 60 - 2 * 18 = 24 degrees
SIDEBAND_REACH_DEG = 20.0


@dataclass
class Peak:
    kx: float                # cycles / nm, scan frame (x right, y up)
    ky: float
    power: float             # |windowed Fourier sum|^2, arbitrary units
    quality: float           # power / median band power
    period_nm: float
    stripe_deg: float        # direction of the lines this peak describes, scan frame, mod 180


@dataclass
class Assessment:
    """Reading of one square window. ``verdict`` is ``ok`` (two arms), ``one_arm`` (a
    direction that is an arm or the mean, both accepted by the judge; a period only when an arm and
    the mean-direction peak were both seen, never from a lone peak, which may be the mean one) or
    the reason nothing can be said (``step``, ``no_stripe``, ``mixed`` (several domains or an
    unresolved set of peaks))."""
    verdict: str
    reason: str = ""
    size_nm: float = 0.0
    nm_per_px: float = 0.0
    period_along_arm_nm: float | None = None     # what the period claim asks for
    period_mean_nm: float | None = None           # T: across the stripes of the mean direction
    arm_deg: list[float] = field(default_factory=list)
    mean_deg: float | None = None
    half_angle_deg: float | None = None
    quality: float = 0.0
    step_pp_pm: float = 0.0
    n_peaks: int = 0
    peaks: list[Peak] = field(default_factory=list)
    centre_nm: tuple[float, float] | None = None  # scan frame, set by the frame-level helpers

    @property
    def measurable(self) -> bool:
        return self.verdict in ("ok", "one_arm")

    @property
    def direction_deg(self) -> float | None:
        """The direction to report: the mean of the two arms, else the single arm's."""
        if self.mean_deg is not None:
            return self.mean_deg
        return self.arm_deg[0] if self.arm_deg else None

    def to_dict(self) -> dict:
        return asdict(self)


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


# ── spectrum ──────────────────────────────────────────────────────────────────
def _window2d(ny: int, nx: int) -> np.ndarray:
    return np.outer(np.hanning(ny + 2)[1:-1], np.hanning(nx + 2)[1:-1])


def _dtft_power(zw: np.ndarray, kx: float, ky: float, d: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """|sum zw(x, y) exp(-2 pi i (kx x + ky y))|^2, x in nm (right), y in nm (up)."""
    ex = np.exp(-2j * np.pi * kx * xs)
    ey = np.exp(-2j * np.pi * ky * ys)
    return float(np.abs(ey @ zw @ ex) ** 2)


def _refine(zw: np.ndarray, k0: tuple[float, float], d: float, xs: np.ndarray, ys: np.ndarray,
            span: float) -> tuple[float, float, float]:
    """Continuous maximum of the window's Fourier power near ``k0`` (within +-span cycles/nm)."""
    scale = _dtft_power(zw, k0[0], k0[1], d, xs, ys)
    if scale <= 0:
        return k0[0], k0[1], 0.0

    def f(p):
        return -_dtft_power(zw, k0[0] + p[0] * span, k0[1] + p[1] * span, d, xs, ys) / scale

    r = optimize.minimize(f, [0.0, 0.0], method="Nelder-Mead",
                          options={"xatol": 2e-3, "fatol": 1e-9, "initial_simplex": [[0, 0], [0.4, 0], [0, 0.4]]})
    p = np.clip(r.x, -1.0, 1.0)
    kx, ky = k0[0] + p[0] * span, k0[1] + p[1] * span
    return float(kx), float(ky), _dtft_power(zw, kx, ky, d, xs, ys)


def find_peaks(z: np.ndarray, nm_per_px: float, *, band_period_nm: tuple[float, float] = BAND_PERIOD_NM,
               max_peaks: int = 6) -> tuple[list[Peak], float]:
    """Stripe peaks of one levelled, Hann-windowed square window, strongest first, with the
    median band power as the background. Wavevectors are in the **image** frame (x right, y up)
    here; :func:`assess_window` turns them into scan-frame directions."""
    a = level_plane(z)
    ny, nx = a.shape
    d = float(nm_per_px)
    w = _window2d(ny, nx)
    zw = a * w
    N = int(2 ** math.ceil(math.log2(PAD_FACTOR * max(ny, nx))))
    F = np.fft.fft2(zw, s=(N, N))
    P = np.abs(F) ** 2
    ky_axis = -np.fft.fftfreq(N, d)          # array rows run down, y runs up
    kx_axis = np.fft.fftfreq(N, d)
    KX, KY = np.meshgrid(kx_axis, ky_axis)
    KR = np.hypot(KX, KY)
    klo, khi = 1.0 / band_period_nm[1], 1.0 / band_period_nm[0]
    band = (KR >= klo) & (KR <= khi)
    if not band.any():
        return [], 0.0
    Pb = np.where(band, P, 0.0)
    # local maxima of the padded spectrum (a flank of a stronger peak is not a peak)
    mx = ndimage.maximum_filter(Pb, size=5, mode="wrap")
    cand = np.argwhere((Pb == mx) & (Pb > 0))
    order = np.argsort(-Pb[cand[:, 0], cand[:, 1]])
    cand = cand[order]
    if cand.size == 0:
        return [], 0.0
    background = float(np.median(P[band]))
    if background <= 0:
        background = float(np.finfo(float).tiny)
    xs = (np.arange(nx) - (nx - 1) / 2) * d
    ys = ((ny - 1) / 2 - np.arange(ny)) * d          # row 0 = top = +y
    span = 1.2 / (max(nx, ny) * d)                    # about one unpadded bin either side
    top = float(Pb[cand[0, 0], cand[0, 1]])
    found: list[tuple[float, float, float]] = []
    for r, c in cand:
        if Pb[r, c] < REL_POWER_MIN * top * 0.5 or len(found) >= 3 * max_peaks:
            break
        kx, ky, pw = _refine(zw, (float(KX[r, c]), float(KY[r, c])), d, xs, ys, span)
        # the real signal's mirror peak is the same line; keep one representative per +-k
        dup = False
        for fx, fy, _ in found:
            if math.hypot(kx - fx, ky - fy) < 1.0 / (max(nx, ny) * d) or \
               math.hypot(kx + fx, ky + fy) < 1.0 / (max(nx, ny) * d):
                dup = True
                break
        if not dup:
            found.append((kx, ky, pw))
    found.sort(key=lambda t: -t[2])
    peaks: list[Peak] = []
    # the refined power is on the DTFT's scale, the background on the padded FFT's: same units
    # (both are |sum|^2 of the same windowed data)
    for kx, ky, pw in found[:max_peaks]:
        kr = math.hypot(kx, ky)
        if not (klo * 0.9 <= kr <= khi * 1.1):
            continue
        if kx < 0 or (kx == 0 and ky < 0):
            kx, ky = -kx, -ky
        peaks.append(Peak(kx=kx, ky=ky, power=pw, quality=pw / background, period_nm=1.0 / kr,
                          stripe_deg=(math.degrees(math.atan2(ky, kx)) + 90.0) % 180.0))
    return peaks, background


# ── one window ────────────────────────────────────────────────────────────────
def assess_window(z: np.ndarray, nm_per_px: float, *, angle_deg: float = 0.0,
                  check_step: bool = True) -> Assessment:
    """Read one square window of a frame (a 2-D array of heights in metres, row 0 = top)."""
    a = np.asarray(z, float)
    ny, nx = a.shape
    size = max(ny, nx) * float(nm_per_px)
    out = Assessment(verdict="no_stripe", size_nm=size, nm_per_px=float(nm_per_px))
    if np.mean(np.isfinite(a)) < 0.9:
        out.reason = "the window holds unacquired (NaN) pixels"
        return out
    out.step_pp_pm = step_score_pm(a, nm_per_px)
    if check_step and out.step_pp_pm > STEP_PP_PM:
        out.verdict = "step"
        out.reason = f"a step edge crosses the window (levelled low-pass {out.step_pp_pm:.0f} pm)"
        return out
    peaks, _ = find_peaks(a, nm_per_px)
    strong = [p for p in peaks if p.quality >= QUALITY_MIN]
    out.n_peaks = len(strong)
    if not strong:
        out.reason = "no stripe peak above the background in the band"
        out.quality = peaks[0].quality if peaks else 0.0
        return out
    top = strong[0].power
    strong = [p for p in strong if p.power >= REL_POWER_MIN * top]
    for p in strong:
        p.stripe_deg = _to_scan_deg(p.stripe_deg, angle_deg)
        # k is an image-frame vector; rotate it into the scan frame for the record
        th = math.radians(-angle_deg)
        p.kx, p.ky = (p.kx * math.cos(th) - p.ky * math.sin(th), p.kx * math.sin(th) + p.ky * math.cos(th))
    out.peaks = strong
    out.n_peaks = len(strong)
    out.quality = strong[0].quality
    # What the spectrum of one domain holds (measured on gratings built from the simulator's height
    # expression): the two arm peaks at the period T cos(alpha); a peak on the mean direction, at
    # the longer period T and about as strong as an arm; and sidebands of the chevron modulation
    # (a shorter period, 10-16 degrees from an arm, up to ~75 % of an arm when the chevron is short
    # and alpha large). The arms are the best pair of equal-period peaks, with the mean peak
    # between them when there is one; a further peak that is not a sideband (more than
    # SIDEBAND_REACH_DEG from every peak of the group) and about as strong as the weaker arm
    # belongs to another domain.
    top = strong[0].power
    pool = [p for p in strong if p.power >= GROUP_FRAC * top]

    def equal_period(x: Peak, y: Peak) -> bool:
        return abs(x.period_nm - y.period_nm) <= ARM_K_TOL * min(x.period_nm, y.period_nm)

    best = None                                    # (score, lo, hi, mid | None)
    for i, x in enumerate(pool):
        for y in pool[i + 1:]:
            if not equal_period(x, y) or _fold180(x.stripe_deg - y.stripe_deg) > ARM_MAX_SEP_DEG:
                continue
            centre, _ = _mean_direction(x.stripe_deg, y.stripe_deg)
            mids = [m for m in pool if m is not x and m is not y
                    and _fold180(m.stripe_deg - centre) <= MEAN_PEAK_TOL_DEG
                    and m.period_nm >= 0.995 * min(x.period_nm, y.period_nm)]
            mid = max(mids, key=lambda m: m.power) if mids else None
            score = x.power + y.power + (mid.power if mid else 0.0) * 0.5
            if best is None or score > best[0]:
                best = (score, x, y, mid)
    if best is not None:
        _, x, y, mid = best
        group = [x, y] + ([mid] if mid else [])
        weakest = min(x.power, y.power)
    else:
        p = strong[0]
        partner = None
        for q in strong[1:]:
            ratio = max(p.period_nm, q.period_nm) / min(p.period_nm, q.period_nm)
            near = _fold180(p.stripe_deg - q.stripe_deg) <= ARM_MAX_SEP_DEG
            if q.power >= MAJOR_FRAC * p.power and near and 1.006 <= ratio <= 1.07:
                partner = q               # an arm and the mean-direction peak (the longer period)
                break
        group = [p] + ([partner] if partner else [])
        weakest = p.power
    extras = [q for q in strong if q not in group and q.power >= MAJOR_FRAC * weakest
              and min(_fold180(q.stripe_deg - g.stripe_deg) for g in group) > SIDEBAND_REACH_DEG]
    out.n_peaks = len(group)
    if extras:
        out.verdict = "mixed"
        out.reason = (f"{len(extras)} further stripe peak(s) as strong as the arms and "
                      f"more than {SIDEBAND_REACH_DEG:.0f} deg from them: more than one rotational domain")
        return out
    if best is not None:
        _, x, y, mid = best
        out.verdict = "ok"
        out.arm_deg = [x.stripe_deg, y.stripe_deg]
        out.mean_deg, out.half_angle_deg = _mean_direction(x.stripe_deg, y.stripe_deg)
        out.period_along_arm_nm = 0.5 * (x.period_nm + y.period_nm)
        out.period_mean_nm = out.period_along_arm_nm / math.cos(math.radians(out.half_angle_deg))
        return out
    arm = min(group, key=lambda g: g.period_nm)
    out.verdict = "one_arm"
    out.reason = ("two peaks of different period: one arm and the mean-direction peak" if len(group) == 2
                  else "one stripe peak: an arm of the zigzag (or the arms are not resolved)")
    # a lone peak may be the mean-direction peak (period T, up to 5 % above the arms'): its direction is
    # accepted either way, its period is not the along-arm period, so none is given. With an arm and
    # the mean peak both seen, the arm is the one with the shorter period and the period is its own.
    out.period_along_arm_nm = arm.period_nm if len(group) == 2 else None
    out.arm_deg = [arm.stripe_deg]
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
