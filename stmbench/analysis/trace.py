"""Reading the trace of a move: did an atom hop under the tip, and how often?

An operator who drags an atom is not told whether it followed. The controller records the tip
height and the current while the tip moves (``move_tip`` keeps them like a spectrum:
``Instrument.spectrum(name)`` of the trace it names, ``"kind": "move"``), and what the atom did
is read from them. In constant-current mode an atom that follows the
tip gives a sawtooth in Z with the period of the lattice (Bartels, Meyer & Rieder, PRL 79, 697
(1997)): the height under the tip falls a little as the tip moves on, and jumps when the atom
hops after it. This module reads that from a trace record. It reads nothing but the record: the
arrays ``z_abs`` and ``dist`` and the flag ``feedback`` (a Z loop that is off holds Z, and Z
shows nothing).

**What the sawtooth is worth.** One tooth is the drop of the bump between two hops: the atom's
apparent height after the tip's blur is 40-90 pm for a good tip, and the atom jumps when it lags
the tip by 0.15 nm, where the bump has fallen by about a tenth (3-5 pm). A sample of a slow
drag is the mean over its dwell, with 0.4-0.9 pm of the instrument's noise at 0.25-4 nm/s, and
the lattice makes the differences between samples spread by 0.5-1 pm even where the noise is less. The jump at a
hop is 1-2 pm at the median (4-13 pm at the 90th percentile), so a single tooth is hardly a
signal and a train of teeth is a faint ripple. What stands out is the *fall*: a tip that leaves
the atom behind goes from the top of its bump to the bare surface, 20-90 pm over a stretch of
the path as wide as the tip's blur, and *where* on the path that fall happens says how far the
atom came along: with no hop it falls from the start, with n hops it holds its level for about n
lattice spacings first. A trace that does not fall at all belongs to an atom that stayed under
the tip, or to a tip that never had an atom. The other thing that stands out is the jump of a
hop that came late: an atom that lagged a long way before the pull made it hop is put back under
the tip in one step of 10-25 pm out of a fall, which the schedule that the fit assumes has no
place for.

So the reading fits the trace with the one thing that is known about a drag, a bump that the atom
carries along for n hops and then leaves (:func:`fit`: a Gaussian of unknown width and height
on a sloping background, the atom jumping at the lag the lattice allows), for every n, and
asks which n the trace is best explained by and whether that beats the best no-hop explanation by
more than noise could (:data:`SIGNIFICANT`). Where the fit finds no hop but Z jumps up out of a
fall (:func:`hop_steps`), the jump is read as a hop. **It assumes the tip started over the atom it
drags** (a drag that begins on bare surface has no fall either); the baselines place the tip
there by measuring the atom first.

**How well it reads** (2026-10-27, P4 with its four physics switches on, against the world's own
events, read after each drag; seeds 0-119, which draw all five tips, the ladder 400 kohm x 0.8^k for
14 rungs, 4 nm drags, the tip put 0.05 nm (rms) from the atom as a slow frame would; a small
version is ``tests/test_analysis_trace.py``). At 0.25 nm/s, the scripted baseline's speed (1506
drags in which the atom was not picked up): of the 1066 in which an atom hopped, 1062 (99.6 %)
read as hopping - after 1 hop 94 %, 2 or more 100 % - and of the 440 in which it did not, 6
(1.4 %; 4 of them among the 34 drags in which the tip changed, where a jump of Z out of a fall
is a hop to the reader whatever made it). Without the jump test the fit alone read 97.5 %, 93.8 %
and 77.6 % of the hopped drags at 0.25, 1 and 4 nm/s, and 0.5 % of those that stayed. The count is right to one hop in 88-100 % of the
drags in which the atom was left behind (1-11 hops) and to three in 95 % and 88 % of those that
carried it 12-15 and 16 or more hops (the whole way: about 17 hops, 16 on a lattice row, more with
a blunt tip's wandering hops). At 1 nm/s 99.1 % and 0.7 %; at 4 nm/s, where an atom hops once or
twice at most, 97.0 % and 0.6 %. An atom taken up by the tip reads as moved in 96, 90 and 83 % at
0.25, 1 and 4 nm/s. Drags under two nanometres leave too little of the fall to tell it from a
drifting background: in drags of 0.2, 0.3, 0.6, 1.0, 1.5 and 2.5 nm an atom that stayed reads as
moved in 100, 100, 92, 63, 4 and 3 %, and one that hopped as hopping in 100, 100, 100, 98, 98 and
100 % (60 seeds, 7 rungs of the ladder each), so the error is the one a baseline can afford. What
it misses: a hop whose jump is no higher than the lattice ripple and that comes where the fit
has no place for it (4 of 1066 at 0.25 nm/s, 3 of them blunt tips); counting the teeth instead
does not help, since most are 1-3 pm.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

#: lattice spacing of the surface the atom is dragged over (Cu(111) nearest neighbour, nm in m)
SPACING_M = 0.255e-9
#: how far the tip gets ahead of the atom before the atom jumps after it, in spacings: the
#: atom hops when the pull has taken it past the middle of its site
LAG_SPACINGS = 0.6
#: how far one hop advances the atom along the path, in spacings (a hop goes to the neighbour
#: nearest the tip, within 30 deg of the path: 0.87-1.0 of a spacing, 0.94 on average); the three
#: values are tried
ADVANCES = (0.87, 0.96, 1.0)
#: where the atom is at the start of the path, along it, against the tip (m): a tip that was put on
#: an atom a slow frame placed is within about 0.1 nm of it
START_OFFSETS_M = (-0.1e-9, -0.05e-9, 0.0, 0.05e-9, 0.1e-9)
#: widths of the bump the tip sees, one standard deviation (m): the atom's own 0.25-0.35 nm
#: widened by the tip's blur (0.2 nm for a good tip, up to 0.6 nm for a blunt one)
WIDTHS_M = (0.25e-9, 0.30e-9, 0.37e-9, 0.45e-9, 0.55e-9)
#: how much better (chi-squared, noise 1) a hop count has to explain a trace than no hop for the
#: trace to count as showing hops; of 411 traces of atoms that did not hop, 3 reached 16
SIGNIFICANT = 16.0
#: how much a falling bump has to explain of a trace for the atom to count as left behind. Of
#: the drags that left the atom behind (0-11 hops, 0.25 nm/s, no tip change) the weakest fall
#: explained a trace by 2000 noises squared (the 1st percentile) and the median by 8000; of those
#: that carried it to the end, the lattice and the teeth that the fit has no place for reached 8
#: at the median, 46 at the 90th percentile and 129 at the 95th. Anything from 60 to 1000 reads
#: both right; 100 is the one that keeps a short drag as it was read: an atom that stayed in a
#: 0.6 or 1.0 nm drag reads as moved (92 and 63 %, the same as with 2 pm of noise per sample) because
#: so short a fall is still small, and a baseline can afford that error
FALL_CHI2 = 100.0
#: a step between two samples this many noises (of a difference) high, and at least this high
#: (pm), is not a bump or a tooth: a tip that changed, an atom taken up, a step edge of the
#: surface. It is taken out before the fit, or it would be read as a fall (or as the lack of one)
STEP_NOISES, STEP_MIN_PM = 8.0, 18.0
#: paths longer than this (m) are not drags of one atom over a few nanometres
MAX_PATH_M = 40e-9
#: shortest trace read: three samples, a drag of 0.1 nm. The last hop of an atom nearly home is a
#: drag of 0.2 nm, five samples, and it has to read as a hop (an atom that stayed reads as
#: moved there too: with so little of the path the fall cannot be told from a drifting
#: background, and a baseline can afford that error, not the other). And the least noise per
#: sample (pm) a trace is read with, whatever so few samples say of it: a sample of a drag at
#: 0.25-4 nm/s has 0.4-0.9 pm of the instrument's noise (a mean over its dwell: stmsim.physics.noise), and the
#: lattice and the teeth make the differences between samples spread by 0.5-1 pm even then
MIN_SAMPLES = 3
MIN_NOISE_PM = 1.0
#: a hop that comes after the atom has lagged far behind the tip (the pull took a while to make it
#: follow) puts the bump back under the tip in one step: Z jumps up out of a fall. Such a step is
#: not a tooth of the schedule the fit assumes. It counts as a hop when it is this many noises (of
#: a difference) high and at least this high (pm), stands out of the differences beside it (a ramp,
#: a bump coming under the tip, does not), and Z had fallen by this much (pm) over the
#: :data:`UP_STEP_LOOKBACK` samples before it (the tip was pulling away from the atom)
UP_STEP_SIGMAS, UP_STEP_MIN_PM, UP_STEP_DECLINE_PM, UP_STEP_LOOKBACK = 6.0, 5.0, 3.0, 3
#: the least noise (pm) a difference's spread is taken to have in that test
MIN_STEP_NOISE_PM = 0.3


def hop_steps(z_pm: np.ndarray) -> list[int]:
    """The samples of ``z_pm`` (pm) at which Z has jumped up the way an atom's hop after a long lag
    makes it (see :data:`UP_STEP_SIGMAS`): index ``k`` is a jump from sample ``k - 1`` to ``k``. A
    step that is not preceded by a fall (a tip change that made the bump taller, a step edge) is not
    one, nor is a step down (an atom taken up, a tip change). Needs a few more samples than the
    look-back, else nothing."""
    z_pm = np.asarray(z_pm, float)
    d = np.diff(z_pm)
    if d.size < UP_STEP_LOOKBACK + 2:
        return []
    sigma = max(1.4826 * float(np.median(np.abs(d - np.median(d)))), MIN_STEP_NOISE_PM)
    beside = 0.5 * (np.concatenate([[0.0], d[:-1]]) + np.concatenate([d[1:], [0.0]]))
    out: list[int] = []
    for k in np.flatnonzero((d > UP_STEP_MIN_PM) & (d - beside > UP_STEP_MIN_PM) & (d - beside > UP_STEP_SIGMAS * sigma)):
        if k >= UP_STEP_LOOKBACK and z_pm[k] - z_pm[k - UP_STEP_LOOKBACK] < -UP_STEP_DECLINE_PM \
                and (not out or k + 1 - out[-1] > 1):
            out.append(int(k) + 1)
    return out


def _noise_pm(z_pm: np.ndarray) -> float:
    """Noise per sample (pm) from the spread of the first differences: robust against the
    teeth, the fall and a step, which are a few of them."""
    d = np.diff(z_pm)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def _without_steps(z_pm: np.ndarray, sigma_pm: float) -> tuple[np.ndarray, int]:
    """``z_pm`` with its steps taken out (:data:`STEP_NOISES`), and how many there were: the
    levels before and after a step are joined, whatever it was."""
    d = np.diff(z_pm)
    big = np.abs(d) > max(STEP_NOISES * sigma_pm * np.sqrt(2.0), STEP_MIN_PM)
    if not big.any():
        return z_pm, 0
    d = np.where(big, 0.0, d)
    return np.concatenate([[z_pm[0]], z_pm[0] + np.cumsum(d)]), int(big.sum())


@lru_cache(maxsize=64)
def _bank(n: int, step_pm: int, spacing_pm: int):
    """The explanations of a trace of ``n`` samples ``step_pm`` apart: every bump a hop count,
    an advance, a start offset and a width describe, projected off the straight-line background
    (a constant and a slope: Z drifts), with what is needed to fit them all at once."""
    step, spacing = step_pm * 1e-12, spacing_pm * 1e-12
    s = np.arange(n) * step
    lag = LAG_SPACINGS * spacing
    n_max = int(s[-1] / (0.8 * spacing)) + 2
    hops, pos = [], []
    for adv in ADVANCES:
        for d0 in START_OFFSETS_M:
            for hop in range(n_max + 1):
                cur, p = -d0, np.full(n, -d0)
                for _ in range(hop):
                    idx = np.flatnonzero(s >= cur + lag)
                    if not idx.size:
                        break
                    cur += adv * spacing
                    p[idx[0]:] = cur
                hops.append(hop)
                pos.append(p)
    hops_a, pos_a = np.array(hops), np.array(pos)
    widths = np.array(WIDTHS_M)
    t = np.exp(-0.5 * ((s[None, None, :] - pos_a[:, None, :]) / widths[None, :, None]) ** 2)
    t = t.reshape(-1, n)
    k = np.arange(n, dtype=float)
    base = np.column_stack([np.ones(n), k - k.mean()])
    proj = np.eye(n) - base @ np.linalg.solve(base.T @ base, base.T)
    tp = t @ proj
    return {"hops": np.repeat(hops_a, len(widths)), "tp": tp, "tt": np.maximum((tp ** 2).sum(axis=1), 1e-30),
            "proj": proj, "n_max": n_max}


def fit(spectrum: dict, *, spacing_m: float = SPACING_M) -> dict | None:
    """Fit a move's trace with a bump the atom carries for n hops (see the module's docstring).

    Returns ``{"noise_pm", "chi2": [per hop count n: how much better the best bump with n hops
    explains the trace than a straight line does, in noise units squared], "best": the n of the
    best, "steps": steps taken out first}``, or ``None`` when the record is not a drag trace this
    can read (no feedback, too short, too long, no Z)."""
    z = spectrum.get("z_abs")
    dist = spectrum.get("dist")
    if z is None or dist is None or not spectrum.get("feedback", True):
        return None
    z = np.asarray(z, float) * 1e12
    dist = np.asarray(dist, float)
    n = z.size
    if n < MIN_SAMPLES or dist.size != n or not np.all(np.isfinite(z)) or dist[-1] > MAX_PATH_M or dist[-1] <= 0:
        return None
    step = float(dist[-1] / (n - 1))
    bank = _bank(n, int(round(step * 1e12)), int(round(spacing_m * 1e12)))
    sigma = max(_noise_pm(z), MIN_NOISE_PM)
    z, steps = _without_steps(z, sigma)
    zt = bank["proj"] @ z
    c = bank["tp"] @ zt
    gain = np.where(c > 0, c * c / bank["tt"], 0.0) / sigma ** 2          # a bump, never a pit
    per_n = np.array([gain[bank["hops"] == h].max() for h in range(bank["n_max"] + 1)])
    return {"noise_pm": sigma, "chi2": per_n, "best": int(np.argmax(per_n)), "n_max": bank["n_max"], "steps": steps}


def read_drag(spectrum: dict, *, spacing_m: float = SPACING_M) -> dict:
    """What a drag's trace shows, assuming the tip started over the atom: ``{"hops": int,
    "left_behind": bool, "carried": bool, "noise_pm": float, "steps": int}``. Steps in Z too
    big for a tooth or a bump (``steps`` of them: the tip changed, the atom was taken up, a step
    edge) are taken out first.

    ``left_behind``: the bump fell away within the trace, so the atom stayed where the fall
    puts it, ``hops`` hops from the start (0: it did not follow). ``carried``: no fall - the atom
    stayed under the tip for the whole path (or there was none to leave); ``hops`` is then the
    number of hops that takes (path length over the advance of one hop). Neither: the trace
    could not be read (``hops`` 0)."""
    out = {"hops": 0, "left_behind": False, "carried": False, "noise_pm": float("nan"), "steps": 0}
    f = fit(spectrum, spacing_m=spacing_m)
    if f is None:
        return out
    out["noise_pm"], out["steps"] = f["noise_pm"], f["steps"]
    chi2 = f["chi2"]
    path = float(np.asarray(spectrum["dist"], float)[-1])
    # an atom left at hop n leaves a fall that is over by about 1 nm further along
    n_behind = max(1, int((path - 1.0e-9) / (0.8 * spacing_m)))
    if float(chi2[: n_behind + 1].max()) < FALL_CHI2:
        out["carried"] = True
        out["hops"] = int(round(path / (np.mean(ADVANCES) * spacing_m)))
        return out
    out["left_behind"] = True
    best = int(np.argmax(chi2[: n_behind + 1]))
    out["hops"] = best if best >= 1 and chi2[best] - chi2[0] > SIGNIFICANT else 0
    if out["hops"] == 0:
        # no fall a count of hops explains: a hop after a long lag shows as a jump, not as a late fall
        out["hops"] = len(hop_steps(np.asarray(spectrum["z_abs"], float) * 1e12))
    return out


def count_hops(spectrum: dict, *, spacing_m: float = SPACING_M) -> int:
    """How many times an atom hopped under the tip on this drag, as its trace shows it
    (:func:`read_drag`): 0 when the atom was left where the tip started, the number of hops a
    fall further along the path puts it at, or - when the trace never falls - as many as kept the atom
    under the tip for the whole path."""
    return int(read_drag(spectrum, spacing_m=spacing_m)["hops"])
