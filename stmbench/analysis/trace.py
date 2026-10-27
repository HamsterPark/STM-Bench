"""Reading the trace of a move: did an atom hop under the tip, and how often?

An operator who drags an atom is not told whether it followed. The controller records the tip
height and the current while the tip moves (``move_tip`` keeps them like a spectrum, contract
C1), and what the atom did is read from them. In constant-current mode an atom that follows the
tip gives a sawtooth in Z with the period of the lattice (Bartels, Meyer & Rieder, PRL 79, 697
(1997)): the height under the tip falls a little as the tip moves on, and jumps when the atom
hops after it. This module reads that from a trace record. It reads nothing but the record: the
arrays ``z_abs`` and ``dist`` and the flag ``feedback`` (a Z loop that is off holds Z, and Z
shows nothing).

**What the sawtooth is worth.** One tooth is a few picometres (the atom's apparent height after
the tip's blur is 40-90 pm for a good tip, and the atom jumps when it lags the tip by 0.15 nm,
where the bump has fallen by about a tenth), against a Z noise of about 2 pm per sample. A single
hop is therefore hardly a signal, and a train of teeth is seen only as a faint ripple. What stands
out is the *fall*: a tip that leaves the atom behind goes from the top of its bump to the bare
surface, 20-90 pm over a stretch of the path as wide as the tip's blur, and *where* on the path that
fall happens says how far the atom came along: with no hop it falls from the start, with n hops it
holds its level for about n lattice spacings first. A trace that does not fall at all belongs to
an atom that stayed under the tip, or to a tip that never had an atom.

So the reading fits the trace with the one thing that is known about a drag, a bump that the atom
carries along for n hops and then leaves (:func:`fit`: a Gaussian of unknown width and height
on a sloping background, the atom jumping at the lag the lattice allows), for every n, and
asks which n the trace is best explained by and whether that beats the best no-hop explanation by
more than noise could (:data:`SIGNIFICANT`). **It assumes the tip started over the atom it
drags** (a drag that begins on bare surface has no fall either); the baselines place the tip
there by measuring the atom first.

**How well it reads** (2026-10-27, against the world's own events, read after each drag; seeds
0-119, which draw all five tips, the ladder 400 kohm x 0.8^k for 14 rungs, 4 nm drags, the tip put
0.05 nm (rms) from the atom as a slow frame would; a small version is
``tests/test_analysis_trace.py``). At 0.25 nm/s, the scripted baseline's speed (1506 drags in
which the atom was not picked up): of the 1066 in which an atom hopped, 1029 (96.5 %) read as
hopping - after 1 hop 78 %, 2-3 hops 94 %, 4 or more 99-100 % - and of the 440 in which it did
not, 4 (0.9 %). The count is right to one hop in 90-100 % of the drags in which the atom was
left behind (1-11 hops) and to three in 89 % of those in which it was carried the whole way
(about 17 hops: 16 on a lattice row, more with a blunt tip's wandering hops). At 1 nm/s 91 % and
0.9 %; at 4 nm/s, where an atom hops once or twice at most, 78 % and 1.0 %. An atom taken up by
the tip reads as moved in 84-96 %. Drags under two nanometres leave too little of the fall to
tell it from a drifting background: in drags of 0.3, 0.6, 1.0 and 1.5 nm an atom that stayed
reads as moved in 100, 92, 63 and 48 %, and one that hopped as hopping in 100, 98, 96 and 96 %
(2.5 nm: 0 % and 96 %), so the error is the one a baseline can afford.
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
#: how much a falling bump has to explain of a trace for the atom to count as left behind
FALL_CHI2 = 30.0
#: a step between two samples this many noises (of a difference) high, and at least this high
#: (pm), is not a bump or a tooth: a tip that changed, an atom taken up, a step edge of the
#: surface. It is taken out before the fit, or it would be read as a fall (or as the lack of one)
STEP_NOISES, STEP_MIN_PM = 8.0, 18.0
#: paths longer than this (m) are not drags of one atom over a few nanometres
MAX_PATH_M = 40e-9
#: shortest trace read (a drag of 0.25 nm: the last hop or two of an atom nearly home), and the
#: least noise per sample (pm) it is read with, whatever so few samples say of it (the
#: instrument's own is 2 pm)
MIN_SAMPLES = 6
MIN_NOISE_PM = 1.5


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
    return out


def count_hops(spectrum: dict, *, spacing_m: float = SPACING_M) -> int:
    """How many times an atom hopped under the tip on this drag, as its trace shows it
    (:func:`read_drag`): 0 when the atom was left where the tip started, the number of hops a
    fall further along the path puts it at, or - when the trace never falls - as many as kept the atom
    under the tip for the whole path."""
    return int(read_drag(spectrum, spacing_m=spacing_m)["hops"])
