"""Chance audit: how much of a paper a fixed answer sheet reproduces without measuring.

The hidden truth is drawn per seed inside ranges that are public (the scenario files ship
with the package), so the natural attack on a reported-value claim is not the literature
value but a number near the middle of the drawn range. This module draws the truth for many
seeds and searches, with the judge's own tolerance helpers, for the constant answer that
passes the most seeds: per claim, and for the paper as a whole.

Claims fall into three groups:

* ``knowledge`` — the claim carries ``knowledge: allowed``: a material constant (the
  herringbone period, the Cu(111) band bottom and effective mass). Knowing it is part of the
  skill, so it is reported here for the record and takes no part in the gate.
* ``measure`` — every other reported-value claim. The joint chance of these claims is the
  number a scenario is held to; the gate is applied by the release checks, not here.
* ``state`` — position and constraint claims, judged on where the atoms actually are.
  No report can pass them without the manipulation having happened, so they are not audited.

Evidence rules are treated as satisfied: the evidence a claim needs is cheap to acquire,
and the audit asks what the *value* check alone lets through.
"""
from __future__ import annotations

import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import claims as C

#: where a fixed answer sheet puts its position-bearing angle claims (scan-frame nm), in claim
#: order: far enough apart that two of them can land in different rotational domains
DEFAULT_SPOTS_NM: tuple[tuple[float, float], ...] = ((0.0, 0.0), (-90.0, 0.0), (90.0, 0.0),
                                                      (0.0, 90.0), (0.0, -90.0))
SCALAR_GRID = 401
ANGLE_STEP_DEG = 0.5


def claim_policy(claim: dict) -> str:
    """``knowledge`` | ``measure`` | ``state`` (see the module docstring)."""
    if claim.get("kind") in ("position", "constraint"):
        return "state"
    return "knowledge" if claim.get("knowledge") == "allowed" else "measure"


# ── truth samples ───────────────────────────────────────────────────────────
def _angle_spots(claims: Sequence[dict]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    k = 0
    for c in claims:
        if c.get("kind") == "angle" and c.get("truth") == "orientation_at":
            out[str(c["id"])] = DEFAULT_SPOTS_NM[min(k, len(DEFAULT_SPOTS_NM) - 1)]
            k += 1
    return out


def _sample_seed(args) -> dict:
    """The truths one seed's claims are judged against (picklable for a process pool)."""
    path, seed, work = args
    from stmsim.scenario import Scenario

    sc = Scenario.load(path)
    w = sc.build_world(int(seed), session_dir=Path(work) / f"s{seed}")
    truth = w.truth()
    spots = _angle_spots(sc.claims)
    out: dict = {}
    for c in sc.claims:
        cid = str(c["id"])
        if claim_policy(c) == "state":
            continue
        if c.get("truth") == "orientation_at":
            x, y = spots[cid]
            info = w.surface.orientation_at(x * 1e-9, y * 1e-9) or {}
            stripe = info.get("stripe_deg")
            out[cid] = None if stripe is None else {
                "stripe_deg": float(stripe), "accept": [float(stripe), *map(float, info.get("arms_deg") or [])]}
            continue
        val = C.truth_at_path(truth, str(c["truth"]))
        if c["kind"] == "peak_in_list":
            out[cid] = [float(v) for v in (val or [])]
        else:
            out[cid] = None if val is None else float(val)
    return out


def sample_truths(scenario_path: str | Path, seeds: Iterable[int], *, jobs: int = 1,
                  work_dir: str | Path | None = None) -> list[dict]:
    """One dict of claim truths per seed, in seed order."""
    seeds = [int(s) for s in seeds]
    with tempfile.TemporaryDirectory(prefix="stmbench_audit_") as tmp:
        work = Path(work_dir or tmp)
        tasks = [(str(scenario_path), s, str(work)) for s in seeds]
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                return list(ex.map(_sample_seed, tasks))
        return [_sample_seed(t) for t in tasks]


# ── pass matrices: candidate answers × seeds ────────────────────────────────
@dataclass
class ClaimGrid:
    claim: dict
    candidates: np.ndarray                    # the constant answers tried
    passes: np.ndarray                        # bool, (n_candidates, n_seeds)
    ranks: np.ndarray | None = None           # peak_in_list: matched rank, -1 = none
    truths: list = field(default_factory=list)
    cache: dict = field(default_factory=dict, repr=False)

    @property
    def id(self) -> str:
        return str(self.claim["id"])


def _scalar_grid(claim: dict, truths: list) -> ClaimGrid:
    t = np.asarray([np.nan if v is None else v for v in truths], float)
    finite = t[np.isfinite(t)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    span = (hi - lo) or abs(hi) or 1.0
    cand = np.linspace(lo - 0.3 * span, hi + 0.3 * span, SCALAR_GRID)
    tol = claim.get("tol") or {}
    lim = np.maximum(float(tol.get("abs", 0.0) or 0.0), float(tol.get("rel", 0.0) or 0.0) * np.abs(t))
    lim = np.where(lim > 0, lim, 1e-12)
    passes = np.abs(cand[:, None] - t[None, :]) <= lim[None, :]
    passes &= np.isfinite(t)[None, :]
    return ClaimGrid(claim, cand, passes, truths=list(truths))


def _angle_grid(claim: dict, truths: list) -> ClaimGrid:
    cand = np.arange(0.0, 180.0, ANGLE_STEP_DEG)
    tol = claim.get("tol") or {}
    passes = np.zeros((cand.size, len(truths)), bool)
    for j, info in enumerate(truths):
        if not info:
            continue
        for a in info["accept"]:
            passes[:, j] |= np.asarray([C.within_angle(float(c), a, tol) for c in cand])
    return ClaimGrid(claim, cand, passes, truths=list(truths))


def _peak_grid(claim: dict, truths: list) -> ClaimGrid:
    tol = claim.get("tol") or {}
    rank_max = int(claim.get("rank_max", 3))
    abs_tol = float(tol.get("abs", 0.0) or 0.0) or 10.0
    firsts = [p[:rank_max] for p in truths if p]
    allp = np.concatenate([np.asarray(p, float) for p in firsts]) if firsts else np.zeros(1)
    cand = np.arange(allp.min() - 3 * abs_tol, allp.max() + 3 * abs_tol, abs_tol / 10.0)
    ranks = np.full((cand.size, len(truths)), -1, int)
    for j, peaks in enumerate(truths):
        for r, p in reversed(list(enumerate((peaks or [])[:rank_max]))):
            hit = np.asarray([C.within_scalar(float(c), float(p), tol) for c in cand])
            ranks[hit, j] = r          # reversed: the lowest matching rank wins, as in the judge
    return ClaimGrid(claim, cand, ranks >= 0, ranks=ranks, truths=list(truths))


def claim_grid(claim: dict, truths: list) -> ClaimGrid:
    if claim.get("truth") == "orientation_at" or claim.get("kind") == "angle":
        return _angle_grid(claim, truths)
    if claim.get("kind") == "peak_in_list":
        return _peak_grid(claim, truths)
    return _scalar_grid(claim, truths)


# ── joint search ────────────────────────────────────────────────────────────
def _truths_apart(g: ClaimGrid, peer: ClaimGrid, sep: float) -> np.ndarray:
    """Per seed: are the truths at the two claims' places at least ``sep`` degrees apart?
    (the judge's ``distinct_from`` for angles compares truths, not answers)"""
    key = ("apart", peer.id, sep)
    if key not in g.cache:
        g.cache[key] = np.asarray(
            [bool(a and b and C.angle_diff(a["stripe_deg"], b["stripe_deg"]) >= sep)
             for a, b in zip(g.truths, peer.truths)])
    return g.cache[key]


def joint_pass(grids: Sequence[ClaimGrid], choice: Sequence[int]) -> np.ndarray:
    """Per seed: does the answer sheet ``choice`` (one candidate index per grid) pass every
    claim, including the judge's pairwise rules (``distinct_from``, ``greater_than``)?"""
    ok = np.ones(grids[0].passes.shape[1], bool)
    by_id = {g.id: (g, i) for g, i in zip(grids, choice)}
    for g, i in zip(grids, choice):
        ok &= g.passes[i]
        other = g.claim.get("distinct_from")
        if other and str(other.get("claim")) in by_id:
            peer, k = by_id[str(other["claim"])]
            if g.ranks is not None and peer.ranks is not None:
                ok &= g.ranks[i] != peer.ranks[k]
            elif g.truths and peer.truths:
                ok &= _truths_apart(g, peer, float(other.get("min_sep_deg", other.get("min_sep", 0.0))))
        gt = g.claim.get("greater_than")
        if gt and str(gt) in by_id:
            peer, k = by_id[str(gt)]
            if not float(g.candidates[i]) > float(peer.candidates[k]):
                ok &= False
    return ok


def _rules_between(g: ClaimGrid, peer: ClaimGrid) -> dict:
    """The judge's pairwise rules linking two claims, in either direction."""
    rules = {"ranks_distinct": False, "apart": None, "order": 0}
    for a, b, sign in ((g, peer, 1), (peer, g, -1)):
        other = a.claim.get("distinct_from")
        if other and str(other.get("claim")) == b.id:
            if a.ranks is not None and b.ranks is not None:
                rules["ranks_distinct"] = True
            elif a.truths and b.truths:
                rules["apart"] = _truths_apart(a, b, float(other.get("min_sep_deg", other.get("min_sep", 0.0))))
        if a.claim.get("greater_than") is not None and str(a.claim["greater_than"]) == b.id:
            rules["order"] = sign            # +1: g's answer must exceed peer's; -1: the reverse
    return rules


def _pair_counts(g: ClaimGrid, peer: ClaimGrid) -> np.ndarray:
    """Seeds passed by every (candidate of g, candidate of peer) pair, pairwise rules included."""
    rules = _rules_between(g, peer)
    if rules["ranks_distinct"]:
        n_rank = int(max(g.ranks.max(), peer.ranks.max())) + 1
        counts = np.zeros((g.candidates.size, peer.candidates.size), np.int64)
        for r1 in range(n_rank):
            a = (g.ranks == r1).astype(np.int64)
            for r2 in range(n_rank):
                if r1 != r2:
                    counts += a @ (peer.ranks == r2).astype(np.int64).T
    else:
        a = g.passes.astype(np.int64)
        if rules["apart"] is not None:
            a = a * rules["apart"][None, :].astype(np.int64)
        counts = a @ peer.passes.astype(np.int64).T
    if rules["order"] > 0:
        counts = counts * (g.candidates[:, None] > peer.candidates[None, :])
    elif rules["order"] < 0:
        counts = counts * (g.candidates[:, None] < peer.candidates[None, :])
    return counts


def _best_step(grids: Sequence[ClaimGrid], choice: Sequence[int], c: int) -> tuple[int, float]:
    """With every other claim's answer fixed, the best answer for claim ``c`` (exact)."""
    g = grids[c]
    others = [k for k in range(len(grids)) if k != c]
    mask = joint_pass([grids[k] for k in others], [choice[k] for k in others])
    ok = g.passes & mask[None, :]
    for k in others:
        peer, j = grids[k], choice[k]
        rules = _rules_between(g, peer)
        if rules["ranks_distinct"]:
            ok &= g.ranks != peer.ranks[j][None, :]
        if rules["apart"] is not None:
            ok &= rules["apart"][None, :]
        if rules["order"] > 0:
            ok &= (g.candidates > peer.candidates[j])[:, None]
        elif rules["order"] < 0:
            ok &= (g.candidates < peer.candidates[j])[:, None]
    frac = ok.mean(axis=1)
    i = int(np.argmax(frac))
    return i, float(frac[i])


def _linked(grids: Sequence[ClaimGrid]) -> bool:
    """Whether any two of the claims are tied by a pairwise rule of the judge."""
    for a in range(len(grids)):
        for b in range(a + 1, len(grids)):
            r = _rules_between(grids[a], grids[b])
            if r["ranks_distinct"] or r["apart"] is not None or r["order"]:
                return True
    return False


def _best_triple(grids: Sequence[ClaimGrid]) -> tuple[list[int], float]:
    """Exact best sheet for three claims with no pairwise rule between them: for every answer
    to the first claim, the seed counts of all (second, third) pairs in one matrix product.
    Answers whose own pass count cannot beat the best so far are skipped."""
    g0, g1, g2 = grids
    n = g0.passes.shape[1]
    p1 = g1.passes.astype(np.float32)
    p2t = g2.passes.astype(np.float32).T
    marg = g0.passes.sum(axis=1)
    best_val, best = -1.0, [0, 0, 0]
    for i in np.argsort(-marg):
        if marg[i] <= best_val:
            break
        m = (p1 * g0.passes[i][None, :]) @ p2t
        j, k = np.unravel_index(int(np.argmax(m)), m.shape)
        if m[j, k] > best_val:
            best_val, best = float(m[j, k]), [int(i), int(j), int(k)]
    return best, best_val / n


def best_joint(grids: Sequence[ClaimGrid], *, sweeps: int = 8, restarts: int = 5) -> tuple[list[int], float]:
    """The fixed answer sheet that passes the most seeds on every claim at once.

    One or two claims: exact (two claims through a seed-count matrix). Three claims with no
    pairwise rule between them: exact (:func:`_best_triple`). Otherwise coordinate ascent, each
    step exact, from the per-claim optima and from the next-best candidates of each claim; the
    result is then a lower bound on the best sheet (on P5, before the exact search, ascent
    reported 4.0 % at a looser tolerance and 4.3 % at a tighter one)."""
    if not grids:
        return [], 0.0
    n = grids[0].passes.shape[1]
    if len(grids) == 1:
        m = grids[0].passes.mean(axis=1)
        i = int(np.argmax(m))
        return [i], float(m[i])
    if len(grids) == 2:
        counts = _pair_counts(grids[0], grids[1])
        i, j = np.unravel_index(int(np.argmax(counts)), counts.shape)
        return [int(i), int(j)], float(counts[i, j]) / n
    if len(grids) == 3 and not _linked(grids):
        return _best_triple(grids)
    marg = [g.passes.mean(axis=1) for g in grids]
    base = [int(np.argmax(m)) for m in marg]
    starts = [base]
    for c, m in enumerate(marg):
        for i in np.argsort(-m)[1:restarts]:
            starts.append(base[:c] + [int(i)] + base[c + 1:])
    best_choice, best_val = list(base), float(joint_pass(grids, base).mean())
    for start in starts:
        choice = list(start)
        val = float(joint_pass(grids, choice).mean())
        for _ in range(sweeps):
            improved = False
            for c in range(len(grids)):
                i, v = _best_step(grids, choice, c)
                if v > val + 1e-12:
                    choice[c], val, improved = i, v, True
            if not improved:
                break
        if val > best_val:
            best_choice, best_val = list(choice), val
    return best_choice, best_val


# ── the audit ───────────────────────────────────────────────────────────────
def audit_scenario(scenario_path: str | Path, n_seeds: int = 200, *, jobs: int = 1,
                   first_seed: int = 0) -> dict:
    """Per-claim and joint chance for one paper scenario (a JSON-ready dict)."""
    from stmsim.scenario import Scenario

    sc = Scenario.load(scenario_path)
    seeds = list(range(int(first_seed), int(first_seed) + int(n_seeds)))
    samples = sample_truths(scenario_path, seeds, jobs=jobs)
    grids: dict[str, ClaimGrid] = {}
    rows = []
    for c in sc.claims:
        cid, pol = str(c["id"]), claim_policy(c)
        row = {"id": cid, "kind": c["kind"], "policy": pol}
        if pol != "state":
            g = claim_grid(c, [s.get(cid) for s in samples])
            grids[cid] = g
            m = g.passes.mean(axis=1)
            i = int(np.argmax(m))
            row.update(best_constant=float(g.candidates[i]), pass_frac=float(m[i]))
            if c["kind"] in ("scalar",):
                t = np.asarray([v for v in g.truths if v is not None], float)
                row.update(truth_min=float(t.min()), truth_max=float(t.max()))
        rows.append(row)
    out = {"scenario": sc.id, "n_seeds": len(seeds), "first_seed": int(first_seed), "claims": rows}
    for label, keep in (("measure", ("measure",)), ("all", ("measure", "knowledge"))):
        sel = [grids[str(c["id"])] for c in sc.claims if claim_policy(c) in keep]
        if sel:
            choice, frac = best_joint(sel)
            out[f"joint_{label}"] = {"pass_frac": frac,
                                     "answer": {g.id: float(g.candidates[i]) for g, i in zip(sel, choice)}}
        else:
            out[f"joint_{label}"] = None
    # what a fixed sheet banks on average: the per-claim optima, state claims at zero
    out["guess_claim_fraction"] = float(np.mean([r.get("pass_frac", 0.0) for r in rows]))
    return out


def format_report(res: dict) -> str:
    lines = [f"{res['scenario']}  ({res['n_seeds']} seeds from {res['first_seed']})"]
    lines.append(f"  {'claim':<28} {'kind':<13} {'policy':<10} {'best constant':>15} {'pass':>7}")
    for r in res["claims"]:
        if r["policy"] == "state":
            lines.append(f"  {r['id']:<28} {r['kind']:<13} {r['policy']:<10} {'—':>15} {'—':>7}")
            continue
        lines.append(f"  {r['id']:<28} {r['kind']:<13} {r['policy']:<10} "
                     f"{r['best_constant']:>15.4g} {100 * r['pass_frac']:>6.1f}%")
    for label in ("measure", "all"):
        j = res.get(f"joint_{label}")
        txt = "— (no such claims)" if j is None else f"{100 * j['pass_frac']:.1f}%"
        lines.append(f"  joint, {label} claims: {txt}")
    lines.append(f"  expected claim fraction of a fixed sheet: {res['guess_claim_fraction']:.2f}")
    return "\n".join(lines)

