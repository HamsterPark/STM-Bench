"""P3 — Crommie/Lutz/Eigler 1993: the confined states inside a quantum corral.

Image the ring, find its centre from where the atoms are, take a spectrum there, read the two
lowest resolvable resonances off it. Imaging stays at a high junction resistance throughout:
at a manipulation resistance the survey would take the corral apart before it is measured.

The repair variant arrives with a few adjacent sites empty and the same number of spare atoms
outside the gap. The baseline works out where the missing atoms belong from the ones that are
there — the ring's angular spacing gives the slot width, and a run of empty slots is a gap —
then supplies each spare and its slot to ``MoveAtomTo``. It never learns the pull threshold; the
composite's own retry lowers the resistance once when an atom does not follow.
"""
from __future__ import annotations

import json
import math

from ._clusters import bright_blobs
from ._drift import null_the_drift
from ._runner import Baseline

# an adatom is 70 pm on a surface whose steps are 208: without a plane subtraction and
# a threshold near the noise, the segmentation never sees it. Measured 2026-09-07: the
# defaults find nothing but frame-corner artefacts, these find every atom to 0.1 nm.
CLUSTER_THRESHOLD_MAD = 2.0

SURVEY_NM = 70.0
#: cheap registration frames for the drift measurement — they only have to correlate
DRIFT_NM, DRIFT_PX = 40.0, 128
#: wide enough to cover the whole area the task points at (Surface.CLEAR_AREA_REACH_M)
#: 0.18 nm/px: an adatom is 0.7 nm across, so it has to be several pixels wide
SURVEY_PX = 384
V_LO, V_HI = -0.6, 0.4
RING_TOL = 0.25              # |r − R| / R inside which a cluster counts as a ring atom
SPARE_MIN = 1.20             # r / R beyond which a cluster counts as a spare


def run_p3(host, scenario, *, seed: int = 0, extra: dict | None = None) -> dict:
    b = Baseline(host, scenario, seed)
    extra = dict(extra or {})
    repair = any(c["id"] == "ring_occupancy" for c in scenario.claims)

    ring = _locate_corral(b, float(extra.get("survey_nm", SURVEY_NM)))
    if ring is None:
        return b.finish(note="no ring of atoms found in the survey")
    cx, cy, radius, atoms = ring

    # Null the drift before anything long. Over this budget the sample walks further than the
    # corral is wide, and the repair variant drags atoms onto sites that have to still be there
    # when the spectrum is taken. A ring of adatoms is rich 2-D structure, so correlation suits.
    drift = null_the_drift(b, (cx, cy), frame_nm=DRIFT_NM, pixels=DRIFT_PX)
    filled = 0

    if repair:
        filled = _repair_ring(b, cx, cy, radius, atoms)
        # a close frame of the finished ring: the evidence that the repair happened, and the
        # picture the next step's centre is read off
        b.run("ScanAt", {"center_x_m": cx, "center_y_m": cy,
                         "size_m": (2.4 * radius + 4e-9)}, optional=True)
        b.run("SaveScan", {}, optional=True)
        again = _ring_from_latest(b, cx, cy)
        if again is not None:
            cx, cy, radius, atoms = again

    b.run("ConfigureSTS", {"start_v": V_LO, "end_v": V_HI, "num_points": 200, "z_offset_m": 0.0},
          optional=True)
    res = b.run("SpectroscopyAtPositions", {
        "positions": json.dumps([{"x_m": cx, "y_m": cy, "label": "corral centre"}]),
        "expected_coord_epoch": _coord_epoch(), "run_tag": "corral", "assess": False,
    }, optional=True)
    paths = b.data(res, "dat_paths") or [str(p) for p in b.dats("corral_p*.dat")]
    if not paths:
        return b.finish(note="no spectrum at the corral centre", drift=drift,
                        centre_nm=[cx * 1e9, cy * 1e9], atoms_placed=filled)

    peaks_res = b.run("FindSpectralPeaks", {"dat_path": str(paths[-1]),
                                            "bias_min_v": V_LO + 0.02, "bias_max_v": 0.3},
                      optional=True)
    energies = b.data(peaks_res, "energies_mev") or []
    if len(energies) >= 1:
        b.report("peak_lo_mev", float(energies[0]), "meV")
    if len(energies) >= 2:
        b.report("peak_hi_mev", float(energies[1]), "meV")
    return b.finish(drift=drift, centre_nm=[cx * 1e9, cy * 1e9], radius_nm=radius * 1e9,
                    n_ring_atoms=len(atoms), atoms_placed=filled,
                    peaks_mev=[round(e, 1) for e in energies[:4]])


# ── repair ──

def _repair_ring(b: Baseline, cx: float, cy: float, radius: float, atoms: list) -> int:
    """Move every spare atom into an empty ring slot. Returns how many were placed."""
    placed = 0
    for _ in range(8):                              # one atom per pass, re-imaging between
        clusters = _clusters_of_latest(b)
        if clusters is None:
            break
        slots = _empty_slots(clusters, cx, cy, radius)
        spares = _spares(clusters, cx, cy, radius)
        if not slots or not spares:
            break
        # the closest spare/slot pair, so the drag is short and stays clear of the ring
        sx, sy, tx, ty = min(
            ((s[0], s[1], t[0], t[1]) for s in spares for t in slots),
            key=lambda q: math.hypot(q[2] - q[0], q[3] - q[1]))
        move = b.run("MoveAtomTo", {"atom_x_m": sx, "atom_y_m": sy,
                                    "target_x_m": tx, "target_y_m": ty,
                                    "verify": True}, optional=True)
        if b.data(move, "moved") is not True:
            break                                   # the atom did not follow; stop rather
        placed += 1                                 # than push the tip around at low R
    return placed


def _empty_slots(clusters: list, cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    """Where the missing ring atoms belong, from the angular spacing of the ones present."""
    ring = [c for c in clusters
            if abs(math.hypot(c[0] - cx, c[1] - cy) - radius) <= RING_TOL * radius]
    if len(ring) < 8:
        return []
    ang = sorted(math.atan2(y - cy, x - cx) for x, y in ring)
    steps = [ang[i + 1] - ang[i] for i in range(len(ang) - 1)]
    steps.append(ang[0] + 2 * math.pi - ang[-1])
    step = sorted(steps)[len(steps) // 2]           # the median spacing is one slot
    if step <= 0:
        return []
    out: list[tuple[float, float]] = []
    for i, gap in enumerate(steps):
        missing = int(round(gap / step)) - 1
        if missing <= 0:
            continue
        a0 = ang[i]
        for k in range(1, missing + 1):
            a = a0 + gap * k / (missing + 1)
            out.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return out


def _spares(clusters: list, cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    return [c for c in clusters if math.hypot(c[0] - cx, c[1] - cy) >= SPARE_MIN * radius]


# ── imaging ──

def _locate_corral(b: Baseline, survey_nm: float):
    """Centre, radius and atoms of the ring, from one survey frame."""
    b.run("ScanAt", {"center_x_m": 0.0, "center_y_m": 0.0, "size_m": survey_nm * 1e-9,
                     "pixels": SURVEY_PX}, optional=True)
    b.run("SaveScan", {}, optional=True)
    return _ring_from_latest(b, 0.0, 0.0)


def _ring_from_latest(b: Baseline, cx0: float, cy0: float):
    clusters = _clusters_of_latest(b)
    if not clusters or len(clusters) < 8:
        return None
    return _fit_ring(clusters, cx0, cy0)


def _fit_ring(clusters: list, cx0: float, cy0: float):
    """A ring of atoms, told apart from a scatter of adsorbates by how tight its radii are.

    Spares sit outside the ring, so the mean of every cluster is not the centre. The centre is
    re-estimated from the clusters that agree on a radius, twice, which is enough to shake the
    spares off.
    """
    cx = sum(c[0] for c in clusters) / len(clusters)
    cy = sum(c[1] for c in clusters) / len(clusters)
    ring = clusters
    for _ in range(3):
        radii = sorted(math.hypot(c[0] - cx, c[1] - cy) for c in ring)
        radius = radii[len(radii) // 2]
        if radius <= 0:
            return None
        ring = [c for c in clusters
                if abs(math.hypot(c[0] - cx, c[1] - cy) - radius) <= RING_TOL * radius]
        if len(ring) < 8:
            return None
        cx = sum(c[0] for c in ring) / len(ring)
        cy = sum(c[1] for c in ring) / len(ring)
    radii = [math.hypot(c[0] - cx, c[1] - cy) for c in ring]
    radius = sum(radii) / len(radii)
    if radius <= 0 or (max(radii) - min(radii)) > 0.5 * radius:
        return None                       # not a ring: a scatter of adsorbates
    if math.hypot(cx - cx0, cy - cy0) > 80e-9:
        return None                       # nowhere near where the task said to look
    return cx, cy, radius, ring


def _clusters_of_latest(b: Baseline) -> list[tuple[float, float]] | None:
    frame = b.latest_frame()
    if frame is None:
        return None
    # compact bumps above the LOCAL background (see ``_clusters``): a plane-levelled MAD
# threshold on a survey that crosses a step found nothing of the ring (2026-09-13 calibration gate)
    return list(bright_blobs(frame))


def _coord_epoch() -> int:
    try:
        from mast.core.coord_epoch import read_current_epoch

        return int(read_current_epoch() or 0)
    except Exception:  # noqa: BLE001
        return 0
