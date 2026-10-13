"""Adatoms in a frame, and the lattice under them (``stmbench.analysis.adatoms``)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmbench.analysis.adatoms import bumps, lattice_angle, lattice_vector, locate


def _hollows(phi: float, a: float, n: int, rng) -> np.ndarray:
    """``n`` hollow sites of a lattice whose rows run along ``phi``, at least 3 nm apart within
    ±12 nm, on an arbitrary origin."""
    b = a * np.array([[math.cos(phi), math.cos(phi + math.pi / 3)],
                      [math.sin(phi), math.sin(phi + math.pi / 3)]])
    pts: list[np.ndarray] = []
    while len(pts) < n:
        p = b @ rng.integers(-60, 61, 2)
        if max(abs(p)) < 12 and all(math.hypot(*(p - q)) > 3 for q in pts):
            pts.append(p)
    return np.array(pts) + np.array([0.31, -1.17])


@pytest.mark.parametrize("seed", range(6))
def test_eight_atoms_give_the_lattice_orientation(seed):
    """Atoms on one sublattice of hollows: the vectors between them are lattice vectors. Eight
    atoms placed to 0.02 nm fix the rows to a fraction of a degree, with this lattice's
    0.255 nm against the ruler's 4 K literature value."""
    rng = np.random.default_rng(seed)
    phi = float(rng.uniform(0, math.pi / 3))
    pts = _hollows(phi, 0.255, 8, rng) + rng.normal(0.0, 0.02, (8, 2))
    got, rms, n = lattice_angle(pts)
    assert abs((math.degrees(got - phi) + 30.0) % 60.0 - 30.0) < 0.3
    assert rms < 0.05 and n == 28


@pytest.mark.parametrize("shear", [(0.0025, -0.0012), (0.0, 0.004), (-0.003, -0.003)])
def test_the_ruler_takes_out_the_shear_drift_puts_into_a_frame(shear):
    """Rows are taken one after the other, so drift the compensation left shifts each row's
    atoms by an amount growing with the row. A y-stretch of 0.4 % (0.16 nm across a 40 nm,
    80 s frame) put a plain orientation search 12° off (median of 8 lattices); with the shear
    searched too, eight atoms 40 nm apart give the rows to a tenth of a degree."""
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        phi = float(rng.uniform(0, math.pi / 3))
        b = 0.255 * np.array([[math.cos(phi), math.cos(phi + math.pi / 3)],
                              [math.sin(phi), math.sin(phi + math.pi / 3)]])
        pts: list[np.ndarray] = []
        while len(pts) < 8:
            q = b @ rng.integers(-90, 91, 2)
            if max(abs(q)) < 19 and all(math.hypot(*(q - o)) > 3 for o in pts):
                pts.append(q)
        arr = np.array(pts)
        arr = arr + arr[:, 1:2] * np.array([shear]) + rng.normal(0, 0.015, arr.shape)
        got, rms, _ = lattice_angle(arr)
        assert abs((math.degrees(got - phi) + 30.0) % 60.0 - 30.0) < 0.1, (shear, seed)
        assert rms < 0.04


def test_four_atoms_are_not_enough():
    """Four atoms leave three independent vectors, and another orientation fits them as well as
    the right one often enough: why the reference wants eight."""
    wrong = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        phi = float(rng.uniform(0, math.pi / 3))
        got, _, _ = lattice_angle(_hollows(phi, 0.255, 4, rng) + rng.normal(0.0, 0.02, (4, 2)))
        wrong += abs((math.degrees(got - phi) + 30.0) % 60.0 - 30.0) > 0.3
    assert wrong >= 2


def test_the_ruler_needs_three_atoms():
    with pytest.raises(ValueError):
        lattice_angle([(0.0, 0.0), (3.0, 1.0)])


def test_the_lattice_vector_nearest_a_displacement():
    """Rows along x: 4 nm falls between 15 and 16 spacings (3.825 and 4.08 nm), and the
    vectors half a spacing over are 0.22 nm off the axis."""
    assert lattice_vector((4.0, 0.0), 0.0, a_nm=0.255) == pytest.approx((4.08, 0.0), abs=1e-9)
    v = lattice_vector((4.0, 0.0), math.radians(30.0), a_nm=0.255)
    assert math.hypot(v[0] - 4.0, v[1]) < 0.255 / math.sqrt(3) + 1e-9      # within a cell's reach


def test_a_bump_beside_a_step_and_below_a_row_jump():
    """A 70 pm atom 3 nm from a 208 pm step, in a frame whose lower rows a tip change dropped by
    90 pm: the running-median background follows both, and only the atom is a compact bump
    (a levelling plane and a 3 nm high-pass lost atoms to exactly this, 2026-10-13)."""
    npx, n = 0.1, 80
    yy, xx = np.mgrid[0:n, 0:n] * npx
    z = np.where(xx > 6.0, 208e-12, 0.0)
    z = z + 70e-12 * np.exp(-((xx - 3.0) ** 2 + (yy - 4.0) ** 2) / (2 * 0.38 ** 2))
    z[64:, :] -= 90e-12
    z = z + np.random.default_rng(0).normal(0.0, 1e-12, z.shape)
    found, _ = bumps(z, npx)
    assert len(found) == 1
    col, row, h = found[0]
    assert abs(col * npx - 3.0) < 0.15 and abs(row * npx - 4.0) < 0.15 and h > 40e-12


def test_an_atom_beside_a_ring_of_atoms_is_found():
    """A 5 nm frame over an atom 1.5 nm from a corral, the ring's arc across the top of it: the
    atom is found. The noise level comes from neighbouring pixels, so the arc does not count as
    noise (on a rendered frame beside a P3-repair corral, a spread taken over the frame's values
    had put the threshold above a 56 pm atom)."""
    npx, n = 0.1, 50
    yy, xx = np.mgrid[0:n, 0:n] * npx
    x, y = xx - 2.5, 2.5 - yy                                  # frame centre at 0, row 0 at the top
    z = np.random.default_rng(1).normal(0.0, 2.7e-12, x.shape)
    for k in range(-4, 5):                                     # ring atoms 0.85 nm apart, 1.5 nm up
        z += 45e-12 * np.exp(-((x - 0.85 * k) ** 2 + (y - 1.5) ** 2) / (2 * 0.36 ** 2))
    z += 56e-12 * np.exp(-((x + 0.2) ** 2 + y ** 2) / (2 * 0.36 ** 2))
    found, _ = bumps(z, npx, compact_frac=0.9)
    assert any(abs(c * npx - 2.5 + 0.2) < 0.25 and abs(2.5 - r * npx) < 0.25 for c, r, h in found)


def test_a_slow_frame_places_the_atom_and_a_fast_one_shifts_it_along_the_scan():
    """On a rendered frame (seed 28 arrives with a good tip): at 20 nm/s the atom is where the
    sample has it; at 280 nm/s the Z loop's lag puts it ~0.15 nm further along the fast axis."""
    from stmbench.instrument import Instrument
    from stmbench.trackB import claims as C

    inst = Instrument("P4_atom_positioning_cu111", 28)
    w = inst.world
    reg = w.surface.site.adatoms
    atom = reg.atoms[reg.target["atom_id"]]
    ax, ay = (v * 1e9 for v in reg.site_xy(atom.i, atom.j))
    off = {}
    for name, size, px, line_s in (("slow", 5.0, 50, 0.25), ("fast", 28.0, 280, 0.1)):
        ox, oy, _ = w.drift_offset(w.clock.sim())
        cx, cy = ax - ox * 1e9, ay - oy * 1e9
        f = inst.frame(inst.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=size, pixels=px,
                                line_time_s=line_s)["frame"])
        x, y, _ = min(locate(f), key=lambda a: math.hypot(a[0] - cx, a[1] - cy))
        ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
        sx, sy = C.to_sample_frame(ev, x, y)
        off[name] = (sx - ax, sy - ay)
    assert math.hypot(*off["slow"]) < 0.05, off
    assert 0.08 < off["fast"][0] < 0.25 and abs(off["fast"][1]) < 0.05, off
