"""Spectra and dI/dV maps see every apex of the tip, each by its share of the current
(DESIGN §9.3, FIDELITY's Tip table).

Until 2026-11-24 both took the sample LDOS under the tip's centre only, while images saw every
apex. A tip with one apex at its centre (sharp, blunt or dirty) must still see exactly what it saw
then, and no random stream may gain or lose a draw."""
from __future__ import annotations

import numpy as np
import pytest

from stmbench.instrument import Instrument
from stmsim.physics.junction import LDOSTemplate, ParallelLDOS, iv_curve
from stmsim.physics.tip import Apex
from stmsim.physics.world import World

P3 = "P3_corral_cu111"
P4 = "P4_atom_positioning_cu111"
SEED = 0
#: a second apex 1.26 nm off and 30 pm shorter, as a double tip arrives (Tip.apply_condition)
SECOND = Apex(1.2e-9, 0.4e-9, -0.03e-9, 0.8)


def _states(w) -> dict:
    """The state of every random generator the world, its tip and its surface own."""
    gens = {"world": w.rng, "tip": w.tip.rng, "noise": w.noise.rng, "df": w.df_rng,
            "outscan": w.rng_outscan, "recovery": w.rng_recovery, "aniso": w.rng_aniso,
            "site": w.surface.site.rng}
    reg = w.surface.site.adatoms
    if reg is not None:
        gens |= {"layout": reg.rng_layout, "dynamics": reg.rng_dyn, "traits": reg.rng_traits}
    return {k: g.bit_generator.state for k, g in gens.items()}


def _as_before(mp: pytest.MonkeyPatch) -> None:
    """The code as it was: the LDOS under the tip's centre, whatever the tip."""
    mp.setattr(World, "ldos_under_tip", lambda self, sx, sy: self.surface.ldos_at(sx, sy))
    mp.setattr(World, "ldos_map_under_tip",
               lambda self, x, y, e_ev, kappa_m=None: self.surface.ldos_map(x, y, e_ev))


def _corral_centre_scan_nm(w) -> tuple[float, float]:
    """The corral's centre in scan-frame nm now (privileged: reads the registry)."""
    cx, cy = w.surface.site.adatoms.ring["centre_nm"]
    ox, oy, _ = w.drift_offset(w.clock.sim())
    return cx - ox * 1e9, cy - oy * 1e9


def _quiet(tip) -> None:
    """No spontaneous change during the test, so both runs keep the tip they were given."""
    tip.lambda_per_s = 0.0
    tip.metastable = False


def _sharp(tip):
    tip.apexes = [Apex(0.0, 0.0, 0.0, 1.0)]


def _blunt(tip):
    _sharp(tip)
    tip.apply_condition("blunt", 0.6)


def _dirty(tip):
    _sharp(tip)
    tip.apply_condition("dirty", 0.7)


def _double(tip):
    tip.apexes = [Apex(0.0, 0.0, 0.0, 1.0), Apex(*SECOND.as_tuple())]


def _session(tmp_path, tag: str, make_tip, before: bool, action: bool = True) -> dict:
    """A dI/dV map over the corral and a spectrum at its centre, on the given tip, with the code
    as it is or as it was. ``action=False`` takes the spectrum with ``World.sts_curve`` alone,
    without the sweep's tip hazards (which read its current and so may differ with it)."""
    with pytest.MonkeyPatch.context() as mp:
        if before:
            _as_before(mp)
        inst = Instrument(P3, SEED, tmp_path / f"{tag}_{'before' if before else 'now'}")
        w = inst.world
        make_tip(w.tip)
        _quiet(w.tip)
        x, y = _corral_centre_scan_nm(w)
        inst.act("lockin", on=True, amplitude_mv=10.0)
        r = inst.act("scan", center_x_nm=x, center_y_nm=y, size_nm=10.0, pixels=32)
        maps = {ch: inst.frame(r["frame"], ch)["array"] for ch in ("z", "current", "didv")}
        x, y = _corral_centre_scan_nm(w)
        w.tip_x, w.tip_y = x * 1e-9, y * 1e-9
        if action:
            s = inst.act("sts", v_start=-0.5, v_end=0.1, points=61, backward=False)
            sp = inst.spectrum(s["spectrum"])
        else:
            sp = w.sts_curve(-0.5, 0.1, 61)
        return {"maps": maps, "i": np.array(sp["i"]), "didv": np.array(sp["didv"]),
                "digest": inst.digest(), "states": _states(w)}


@pytest.mark.parametrize("make_tip", [_sharp, _blunt, _dirty], ids=["sharp", "blunt", "dirty"])
def test_one_apex_at_the_centre_sees_what_it_saw_before_bit_for_bit(tmp_path, make_tip):
    """However blunt or dirty, a tip with one apex at its centre takes the old path: the same
    map, the same spectrum, the same random streams, to the last bit."""
    old = _session(tmp_path, make_tip.__name__, make_tip, before=True)
    new = _session(tmp_path, make_tip.__name__, make_tip, before=False)
    for ch in ("z", "current", "didv"):
        assert np.array_equal(old["maps"][ch], new["maps"][ch]), ch
    assert np.array_equal(old["i"], new["i"]) and np.array_equal(old["didv"], new["didv"])
    assert old["digest"] == new["digest"]
    assert old["states"] == new["states"]


def test_a_double_tip_changes_the_spectrum_and_the_map_and_nothing_else(tmp_path):
    """The second apex now reaches the spectrum and the dI/dV map (the omission: it did not);
    the image, which always saw it, and every random stream stay as they were."""
    old = _session(tmp_path, "double", _double, before=True, action=False)
    new = _session(tmp_path, "double", _double, before=False, action=False)
    assert np.array_equal(old["maps"]["z"], new["maps"]["z"])
    assert np.array_equal(old["maps"]["current"], new["maps"]["current"])
    assert not np.array_equal(old["maps"]["didv"], new["maps"]["didv"])
    assert not np.allclose(old["i"], new["i"], rtol=1e-3, atol=0.0)
    assert old["states"] == new["states"]


def _expected_shares(w, x, y, kappa):
    """w·exp(2κ(h − h_max)) over the apexes, h the smooth surface under each plus its dz, normalised."""
    hs = [w.surface.height_smooth(x + a.dx, y + a.dy) + a.dz for a in w.tip.apexes]
    top = np.maximum.reduce(hs)
    q = [a.w * np.exp(2.0 * kappa * (h - top)) for a, h in zip(w.tip.apexes, hs)]
    tot = sum(q)
    return [qi / tot for qi in q]


def test_a_double_tip_sees_each_apex_by_its_share_of_the_current(tmp_path):
    """A spectrum is the apexes' LDOS summed by their share of the current, and a map is the same
    pixel by pixel; a flickering tip is summed over its own apexes, and nothing draws."""
    inst = Instrument(P3, SEED, tmp_path / "run")
    w = inst.world
    _double(w.tip)
    w.tip.flicker_dz_m = 0.05e-9                 # a pass would redraw the second apex's dz
    cx, cy = (v * 1e-9 for v in w.surface.site.adatoms.ring["centre_nm"])
    kappa = w.kappa_m()
    before = _states(w)

    share = [float(s[0]) for s in _expected_shares(w, np.array([cx]), np.array([cy]), kappa)]
    assert sum(share) == pytest.approx(1.0, abs=1e-12)
    assert 0.05 < share[1] < 0.5                 # the second apex carries a real part of the current
    e = np.linspace(-0.5, 0.1, 61)
    seen = w.ldos_under_tip(cx, cy)
    assert isinstance(seen, ParallelLDOS)
    parts = [w.surface.ldos_at(cx + a.dx, cy + a.dy) for a in w.tip.apexes]
    want = share[0] * parts[0].rho(e) + share[1] * parts[1].rho(e)
    np.testing.assert_allclose(seen.rho(e), want, rtol=1e-12, atol=0.0)
    assert np.max(np.abs(seen.rho(e) - parts[0].rho(e))) > 1e-3    # not the centre's LDOS alone
    # iv_curve is linear in the sample LDOS: weighting the LDOS weights the current
    vs, gap, phi = np.linspace(-0.5, 0.1, 13), 0.6e-9, w.phi_junction()
    np.testing.assert_allclose(iv_curve(vs, gap, phi, seen, w.tip.ldos),
                               share[0] * iv_curve(vs, gap, phi, parts[0], w.tip.ldos)
                               + share[1] * iv_curve(vs, gap, phi, parts[1], w.tip.ldos),
                               rtol=1e-12, atol=1e-30)

    xs = cx + np.linspace(-4e-9, 4e-9, 17)
    ys = np.full_like(xs, cy)
    sh = _expected_shares(w, xs, ys, kappa)
    want_map = sum(s * w.surface.ldos_map(xs + a.dx, ys + a.dy, -0.3) for s, a in zip(sh, w.tip.apexes))
    got_map = w.ldos_map_under_tip(xs, ys, -0.3, kappa)
    np.testing.assert_allclose(got_map, want_map, rtol=1e-12, atol=0.0)
    assert not np.allclose(got_map, w.surface.ldos_map(xs, ys, -0.3), rtol=1e-6, atol=0.0)
    assert _states(w) == before                  # the tip's own apexes: no flicker draw, no draw at all


def test_without_a_surface_state_any_tip_sees_the_material_template(tmp_path):
    """No surface state, no position dependence: the template under the centre and a flat map,
    exactly as before, however many apexes the tip has."""
    inst = Instrument(P4, SEED, tmp_path / "run")
    w = inst.world
    _double(w.tip)
    assert w.surface.surface_state is None
    x = np.linspace(-3e-9, 3e-9, 9)
    assert w.apex_current_shares(x, x) is None
    seen = w.ldos_under_tip(0.0, 0.0)
    assert type(seen) is LDOSTemplate and seen == w.surface.ldos_at(0.0, 0.0)
    assert np.array_equal(w.ldos_map_under_tip(x, x, 0.1), np.ones_like(x))


def test_one_apex_off_the_centre_is_sampled_where_it_is(tmp_path):
    """A tip whose only apex sits off its centre (a change can leave one so) takes the LDOS under
    that apex, with all of the current."""
    inst = Instrument(P3, SEED, tmp_path / "run")
    w = inst.world
    w.tip.apexes = [Apex(0.9e-9, -0.5e-9, 0.0, 1.0)]
    cx, cy = (v * 1e-9 for v in w.surface.site.adatoms.ring["centre_nm"])
    (a, s), = w.apex_current_shares(np.array([cx]), np.array([cy]))
    assert float(s[0]) == 1.0
    e = np.linspace(-0.5, 0.1, 31)
    seen = w.ldos_under_tip(cx, cy).rho(e)
    assert np.array_equal(seen, w.surface.ldos_at(cx + 0.9e-9, cy - 0.5e-9).rho(e))
    assert np.max(np.abs(seen - w.surface.ldos_at(cx, cy).rho(e))) > 1e-3
