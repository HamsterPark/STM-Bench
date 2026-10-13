"""The P4 privileged reference solution: an atom moved onto its lattice site by a resistance
ladder, the site computed from a lattice measured with the other atoms."""
from __future__ import annotations

from stmbench.reference import _common, p4


def test_the_reference_moves_the_atom_onto_its_site():
    """Seed 28 arrives with a good tip. No threshold is read: the ladder starts at 400 kΩ, where
    this atom stays put, and goes down until it follows."""
    r = p4.run(28)
    assert r["success"] and r["on_goal"], r
    assert r["bystander_max_shift_nm"] == 0.0
    assert r["abs_err"]["target_site"] < 0.1
    assert r["tip_rounds"] == 0 and r["pickups"] == 0
    att = r["attempts"]
    assert att[0]["r_kohm"] == 400.0 and att[0]["spikes"] == 0 and att[-1]["spikes"] > 0


def test_an_atom_lost_onto_the_tip_is_buried_and_another_one_moved(monkeypatch):
    """An atom that goes onto the tip (here one made to: its thresholds ×100, so the first grab
    lifts it) is buried with a shallow poke far away, and another isolated atom is moved."""
    init = _common.Session.__init__

    def sticky(self, *a, **kw):
        init(self, *a, **kw)
        reg = self.w.surface.site.adatoms
        reg.atoms[reg.target["atom_id"]].r_scale = 100.0

    monkeypatch.setattr(_common.Session, "__init__", sticky)
    r = p4.run(28)
    assert r["success"], r
    assert r["pickups"] == 1 and r["atoms_tried"] == 2


def test_a_site_tied_with_the_nearest_counts():
    """Seed 20: the measured lattice puts the atom on a site within 0.02 nm as near to
    birth + d as the nearest one, which the judge accepts (P4's site_tie_nm)."""
    r = p4.run(20)
    assert r["success"] and not r["on_goal"], r
