"""The rule card: every rule the judge applies to a paper is stated in the task the agent gets,
from the same scenario fields the judge reads."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import pytest

from stmbench.trackB import rules as R
from stmsim.physics.adatoms import GONE_SHIFT_NM, AdatomParams, AdatomRegistry
from stmsim.physics.surface import Surface
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
PAPERS = sorted(SCEN.glob("P*.yaml"))
NAMES = [p.name for p in PAPERS]
P4 = SCEN / "P4_atom_positioning_cu111.yaml"


def _num(x) -> str:
    return f"{float(x):g}"


@pytest.mark.parametrize("name", NAMES)
def test_no_paper_task_states_its_budget_by_hand(name):
    """P1's task once promised 2 h while the overrun rule used the file's 3 h."""
    assert not R.has_budget_sentence(Scenario.load(SCEN / name).task)


@pytest.mark.parametrize("name", NAMES)
def test_the_card_states_the_budget_the_overrun_rule_uses(name):
    sc = Scenario.load(SCEN / name)
    card = R.rule_card(sc)
    assert f"预算 {_num(sc.budget['sim_hours'])} 小时仪器时间" in card
    assert f"{_num(round(100 * (R.OVERRUN_TOLERANCE - 1), 6))}%" in card


@pytest.mark.parametrize("name", NAMES)
def test_every_claim_is_explained_with_its_tolerance_and_evidence(name):
    sc = Scenario.load(SCEN / name)
    card = R.rule_card(sc)
    for c in sc.claims:
        assert c.get("say"), f"{sc.id}: claim {c['id']} has no `say` for the rule card"
        block = card.split(f"{c['id']}", 1)[1]
        assert "判对条件" in block and "证据" in block
        tol = c.get("tol") or {}
        for key in ("abs", "report_nm"):
            if key in tol:
                assert _num(tol[key]) in block, (c["id"], key)
        if "rel" in tol:
            assert f"{_num(100 * float(tol['rel']))}%" in block, c["id"]
        ev = c["evidence"]
        for key in ("min_fov_nm", "max_nm_per_px", "min_points", "near_nm", "max_centre_dist_nm"):
            if key in ev:
                assert _num(ev[key]) in block, (c["id"], key)


def test_the_p4_verification_frame_is_on_the_card():
    """The rule that was missing from P4's task: a close frame, after the last hop, in the
    frame the report names."""
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    rule = next(c for c in sc.claims if c["id"] == "target_site")["evidence"]
    card = R.rule_card(sc)
    assert f"像素尺寸（宽度 ÷ 列数）不大于 {_num(rule['max_nm_per_px'])} nm" in card
    assert "最后一次原子跳动" in card and "frame" in card
    # and the near-tied sites the judge also accepts: a site farther by the tie itself counts (<=)
    tie = next(c for c in sc.claims if c["id"] == "target_site")["tol"]["site_tie_nm"]
    assert f"比最近的格点至多远 {_num(tie)} nm" in card and "不到" not in card


def test_the_card_explains_citations_and_coordinates():
    card = R.rule_card(Scenario.load(PAPERS[0]))
    for needle in ("evidence", "frame", ".sxm", ".dat", "逆时针", "漂移"):
        assert needle in card


def test_task_text_appends_the_card_to_papers_only():
    paper = Scenario.load(SCEN / "P5_sader_jarvis_qplus.yaml")
    assert R.task_text(paper).startswith(paper.task.strip())
    assert R.task_text(paper).endswith(R.rule_card(paper))
    regression = Scenario.load(SCEN / "B5_repair_blunt.yaml")
    assert R.task_text(regression) == regression.task.strip()


def test_the_card_never_states_a_hidden_value():
    """Tolerances are public; the values drawn for a seed are not."""
    for p in PAPERS:
        sc = Scenario.load(p)
        card = R.rule_card(sc)
        for seed in (0, 1):
            for block, body in sc.draw_hidden(seed).items():
                for key, val in body.items():
                    if isinstance(val, float):
                        assert f"{val:.4f}" not in card, (sc.id, block, key, val)


def test_the_judge_reads_the_version_the_card_announces():
    sc = Scenario.load(PAPERS[0])
    assert re.search(rf"评分规则 v{R.RULES_VERSION}", R.rule_card(sc))
    assert R.RULES_VERSION >= 2, "version 1 is the card before P4's was made exact"


# ── version 2: P4's card says what the judge does, and nothing about how to operate ────────────
def test_a_hop_is_explained_by_what_it_is_and_not_by_what_causes_it():
    """The card named the causes of an atom hop (dragging, imaging at a low junction resistance):
    operating knowledge the agent should bring or find, not a scoring rule. P3's repair shares the
    sentence, because its evidence is timed by the same event."""
    for name in ("P4_atom_positioning_cu111.yaml", "P3_corral_repair_cu111.yaml"):
        card = R.rule_card(Scenario.load(SCEN / name))
        assert "最后一次原子跳动（指任何原子从一个晶格位点跳到相邻位点，不论是什么使它动的）" in card, name
        for cause in ("拖动", "带动", "结电阻", "成像"):
            assert cause not in card, (name, cause)


def test_the_card_does_not_use_the_simulators_word_for_the_start_of_the_episode():
    for p in PAPERS:
        assert "出生" not in R.task_text(Scenario.load(p)), p.name
    card = R.rule_card(Scenario.load(P4))
    assert "这个原子本局开始时 3 nm 内没有别的原子" in card
    assert "'它本局开始时的位置平移 (4, 0) nm 后最近的那个晶格位点'" in card


def test_the_report_is_matched_to_the_moved_atom_nearest_to_it():
    """The bystander claim is about "the atom target_site was matched to": the judge matches the report
    to the nearest moved atom and only then asks whether it is within report_nm (test_rules_v1)."""
    card = R.rule_card(Scenario.load(P4))
    assert "所报位置须在离它最近的、被你搬动过的原子 0.6 nm 以内" in card


def test_the_p4_card_defines_the_bystanders_from_the_fields_the_judge_reads():
    """The radius and the shift come from ``initial.adatoms`` (``bystander_nm``, ``target_dx_nm``,
    ``target_dy_nm``), the fields the world is built from, not from a number typed into the claim."""
    sc = Scenario.load(P4)
    claim = next(c for c in sc.claims if c["id"] == "bystander_max_shift_nm")
    assert not re.search(r"\d", claim["say"] + claim["say_en"]), "a figure in `say` would drift from the field"
    block = R.rule_card(sc).split("bystander_max_shift_nm", 1)[1]
    assert "计入的其他原子：本局开始时，离该原子本局开始时的位置不超过 10 nm、" \
           "或离'该位置平移 (4, 0) nm 后最近的那个晶格位点'不超过 10 nm 的每一个其他原子" in block
    assert "位移的算法：每个计入的原子，取它本局开始时的位置到本局结束时所在位置的距离；" \
           "丢失的、或结束时还在针尖上的原子算移动；这个量取其中最大的（没有计入的原子则为 0）" in block
    moved = copy.deepcopy(sc)
    moved.initial["adatoms"].update(bystander_nm=12.5, target_dx_nm=-3.0, target_dy_nm=1.25)
    block = R.rule_card(moved).split("bystander_max_shift_nm", 1)[1]
    assert "不超过 12.5 nm、或离'该位置平移 (-3, 1.25) nm 后最近的那个晶格位点'不超过 12.5 nm" in block
    assert "10 nm" not in block
    # a file that leaves the figures out gets the ones the world is built with (Surface.configure_adatoms)
    bare = copy.deepcopy(sc)
    for key in ("target_dx_nm", "target_dy_nm", "isolation_nm", "bystander_nm"):
        del bare.initial["adatoms"][key]
    assert R.rule_card(bare) == R.rule_card(sc)


def test_the_hop_the_evidence_needs_is_named_by_the_action():
    """The judge counts the ``adatom_hop`` events whose cause is ``folme``, which the instrument gives the
    hops of its one dragging action, move_tip (the last test of this file runs it)."""
    sc = Scenario.load(P4)
    assert "（不需要引用）本局至少有 1 次原子跳动发生在 move_tip 动作中" in R.rule_card(sc)
    assert "拖动" not in R.rule_card(sc)
    # the judge counts events (one per atom per action), so a larger number says what is counted
    three = copy.deepcopy(sc)
    next(c for c in three.claims if c["id"] == "bystander_max_shift_nm")["evidence"]["min_n"] = 3
    assert "本局至少有 3 次原子跳动发生在 move_tip 动作中（同一个原子在一次 move_tip 动作中的多次跳动只算一次）" \
           in R.rule_card(three)
    # another event keeps the cause the judge requires of it
    other = copy.deepcopy(sc)
    other.claims[1]["evidence"] = {"kind": "events", "event": "zspec", "cause": "scan_x", "min_n": 2}
    assert "（不需要引用）本局至少发生过 2 次zspec（原因：scan_x）" in R.rule_card(other)
    other.claims[1]["evidence"] = {"kind": "events", "event": "zspec"}
    assert "（不需要引用）本局至少发生过 1 次zspec" in R.rule_card(other) and "原因" not in R.rule_card(other)


@pytest.mark.parametrize("name", NAMES)
def test_the_card_states_no_controller_command_limit(name):
    """``wire_cmds`` is a limit of the controller TCP path. The benchmark instrument counts no commands
    (it enforces the instrument-time budget and an action limit), so a limit on the card would be a rule
    nothing applies."""
    sc = Scenario.load(SCEN / name)
    assert sc.budget["wire_cmds"] > 0, "the file still holds the number: the card is what leaves it out"
    card = R.task_text(sc)
    assert "控制器" not in card and "命令" not in card and str(int(sc.budget["wire_cmds"])) not in card
    assert card.rstrip().endswith("本次复现不算成功。")             # the budget sentence is the last line


def _p4_atoms():
    """A registry as P4's world builds it (any isolated atom, moved by (4, 0) nm; 3 nm and 10 nm): the
    mover, which has reached its goal, and three atoms of each kind around it, none within 0.3 nm of a
    10 nm limit - lying within 10 nm of the mover's start only, of its goal site only, and of neither."""
    reg = AdatomRegistry(Surface("Cu(111)", seed=3, adsorbate_density_per_um2=0.0).site, AdatomParams())
    mover = reg.add(0, 0, role="field")
    sx, sy = reg.site_xy(0, 0)
    goal = reg.site_of(sx + 4e-9, sy)
    gx, gy = reg.site_xy(*goal)
    pools: dict[str, list[tuple[int, int]]] = {"start": [], "goal": [], "neither": []}
    for i in range(-90, 91):
        for j in range(-90, 91):
            x, y = reg.site_xy(i, j)
            d_start, d_goal = math.hypot(x - sx, y - sy) * 1e9, math.hypot(x - gx, y - gy) * 1e9
            if d_start < 4.0 or d_goal < 4.0:
                continue
            if d_start <= 9.7 and d_goal >= 10.3:
                pools["start"].append((i, j))
            elif d_goal <= 9.7 and d_start >= 10.3:
                pools["goal"].append((i, j))
            elif 10.3 <= d_start <= 14.0 and 10.3 <= d_goal <= 14.0:
                pools["neither"].append((i, j))
    assert all(len(pool) >= 3 for pool in pools.values()), {k: len(v) for k, v in pools.items()}
    atoms = {name: [reg.add(*site, role="field") for site in pool[:3]] for name, pool in pools.items()}
    reg.target = {"rule": "any", "atom_id": mover.id, "d_nm": [4.0, 0.0], "isolation_nm": 3.0,
                  "bystander_nm": 10.0}
    mover.i, mover.j = goal
    mover.history.append((1.0, goal[0], goal[1], "folme"))
    return reg, mover, atoms


def _quantity(reg, mover) -> float:
    """What the ``bystander_max_shift_nm`` claim reads for the atom the report was matched to."""
    return next(c for c in reg.any_target()["candidates"] if c["id"] == mover.id)["bystander_max_shift_nm"]


def _hop(atom, spacings: int) -> None:
    """The atom hopped ``spacings`` sites along one lattice vector: a displacement of as many spacings."""
    atom.i += spacings
    atom.history.append((2.0, atom.i, atom.j, "folme"))


def test_the_judge_counts_the_atoms_the_card_describes_and_measures_them_from_their_own_start():
    """The card's two bystander lines against ``AdatomRegistry.any_target``, the function the claim
    reads: every other atom counts, whatever its history, that lay within the radius of the moved atom's
    start or of its goal site at the start of the episode; each is measured from its own start; the
    quantity is the largest displacement."""
    reg, mover, atoms = _p4_atoms()
    nn = float(reg.site.material.nn_m) * 1e9
    near_start, near_goal, neither = atoms["start"], atoms["goal"], atoms["neither"]
    assert _quantity(reg, mover) == 0.0                      # nothing else has moved
    _hop(near_start[0], 1)                                   # near the start only: counts
    assert _quantity(reg, mover) == pytest.approx(nn)
    _hop(near_goal[0], 2)                                    # near the goal only: counts; the largest wins
    assert _quantity(reg, mover) == pytest.approx(2 * nn)
    _hop(neither[0], 30)                                     # near neither: does not count, however far it went
    assert _quantity(reg, mover) == pytest.approx(2 * nn)
    # who counts is fixed at the start of the episode: an atom that has since moved into the
    # neighbourhood does not join it (a site 1 nm from the mover's start), and one that has moved out stays
    neither[1].i, neither[1].j = 4, 0
    neither[1].history.append((2.0, 4, 0, "folme"))
    assert _quantity(reg, mover) == pytest.approx(2 * nn)
    _hop(near_goal[1], 90)
    assert _quantity(reg, mover) == pytest.approx(90 * nn)


@pytest.mark.parametrize("where", ["start", "goal"])
@pytest.mark.parametrize("how", ["lost", "on the tip"])
def test_a_lost_bystander_or_one_on_the_tip_counts_as_moved(where, how):
    reg, mover, atoms = _p4_atoms()
    victim = atoms[where][2]
    assert _quantity(reg, mover) == 0.0
    if how == "lost":
        victim.status = "lost"
    else:
        reg.pick_up(victim, sim_s=3.0)
    assert _quantity(reg, mover) == GONE_SHIFT_NM > 0.2


def test_with_no_other_atom_in_reach_the_quantity_is_zero():
    reg, mover, atoms = _p4_atoms()
    for group in atoms.values():
        for atom in group:
            if group is not atoms["neither"]:
                reg.atoms.pop(atom.id)                       # nobody within 10 nm of the start or the goal
    assert _quantity(reg, mover) == 0.0
    atoms["neither"][0].status = "lost"                      # a lost atom out of reach is still out of reach
    assert _quantity(reg, mover) == 0.0


def test_the_evidence_hop_is_one_made_during_a_move_tip_action_on_the_instrument():
    """The card: the evidence needs "an atom hop made during a move_tip action", while the last atom hop is
    the last time any atom hopped, whatever moved it. On the instrument, imaging at a low junction resistance
    makes ``adatom_hop`` events too (cause ``scan``): they count as hops, but are not the evidence. A drag
    in a move_tip action makes the ones (cause ``folme``) that are."""
    import numpy as np

    from stmbench.instrument import Instrument
    from stmbench.trackB import claims as C

    sc = Scenario.load(P4)
    seed = next(s for s in range(40) if sc.draw_hidden(s)["tip"]["condition"] == "good")
    rule = next(c for c in sc.claims if c["id"] == "bystander_max_shift_nm")["evidence"]
    inst = Instrument(sc, seed)
    w = inst.world
    w.drift_v_m_per_s = np.zeros(3)                          # the scan frame is the sample frame here
    w.creep_gamma = 0.0
    reg = w.surface.site.adatoms
    atom = reg.atoms[reg.target["atom_id"]]                  # the atom the world keeps a clear corridor for
    ax, ay = (v * 1e9 for v in reg.site_xy(atom.i, atom.j))

    def hops(cause=None):
        return [e for e in w.events if e["kind"] == "adatom_hop" and cause in (None, e["cause"])]

    def met() -> bool:
        return C.check_evidence(rule, events=w.events, report={}, truth={})[0]

    # a junction resistance 0.4 of this atom's pull threshold, on a current range that holds its current
    inst.act("set_current_range", full_scale_na=100.0)
    inst.act("set_bias", bias_v=0.001)
    inst.act("set_setpoint", current_pa=0.001 / (0.4 * reg.params.r_threshold_ohm * atom.r_scale) * 1e12)
    assert not hops() and not met()
    inst.act("scan", center_x_nm=ax, center_y_nm=ay, size_nm=3.0, pixels=16, line_time_s=30.0)
    assert hops() and {e["cause"] for e in hops()} == {"scan"}, "imaging made the atom hop"
    assert not met(), "hops made by imaging are not the evidence"
    # drag it by (4, 0) nm in move_tip actions
    now = reg.site_xy(atom.i, atom.j)
    inst.act("move_tip", x_nm=now[0] * 1e9, y_nm=now[1] * 1e9)
    goal = reg.site_xy(*reg.site_of(atom.x0 + 4e-9, atom.y0))
    inst.act("move_tip", x_nm=goal[0] * 1e9, y_nm=goal[1] * 1e9, speed_nm_s=0.2)
    assert hops("folme") and met()
    moves = [r for r in inst.log if r["kind"] == "act" and r["name"] == "move_tip"]
    for e in hops("folme"):
        t = e["sim_s"] - inst.t0
        assert any(m["t0_s"] - 1e-5 <= t <= m["t1_s"] + 1e-5 for m in moves), e
