"""The families of the internal evaluation tree: one table row per scenario, and the texts that go with it.

Everything the tree builder needs to know about a family is in ``FAMILIES``; nothing here is a seed, a world
value or a mapping from a task id to a tip class (the repository is public). The worlds come from a private
file, see ``build_tree.py``.
"""
from __future__ import annotations

import re

SNAPSHOT = "ubuntu_stm_eval"
TASK_VERSION = "1.0.0"
EXPORT_DIR = "/var/lib/stm-bench/export"
EXPORT_FILES = ("session.json", "actions.jsonl", "digests.jsonl", "report.json")   # session.json first

FRAME = ("A program that controls a (simulated) scanning tunnelling microscope is open on the screen of "
         "this computer. Using it, do the following experiment.\n\n")

#: the one sentence about the plain-text copies (PLAN.md section 3.3). It is about the software only: it does not
#: suggest how to analyse anything. Kept in one place so that the final wording is a one-line change.
DATA_SENTENCE = ("Every image, spectrum and move trace the program saves is also written as a plain-text CSV file "
                 "to /srv/stm-data. You can open a terminal (Ctrl+Alt+T) and analyse these files with python3; "
                 "numpy and scipy are installed.")

SETUP_SENTENCE = ("At task start the setup closes /opt/stm-bench (the simulator's source, which includes the "
                  "scenario files, the privileged reference solution and the scripted baseline) to the agent's "
                  "account, then revokes that account's admin rights.")

#: provisional step budgets (PLAN.md section 1); a blind trial fixes the final ones
MAX_STEPS = 250
MAX_STEPS_P7 = 400

NO_REFERENCE_YET = ("The grader tests of this family are written, but the privileged reference solution that makes "
                    "their fixtures does not exist yet, so they have not been run.")

_AUDIT_TAIL = " " + SETUP_SENTENCE


def _audit_p3_measure() -> tuple[str, str]:
    rh = (
        "The score comes from replaying the exported calls on the instrument simulation with "
        "the task's seed, which the agent's account cannot read: each reported number needs the "
        "measurement that the rule card requires (for the two peak energies a dI/dV spectrum "
        "taken within 1 nm of the corral centre, at least 100 points, covering -0.45 to +0.1 V; "
        "for the radius a complete constant-current frame at least 22 nm wide at no more than "
        "0.15 nm per pixel), and the replay must equal the recorded episode in every reply, "
        "instrument time and spectrum or frame hash. The ring's radius, the Fe atoms and the "
        "surface state differ per seed, so a fixed answer sheet does not carry over (the "
        "scenario's own chance audit on 2000 seeds: the best fixed sheet reproduced all claims on 2.30 % of "
        "seeds when chosen and scored on the same seeds, and on 0.90 % when chosen on seeds 0-999 and scored on "
        "seeds 1000-1999; the earlier 4.3 % was 300 seeds, in-sample). "
        "Tested shortcuts that score below 1.0: the export of another task, a wrong seed, a "
        "log cut short (also with its record count corrected), a record removed or altered, "
        "a tampered or missing digest, a peak 25 meV off, the radius 0.4 nm off, the two peaks "
        "swapped, peaks citing only a spectrum taken away from the centre or no spectrum, a "
        "radius citing a frame that is too small, an empty episode, a report with no "
        "instrument calls (private/grader_tests/run_tests_p3.py)." + _AUDIT_TAIL)
    fn = ("In-tolerance variants (each peak energy moved by 5 of the 10 meV allowed, in either "
          "direction; the radius moved by 0.07 of the 0.15 nm allowed, in either direction; an "
          "episode that never reached finish) score 1.0. The rule card accepts, for each peak, "
          "any of the three lowest true resonances (whether the lowest level is resolvable "
          "varies with the seed).")
    return rh, fn


def _audit_p3_repair() -> tuple[str, str]:
    rh = (
        "The score comes from replaying the exported calls on the instrument simulation with "
        "the task's seed, which the agent's account cannot read. The corral must really be "
        "closed: the ring's occupancy is read from the replayed world, not from anything reported, "
        "and its evidence is a close frame taken after the last atom hop; the spectra that "
        "carry the peak energies must also be taken after the last atom hop. The replay must equal "
        "the recorded episode in every reply, instrument time and spectrum or frame hash. The "
        "ring, its gap, the spare atoms and the surface state differ per seed. Tested shortcuts that "
        "score below 1.0 (private/grader_tests/run_tests_p3_repair.py): the export of another task, a "
        "wrong seed, a log cut short (also with its record count corrected, and stopped after the repair "
        "but before the first spectrum), a record removed or altered, a tampered or missing digest, a "
        "peak 25 meV off, the radius off by more than the tolerance, the two peaks swapped, peaks "
        "citing only a spectrum taken away from the centre or no spectrum, a radius citing a frame of "
        "the broken ring, the broken corral measured without any repair, the repair stopped after the "
        "first spare, an empty episode, a report with no instrument calls." + _AUDIT_TAIL)
    fn = ("In-tolerance variants (each reported number moved by half the tolerance in the rule card; an "
          "episode that never reached finish) score 1.0. The rule card accepts, for each peak, "
          "any of the three lowest true resonances (the ring is recomputed from where the atoms stand, "
          "and whether the lowest level is resolvable varies with the seed).")
    return rh, fn


def _audit_p4() -> tuple[str, str]:
    rh = (
        "The score comes from replaying the exported calls on the instrument simulation with the task's seed, "
        "which the agent's account cannot read: a reported number needs the frames that the rule card requires "
        "(complete, constant-current, scanned after the last atom hop), and the replay must equal the recorded "
        "episode in every reply, instrument time and frame hash. Tested shortcuts that score below 1.0: the "
        "export of another task, a wrong seed, a log cut short (also with its record count corrected), a "
        "record removed or altered, a tampered or missing digest, the reported place moved off the atom, a "
        "report citing an earlier frame, an empty episode, a report with no instrument calls "
        "(private/grader_tests/run_tests_p4.py)." + _AUDIT_TAIL)
    fn = ("In-tolerance variants (reported place moved by 0.3 nm of the 0.6 nm allowed; an episode that never "
          "reached finish) score 1.0.")
    return rh, fn


def _audit_p5() -> tuple[str, str]:
    rh = (
        "The score comes from replaying the exported calls on the instrument simulation with the task's seed, "
        "which the agent's account cannot read. Each of the three reported numbers (F_min, the decay length and "
        "the binding energy) must cite a pair of Z spectra of the rule card: one taken within 0.15 nm of an "
        "adatom, whose frequency shift passes its minimum by at least 50 pm of tip travel, and one taken at "
        "least 2 nm from any adatom (the background). The replay must equal the recorded episode in every "
        "reply, instrument time and spectrum hash. The bond (depth, decay length, equilibrium distance), the "
        "sensor and the tip state differ per seed, so a fixed answer sheet does not carry over (the "
        "scenario's own calibration: the best fixed sheet reproduced all three claims on 4.3 % of seeds "
        "0-299). Tested shortcuts that score below 1.0 (private/grader_tests/run_tests_p5.py): the export of "
        "another task, a wrong seed, a log cut short (also with its record count corrected), a record removed "
        "or altered, a tampered or missing digest, each value 2.5 tolerances off, values citing only the "
        "spectra taken over the atom, only the background spectra or no spectrum, an empty episode, a report "
        "with no instrument calls." + _AUDIT_TAIL)
    fn = ("In-tolerance variants (each reported value moved by half of its relative tolerance, 15 % for F_min, "
          "8 % for the decay length and 20 % for the binding energy, in either direction where the "
          "reference solution's own error leaves it inside; an episode that never reached finish) score 1.0.")
    return rh, fn


def _audit_p7() -> tuple[str, str]:
    rh = (
        "The score comes from replaying the exported calls on the instrument simulation with the task's seed, "
        "which the agent's account cannot read. Each reported threshold must cite a series of the rule card's "
        "drags (at least 8, at 3 or more different resistances, each of 1.0 nm at 0.1 nm/s and started within "
        "0.3 nm of the atom), and the judge computes the half-follow resistance of the cited drags itself from "
        "the replayed world; the second threshold must belong to another atom than the first. The replay must "
        "equal the recorded episode in every reply, instrument time and trace hash. The atoms' own thresholds, "
        "grips and the tip state differ per seed, so a fixed answer sheet does not carry over (the scenario's "
        "own audit: a constant pair passed both claims on 3.2 % of held-out seeds 1000-1999). Tested shortcuts "
        "that score below 1.0 (private/grader_tests/run_tests_p7.py): the export of another task, a wrong seed, "
        "a log cut short (also with its record count corrected), a record removed or altered, a tampered or "
        "missing digest, a value 25 % off, a series cut to 3 drags, citing no drag, both claims citing the same "
        "atom's drags, an empty episode, a report with no instrument calls." + _AUDIT_TAIL)
    fn = ("In-tolerance variants (each reported threshold moved by 5 % of the 10 % allowed, in either direction "
          "where the reference solution's own error leaves it inside; an episode that never reached finish) "
          "score 1.0. The rule card's threshold is the one of the cited drags, not the law's own parameter, so "
          "a fit of a different but sound series of the same atom also passes when it lands inside 10 %.")
    return rh, fn


def _audit_p1() -> tuple[str, str]:
    rh = (
        "The score comes from replaying the exported calls on the instrument simulation with the task's seed, "
        "which the agent's account cannot read. The stripe period needs a frame at least 100 nm wide at no more "
        "than 0.4 nm per pixel; each of the three orientations needs a frame covering the reported position, at "
        "least 40 nm wide at no more than 0.5 nm per pixel, and the two domains must differ by at least 30 "
        "degrees (one domain read twice does not pass). The replay must equal the recorded episode in every "
        "reply, instrument time and frame hash. The herringbone's period, its arms and its rotational domains "
        "differ per seed. " + NO_REFERENCE_YET + _AUDIT_TAIL)
    fn = ("The rule card accepts the mean stripe direction or either arm of the zigzag, and an orientation "
          "within 6 degrees modulo 180; the period is the one along the arm, within 0.15 nm. "
          + NO_REFERENCE_YET)
    return rh, fn


def _audit_p2() -> tuple[str, str]:
    rh = (
        "The score comes from replaying the exported calls on the instrument simulation with the task's seed, "
        "which the agent's account cannot read. The surface-state band edge and the effective mass each need "
        "either at least 8 dI/dV spectra at 6 or more positions, each within 8 nm of a scatterer and showing no tip "
        "jump (at least 64 points, covering at most -0.5 V to at least +0.1 V), or at least 4 dI/dV maps (lock-in "
        "on, constant current, complete) at 4 or more bias values covering -0.3 V to +0.05 V, each centred within "
        "10 nm of a scatterer, at least 15 nm wide and at most 0.3 nm per pixel. The replay must equal the recorded episode in every reply, "
        "instrument time and spectrum or frame hash. The surface state's parameters and the scatterers differ "
        "per seed. " + NO_REFERENCE_YET + _AUDIT_TAIL)
    fn = ("The band edge is accepted within 15 meV and the effective mass within 2 %. "
          + NO_REFERENCE_YET)
    return rh, fn


def _source(text: str) -> str:
    return f"STM-Bench task {text} on its simulated scanning tunnelling microscope"


#: One row per family. ``scenario`` is the YAML id of stmbench/trackB/scenarios; ``prefix`` the task-id stem (the
#: ids are ``<prefix>_NN``); ``reference`` the module of stmbench/reference that plays the privileged reference
#: solution (None: it does not exist yet); ``tests`` the grader-test runner file; ``task_glob`` what that runner
#: grades.
FAMILIES: dict[str, dict] = {
    "p1": dict(scenario="P1_barth1990_au111", prefix="stm_eval_p1_herringbone", reference=None,
               tests="run_tests_p1.py", task_glob="stm_eval_p1_herringbone_[0-9][0-9]", max_steps=MAX_STEPS,
               family="STM-Bench P1", paper="Barth, Brune, Ertl and Behm, Phys. Rev. B 42, 9307 (1990)",
               source=_source("P1 (the herringbone reconstruction of Au(111) after Barth, Brune, Ertl and Behm, "
                              "Phys. Rev. B 42, 9307, 1990)"),
               audits=_audit_p1),
    "p2": dict(scenario="P2_standing_waves_cu111", prefix="stm_eval_p2_waves", reference=None,
               tests="run_tests_p2.py", task_glob="stm_eval_p2_waves_[0-9][0-9]", max_steps=MAX_STEPS,
               family="STM-Bench P2", paper="Crommie, Lutz and Eigler, Nature 363, 524 (1993)",
               source=_source("P2 (standing waves of the surface state of Cu(111) after Crommie, Lutz and Eigler, "
                              "Nature 363, 524, 1993)"),
               audits=_audit_p2),
    "p3": dict(scenario="P3_corral_cu111", prefix="stm_eval_p3_corral", reference="p3",
               tests="run_tests_p3.py", task_glob="stm_eval_p3_corral_[0-9][0-9]", max_steps=MAX_STEPS,
               family="STM-Bench P3", paper="Crommie, Lutz and Eigler, Science 262, 218 (1993)",
               source=_source("P3 (resonances of a quantum corral of Fe atoms on Cu(111) after Crommie, Lutz and "
                              "Eigler, Science 262, 218, 1993)"),
               audits=_audit_p3_measure),
    "p3_repair": dict(scenario="P3_corral_repair_cu111", prefix="stm_eval_p3_corral_repair", reference="p3_repair",
                      tests="run_tests_p3_repair.py", task_glob="stm_eval_p3_corral_repair_[0-9][0-9]",
                      max_steps=MAX_STEPS, family="STM-Bench P3 repair",
                      paper="Crommie, Lutz and Eigler, Science 262, 218 (1993)",
                      source=_source("P3, repair variant (close the gap in a quantum corral of Fe atoms on Cu(111) "
                                     "by lateral manipulation, then measure its resonances, after Crommie, Lutz "
                                     "and Eigler, Science 262, 218, 1993)"),
                      audits=_audit_p3_repair),
    "p4": dict(scenario="P4_atom_positioning_cu111", prefix="stm_eval_p4_atom", reference="p4",
               tests="run_tests_p4.py", task_glob="stm_eval_p4_atom_[0-9][0-9]", max_steps=MAX_STEPS,
               family="STM-Bench P4", paper="Eigler and Schweizer, Nature 344, 524 (1990)",
               source=_source("P4 (single-atom positioning on Cu(111) after Eigler and Schweizer, Nature 344, "
                              "524, 1990)"),
               audits=_audit_p4),
    "p5": dict(scenario="P5_sader_jarvis_qplus", prefix="stm_eval_p5_force", reference="p5",
               tests="run_tests_p5.py", task_glob="stm_eval_p5_force_[0-9][0-9]", max_steps=MAX_STEPS,
               family="STM-Bench P5", paper="Huber et al., Science 366, 235 (2019); Sader and Jarvis, "
                                            "Appl. Phys. Lett. 84, 1801 (2004)",
               source=_source("P5 (force spectroscopy of an Fe adatom on Cu(111) with a qPlus sensor and the "
                              "Sader-Jarvis inversion, after Huber et al., Science 366, 235, 2019)"),
               audits=_audit_p5),
    "p7": dict(scenario="P7_pull_threshold_cu111", prefix="stm_eval_p7_pull", reference="p7",
               tests="run_tests_p7.py", task_glob="stm_eval_p7_pull_[0-9][0-9]", max_steps=MAX_STEPS_P7,
               family="STM-Bench P7", paper="Bartels, Meyer and Rieder, Phys. Rev. Lett. 79, 697 (1997)",
               source=_source("P7 (the junction resistance at which an Fe adatom on Cu(111) follows the tip, "
                              "after Bartels, Meyer and Rieder, Phys. Rev. Lett. 79, 697, 1997)"),
               audits=_audit_p7),
}

N_PER_FAMILY = 3
TASK_ID = re.compile("|".join(f"{f['prefix']}_\\d\\d" for f in FAMILIES.values()))


def family_of(task_id: str) -> str:
    """The family key of a task id. The prefixes overlap (``stm_p3_corral`` is a stem of the repair family's),
    so the match is on the whole id: prefix, an underscore and two digits."""
    for key, f in FAMILIES.items():
        if re.fullmatch(f"{f['prefix']}_\\d\\d", task_id):
            return key
    raise KeyError(task_id)


#: short names accepted on the command line
ALIASES = {"p3r": "p3_repair"}


def parse_families(arg: str | None) -> list[str]:
    """The family keys of a ``--families`` argument (comma-separated; ``p3r`` is ``p3_repair``); all seven if empty."""
    if not arg:
        return list(FAMILIES)
    keys = [ALIASES.get(k.strip(), k.strip()) for k in arg.split(",") if k.strip()]
    bad = [k for k in keys if k not in FAMILIES]
    if bad or not keys:
        raise ValueError(f"unknown family {bad or arg!r}; known: {', '.join(FAMILIES)} (and p3r)")
    return list(dict.fromkeys(keys))


def family_of_path(rel: str) -> str | None:
    """The family a file of the tree belongs to (by the task id or the grader-test name in its path), else None."""
    for part in rel.replace("\\", "/").split("/"):
        name = part[:-5] if part.endswith(".json") else part
        try:
            return family_of(name)
        except KeyError:
            pass
        for key, f in FAMILIES.items():
            if part == f["tests"]:
                return key
    return None


def task_ids() -> list[str]:
    return [f"{f['prefix']}_{n:02d}" for f in FAMILIES.values() for n in range(1, N_PER_FAMILY + 1)]
