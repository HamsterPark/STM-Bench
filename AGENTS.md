# AGENTS.md — repository guide

This file guides code review and changes throughout the repository. For a read-only
review, start with the project summary and source/test map below. Before changing code,
also read the invariants and validation guidance.

## Project and scope

STM-Bench asks: **can an AI operate a simulated scanning tunnelling microscope well enough
to reproduce the measurements in a classic STM paper?** Instrument operation is the task;
data analysis supports it.

- **`stmsim`** is the instrument: hidden tip condition, drift, feedback dynamics,
  preamplifier saturation, spectroscopy, and atom manipulation whose outcome depends on
  junction resistance, speed, tip state and local environment. Controller modules expose
  the physics through an SPM-controller TCP protocol.
- **`stmbench`** is the experiment around it: six scenarios in five paper families,
  per-seed hidden targets, a judge that checks both reported results and acquisition
  evidence, episode budgets and records, the MCP instrument, reference solutions and scripted
  baselines, and replay tools.

The central design choice is that **a correct number without the required measurement
does not count as a reproduction**. Per-seed targets also make a fixed literature answer
insufficient. Follow that contract from a scenario through the result channel to the judge.

This is an **alpha research prototype**. Keep these evidence boundaries explicit:

- The simulator runs independently, and full paper episodes run through the benchmark
  instrument (`stmbench.instrument`, served to any MCP agent). Agent harnesses and control
  runtimes are evaluated through that interface; none of them is part of this repository.
  This checkout does not include the private calibration corpus or historical run ledgers.
- No leaderboard has been published. Historical scripted gates and operator-console trials
  predate the current physics and have not been rerun on this release. They are development
  observations, not current reproduction rates.
- Protocol compatibility is an interface goal, not equivalence to real hardware or support
  for an entire controller catalog. Some parameters come from lab fits, others from literature
  or modeling assumptions; implementation and internal tests alone do not establish physical fidelity.
- Acquisition evidence verifies that the required measurements occurred; it does not prove
  that the reported number was calculated from those measurements.

## Where to start a review

1. Read [README.md](README.md) for the public overview and release status.
2. Trace one task end to end: [P1 scenario](stmbench/trackB/scenarios/P1_barth1990_au111.yaml)
   → [world construction](stmsim/scenario.py) → [structured result channel](stmbench/trackB/results.py)
   → [claim and evidence checks](stmbench/trackB/claims.py)
   → [verdict integration](stmbench/trackB/truth_criteria.py).
3. Use the map below to inspect a design decision alongside its regression tests.
   [docs/DESIGN.md](docs/DESIGN.md) records the rationale and historical experiments in Chinese;
   [docs/FIDELITY.md](docs/FIDELITY.md) lists what the simulator models, its source and its known differences;
   [docs/USAGE.md](docs/USAGE.md) covers operation;
   [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records external components and protocol provenance.

| Question to inspect | Implementation | Tests to read alongside it |
|---|---|---|
| Does scoring require measurement, not just a plausible answer? | [claims.py](stmbench/trackB/claims.py), [results.py](stmbench/trackB/results.py) | [test_claims.py](tests/test_claims.py), [test_papers.py](tests/test_papers.py) |
| Are seeded worlds stable, and does manipulation respond to instrument state? | [scenario.py](stmsim/scenario.py), [world.py](stmsim/physics/world.py), [adatoms.py](stmsim/physics/adatoms.py) | [test_scenario_hidden.py](tests/test_scenario_hidden.py), [test_manipulation_realism.py](tests/test_manipulation_realism.py) |
| Are binary contracts preserved across the client/server boundary? | [codec.py](stmsim/wire/codec.py), [controller modules](stmsim/modules/), [command registry](stmsim/spec/command_registry.json) | [test_codec.py](tests/test_codec.py), [test_world_smoke.py](tests/test_world_smoke.py); no client oracle is bundled |
| Are instrument time and the budget kept on the virtual clock? | [clock.py](stmsim/physics/clock.py), [instrument/core.py](stmbench/instrument/core.py), [rules.py](stmbench/trackB/rules.py) | [test_clock_freeze_and_faults.py](tests/test_clock_freeze_and_faults.py), [test_budget_overrun.py](tests/test_budget_overrun.py), [test_instrument.py](tests/test_instrument.py) |
| Can a result be traced to its acquisition, coordinates and tip history? | [instrument/core.py](stmbench/instrument/core.py), [render.py](stmbench/render.py), [replay/](stmbench/replay/) | [test_instrument.py](tests/test_instrument.py), [test_render.py](tests/test_render.py), [test_replay.py](tests/test_replay.py) |
| Do comparisons respect shared seeds and missing observations? | [stats.py](stmbench/report/stats.py) | [test_stats.py](tests/test_stats.py) |
| Is every judged rule stated in the task, and can a fixed answer pass? | [rules.py](stmbench/trackB/rules.py), [audit.py](stmbench/trackB/audit.py) | [test_rule_card.py](tests/test_rule_card.py), [test_rules_v1.py](tests/test_rules_v1.py), [test_audit.py](tests/test_audit.py) |
| Does an episode replay to the same measurements? | [clock.py](stmsim/physics/clock.py), [instrument/](stmbench/instrument/) | [test_instrument.py](tests/test_instrument.py), [test_mcp_server.py](tests/test_mcp_server.py), [test_determinism_probe.py](tests/test_determinism_probe.py) |
| Can a paper be reproduced by measurement, how precisely, in how much time? And without the truth? | [reference/](stmbench/reference/), [baselines/](stmbench/baselines/), [analysis/](stmbench/analysis/) | [test_reference_p5.py](tests/test_reference_p5.py), [test_reference_p3.py](tests/test_reference_p3.py), [test_reference_p4.py](tests/test_reference_p4.py), [test_baselines_p4.py](tests/test_baselines_p4.py), [test_analysis_tip.py](tests/test_analysis_tip.py), [test_analysis_force.py](tests/test_analysis_force.py), [test_analysis_drift.py](tests/test_analysis_drift.py), [test_analysis_spectra.py](tests/test_analysis_spectra.py), [test_analysis_adatoms.py](tests/test_analysis_adatoms.py), [test_analysis_corral_gap.py](tests/test_analysis_corral_gap.py) |

Other useful boundaries:

- [stmsim/physics/](stmsim/physics/) implements the tip, surface, scanner, junction,
  feedback, surface states, corrals and qPlus/force channels; [profiles/](stmsim/profiles/)
  holds rig parameters. [io/](stmsim/io/) writes `.sxm` and `.dat` files.
- [stmsim/calibrate/](stmsim/calibrate/) fits lab data;
  [stmsim/validate/](stmsim/validate/) compares simulated and real measurements.
  Their external data requirements differ from the standalone physics tests.
- [stmbench/agent/](stmbench/agent/) seats a language model at the instrument with its MCP tools
  and nothing else (`stmbench.cli agent`; tests in [test_agent.py](tests/test_agent.py)).
- [stmbench/reference/](stmbench/reference/) holds the privileged reference solutions and
  [stmbench/baselines/](stmbench/baselines/) the scripted baselines, which see only what an
  MCP agent sees.
- `P1`–`P5` are the paper families; `B1`–`B9` are skill scenarios kept as simulator fixtures
  (the instrument refuses them). [trackA/](stmbench/trackA/) is retained development work
  outside the current scope.

In a review or report, distinguish what the code implements, what the checks you ran
establish, what historical records describe, and what remains unverified. Cite the relevant
source or test; preserve the date and validation scope of any reported result.

## Run and verify

Use Python **3.13** for the CI-tested environment (`pyproject.toml` declares `>=3.13`).
From the repository root, in a virtual environment:

```bash
python -m pip install -e ".[dev,bench]"
python -m stmsim serve --help
python -m stmbench.cli --help
```

A focused first check of scoring, seeded targets and statistical reporting:

```bash
python -m pytest tests/test_claims.py tests/test_papers.py tests/test_scenario_hidden.py tests/test_stats.py -q
```

The full suite, without private calibration data or model API keys (tests that need the
data skip themselves):

```bash
python -m pytest tests -q
```

[CI](.github/workflows/tests.yml) runs Python 3.13 entry-point checks and the tests,
leaving its scratch data root absent so data-dependent tests skip. A passing suite does
not validate autonomous-agent reproduction rates. Read skip/deselection counts with the
passing count; do not turn an unavailable check into a successful one.

To operate the standalone simulator:

```bash
python -m stmsim serve --profile reference-stm --seed 0 --material "Au(111)"
```

This starts a local controller server; connect a compatible TCP client and stop with Ctrl+C.
Full paper episodes also run through the standalone benchmark instrument. An MCP-capable
agent launches the episode server below, calls `get_task` first, and operates the instrument
through its tools. After the episode ends, `verify` replays the recorded actions and checks
the measurements and verdict. Use a new output directory for each episode.

```bash
python -m stmbench.cli episode --scenario P4_atom_positioning_cu111 --seed 0 --out runs/p4_s0
python -m stmbench.cli verify runs/p4_s0
```

Reference solutions and scripted baselines run on the same instrument, without a model:

```bash
python -m stmbench.cli reference --paper P4 --seeds 30
python -m stmbench.cli baseline --paper P4 --seeds 30 --first-seed 450
```

Replays read files with `stmsim.io.sxm_reader`; they take an instrument run directory (see
[docs/USAGE.md](docs/USAGE.md)):

```bash
python -m stmbench.cli replay "<run-directory>" --single-file --open
```

`requires_data` marks tests that read `STM_BENCH_DATA`; the check tests that the root exists,
not that the corpus is complete.

## Invariants to preserve

These apply across the simulator, the instrument, the judge and the presentation layers.

1. **Keep random streams independent.** Layout, dynamics, atom traits, the atom field,
   hidden targets, domains and noise have separate seeded streams. New stochastic mechanisms
   need their own stream; consuming extra draws from an existing stream can silently change
   existing worlds. Preserve the stream keys and seed behavior when refactoring.
2. **Treat hidden-schema edits as experimental changes.** `HIDDEN_SCHEMA` in
   `stmsim/scenario.py` is a whitelist. `resolve_hidden` uses one RNG across sorted blocks
   and keys: adding a random-valued entry can shift later draws **across blocks as well as
   within a block**. Changing a scenario ID also changes its hidden stream. Recheck affected
   seeds and tolerance calibration; sorting makes YAML order irrelevant, not schema changes harmless.
3. **Keep episode truth out of the agent's observations.** `World.truth()`, hidden draws,
   tolerances and diagnostic notes serve the judge and post-episode records. Do not expose
   them through the task, tool replies or any page an agent sees. Replay may reveal truth
   after the episode. The reviewer may inspect source and ledgers; the evaluated agent
   must obtain its answer through the instrument.
4. **Preserve the measurement contract.** Paper results arrive through `ReportResult`,
   not numbers extracted from prose. Reported-value claims need both the report and required
   acquisition evidence; missing reports fail. State constraints are checked directly against
   truth and required evidence. Do not widen tolerances, remove evidence requirements or
   reduce paper-scenario drift to make a baseline pass. Every rule the judge applies is stated
   to the agent by the rule card (`stmbench.trackB.rules`), generated from the same scenario
   fields: never state a budget or an evidence rule by hand in a task, and never add a judge
   rule the card does not state. Under rules version 1 a report names the frame its position
   was read in (`frame`) and the data its value came from (`evidence`), and the evidence rule
   is checked within that data; ledgers record `rules_version`, and older ledgers keep their rules.
5. **Respect clocks and budgets.** Hardware-side processes use the wall clock; scans,
   drift and creep use simulated time, and instrument-time budgets use sim hours.
   Preserve this distinction and the overrun rule. On the host clock (the controller TCP
   path) host load changes elapsed instrument time and can change the outcome; the benchmark
   instrument (`stmbench.instrument`) runs on the `VirtualClock` instead: every action
   advances it by its physical duration, observations neither spend time nor draw random
   numbers, and an episode must replay to the same measurements (`stmbench verify`). Never
   read the host clock or a shared random stream from an observation.
6. **Keep coordinate frames explicit.** The agent reports scan-frame coordinates; the
   sample lives in the sample frame. Saved frames record the drift offset the renderer added
   to every acquisition row (`row_offsets_nm`) and the lateral offset of the apex that imaged
   it (`apex_runs_nm`, where it changes), and the judge maps a reported position through both,
   for the row it falls in, in the frame the report names. In oriented arrays, row 0 is the high-v edge.
   `stmbench.render.pixel_to_scan_nm` must remain the inverse of the judge's footprint mapping
   (`tests/test_render.py`). Preserve units at every boundary.
7. **Respect scan-time history.** A frame is rendered whole at scan start and revealed
   row by row. Spontaneous tip events carry the time of the row they affect; `scan_start`
   records which events its render drew. Mid-frame `world.tip` can already describe a state
   the visible image has not reached. Replay must use recorded event times and states.
8. **Preserve run records.** Use a new output directory for each episode. Do not overwrite
   completed runs or hand-edit them to improve a verdict. `episode.json`, `actions.jsonl`,
   `events.json`, `report.json` and `session/` feed later analysis, `verify` and replay. Schema changes must
   account for existing readers; missing observations remain unknown, not zero or success.
9. **Preserve binary contracts and their provenance.** Command names, field widths, array
   counts and reply layouts are client contracts. Do not infer widths from unannotated Python
   handlers or rename commands. Check the bytes against the documented layout.
   A local registry is not evidence of a clean-room implementation or full catalog coverage.
10. **Keep physical assumptions traceable.** Distinguish empirical fits, literature relations
    and modeling assumptions. When changing a fitted constant, retain or add what was measured,
    when, and the sample count. For other parameters, document the source or rationale and
    the validation that supports the change. Comments should explain why.

## Making a change

- Read the relevant implementation and tests before editing. Keep a change focused; avoid
  coupling a scientific correction with unrelated cleanup or dependency changes.
- For a behavioral fix, add a regression test that demonstrates the failure and run the
  affected tests. Use the full suite for changes crossing subsystem boundaries;
  use the data-dependent checks when the data is available. For documentation
  alone, verify links, commands and factual claims against source. Report what was run and
  what could not be checked.
- Use [stmsim/paths.py](stmsim/paths.py) for environment-derived paths. Default outputs live
  under `STM_BENCH_DATA` (default `~/stm_bench`); explicit CLI output/session paths may override
  them. Keep machine-specific paths and private data out of package source.
  [test_no_machine_paths.py](tests/test_no_machine_paths.py) guards drive-rooted path literals
  and duplicate environment readers.
- Use neutral instrument and dataset aliases in public examples and test fixtures. Keep
  personal paths, private host addresses, internal sample labels and raw acquisition records
  outside the repository. Preserve scientific citations, calibration dates, sample counts
  and uncertainty when editing comments; use factual technical prose.
- Keep the simulator's name vendor-neutral. Preserve required compatibility API and file-format
  tokens allowed by that same guard; do not erase protocol provenance from documentation.
- Write code, comments and docstrings in English. Chinese belongs in scenario task text,
  the replay page, and `docs/DESIGN.md`.
- Update `docs/DESIGN.md` when a design decision changes. Keep operational instructions in
  `docs/USAGE.md` and release claims in `README.md` consistent with the implementation.
  When committing, use one logical change per commit, a present-tense subject, and a body
  explaining why with the test, measurement or trial that motivated it.

## Adding a paper scenario

1. Add a YAML under `stmbench/trackB/scenarios/` with `paper`, `initial`, `hidden`,
   `claims`, `budget` and `task`. Hidden ranges need physical justification; each claim
   needs its kind, truth reference, acceptance threshold or tolerance, acquisition-evidence
   rule and a `say` line (its meaning, for the rule card). Require a report where the claim
   kind calls for one. Mark `knowledge: allowed` only on a material constant.
   Run `python -m stmbench.cli audit chance --scenario <id>` and record the per-claim and
   joint chance in the scenario notes.
2. Implement the required physics and expose the truth through `World.truth()` or an
   existing judge callback such as `orientation_at`. Keep the agent-facing task free of truth.
3. Reuse `claims_verified` unless it cannot express the criterion; only then extend
   `truth_criteria.CRITERIA`.
4. Add a privileged reference solution in `stmbench/reference/` to investigate solvability and
   set tolerances, and a scripted baseline in `stmbench/baselines/` that sees only what an agent
   sees. Record their actual validation status; adding a baseline does not establish that it passes.
5. Test scenario loading, truth resolution, seed variation, evidence rejection and the
   failure of frozen answers across seeds where applicable. State-based manipulation
   claims need tests of the actual movement and unaffected neighbours instead.
