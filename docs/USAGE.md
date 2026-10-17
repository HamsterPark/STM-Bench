# STM-Bench usage

Practical reference for the two packages. The design record is `DESIGN.md`; this file
documents what the code does today.

## 1. Environment

| variable | meaning | default |
|---|---|---|
| `STM_BENCH_DATA` | default root for `runs/`, `calib/`, `sessions/`, `index/`, `trackA/`, `fidelity/`, logs — read through `stmsim.paths.data_root()`; explicit CLI paths can override defaults | `~/stm_bench` on every platform |
| `STM_BENCH_CORPUS_INDEX` | read-only index of the real-instrument corpus (private; calibration scripts and Track A manifests) — `stmsim.paths.corpus_index_dir()` | `<data-root>/corpus_index` |
| `STM_BENCH_RAW` | raw instrument-file mirror (`.dat` tree for `stmsim.calibrate.index_dat`; private) — `stmsim.paths.raw_mirror_root()` | `<data-root>/raw` |
| `STM_BENCH_INDEX` | **deprecated** old name of `STM_BENCH_CORPUS_INDEX`; still honoured when the new one is unset, with a `DeprecationWarning` | — |

All three `STM_BENCH_*` lookups live in one module, `stmsim/paths.py`
(`stmbench.paths` re-exports it); blank values count as unset and `~` is expanded. Script
`--out` / `--index` / `--root` defaults are computed from it at parse time, so setting the
variables is enough — no script needs an explicit path on the lab machine.

Python 3.13. The private calibration corpus is not included in this repository.

```bash
python -m pip install -e ".[dev,bench]"
export STM_BENCH_DATA=/data/stm_bench
```

In PowerShell, use `$env:STM_BENCH_DATA = "$env:USERPROFILE/stm_bench"`. Existing
installations that used the former lab defaults must set the data, corpus and raw-data
variables explicitly.

## 2. The simulator on its own

```bash
python -m stmsim serve --profile reference-stm --seed 0 --material "Au(111)" \
    --ports 6501,6502,6503,6504 --session-dir "$STM_BENCH_DATA/sessions/serve" \
    --time-scale 1.0 --contamination 0.0 [--approached] [-v]
```

- `--profile` — rig profile YAML in `stmsim/profiles/` (`reference-stm`: 4 K, ±169.5 nm Z,
  10 nA preamp, `z_extend_sign = -1`). Rig numbers live in YAML, never in code.
- `--material` — `Au(111)`, `Cu(111)`, `HOPG` (lattice, herringbone, LDOS template).
- `--ports` — four loopback ports, one per client connection role (main / monitor / data /
  emergency). Each connection is served serially; all share one `World`.
- `--session-dir` — where `Scan_Save` writes `.sxm` (a client looks for the newest file in
  `Util_SessionPathGet`).
- `--time-scale` — sim seconds per wall second for scan lines and slow physics. Hardware-
  side processes (tip shaper, bias pulse, oscilloscope screens, auto-approach cycles,
  spectroscopy) always run at 1× wall time because clients sample them with the wall clock.
- `--approached` — start in tunnelling instead of withdrawn.
- `--contamination` — surface contamination density (affects λ, the tip-change hazard).

The banner prints the bound ports and `implemented_verbs`. Commands that are not
implemented, or that belong to a module the rig profile declares "not loaded", return a
controller-style error whose text contains `NeedModule`; every command answers in under 5 s
(clients' receive timeout). Stop with Ctrl-C (graceful; ports are released).

In-process use (tests, batch rendering):

```python
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.modules import build_dispatcher
from stmsim.wire.server import WireServer

world = World(rig=RigProfile.load("reference-stm"), seed=0, session_dir=".../session")
srv = WireServer(build_dispatcher(world), ports=[0, 0, 0, 0]).start()   # 0 = ephemeral
print(srv.bound_ports, world.truth())
srv.stop()
```

Scenario objects build a world from YAML:
`stmsim.scenario.Scenario.load(path).build_world(seed, session_dir)`.

This TCP path runs on the host clock (sim time = wall time × `--time-scale`), so host speed
changes what a session measures; it is an interface for control software, not a judged
benchmark condition. Benchmark episodes run through the instrument (§3).

## 3. Episodes

### The benchmark instrument

`stmbench.instrument` runs a paper scenario on the simulator without the wire protocol, on a
**virtual clock**: instrument time moves only when an action advances it, by the
action's physical duration (a frame takes pixels × 2 × line time; a spectrum its points × (settle
+ integration) plus overheads). With the same implementation and numerical environment,
the same seed and actions reproduce the measurements and verdict independently of host
waiting time. An episode can be re-checked from its ledger.

```python
from stmbench.instrument import Instrument
inst = Instrument("P5_sader_jarvis_qplus", seed=0, out_dir="runs/p5_s0")
print(inst.task())                      # the task with its rule card
inst.act("pll", on=True, amplitude_pm=50)
inst.act("move_tip", x_nm=1.2, y_nm=-0.4)
r = inst.act("zspec", sweep_nm=0.5, z_offset_nm=0.3, points=256)
inst.spectrum(r["spectrum"])            # read-only: no instrument time, no random draw
inst.report("f_min_pn", -150.0, evidence=[r["spectrum"], "..."])
inst.finish()                           # judged with the version-1 rules; ledger written
```

Actions (`stmbench.instrument.ACTIONS`, each with a JSON schema): `scan`, `set_bias`,
`set_setpoint`, `feedback`, `set_current_range`, `move_tip` (in feedback; at a low junction
resistance it drags what the tip grips), `pulse`, `poke`, `sts`, `zspec`, `pll`, `lockin`
(bias modulation on: every scan also records a dI/dV map at the scan bias, channel `didv`),
`drift_comp`, `withdraw`, `approach`, `coarse_move`, `wait`. Only hard instrument limits apply; there is no
safety gate and there are no composite skills. Observations (`status`, `info`, `list_data`,
`frame`, `spectrum`) spend no instrument time and draw no random number, so how often an agent
looks cannot change what it later measures.

Any MCP agent can take the instrument's seat:

```bash
python -m stmbench.cli episode --scenario P4_atom_positioning_cu111 --seed 0 --out runs/p4_s0   # MCP stdio
python -m stmbench.cli verify runs/p4_s0          # replay the actions, compare every measurement and the verdict
```

A frame observation carries `t_start_s` (the instrument time it began: a shift between two frames
over their time difference is the drift `drift_comp` takes), and a Z spectrum carries `z_abs`, the
loop's Z at every point as the `.dat` file has it (what lines a curve over an atom up with one
over bare surface).

The MCP tools are the actions, the observations (`show_frame` returns a PNG rendered one fixed
way — line-flattened, 0.5–99.5 % contrast, integer enlargement to ≥ 512 px — and
`frame_values` the numbers), `get_task` (task, rule card, manual), `report_result`,
`report_tip_state` and `finish`. The ledger is `episode.json` (verdict, hashes of every
measurement), `actions.jsonl` (every call, refused ones included), `events.json`,
`report.json` and `session/` (the saved `.sxm` / `.dat` files). A scenario whose success kind
the instrument cannot judge (the skill scenarios below) is refused when the episode opens.

### The paper scenarios

These are the benchmark. One file per paper, plus a second variant for P3.

| id | family | what has to be reproduced | budget |
|---|---|---|---|
| `P1_barth1990_au111` | P1 | herringbone stripe period, one orientation with its position, two rotational domains | 3.0 h / 400 k cmds |
| `P2_standing_waves_cu111` | P2 | Cu(111) surface-state band bottom E₀ and effective mass m* | 3.0 h / 400 k |
| `P3_corral_cu111` | P3 | the corral radius and the two lowest resolvable resonances at its centre | 1.5 h / 150 k |
| `P3_corral_repair_cu111` | P3 | finish a ring that arrives with 3–6 atoms missing, then measure its two lowest resolvable centre-spectrum resonances | 2.5 h / 300 k |
| `P4_atom_positioning_cu111` | P4 | any isolated Fe atom moved 4 nm along +x onto its lattice site, the atoms within 10 nm unmoved | 1.5 h / 200 k |
| `P5_sader_jarvis_qplus` | P5 | force minimum, decay length and binding energy from Δf(z) | 1.5 h / 100 k |

All six use `success: {kind: claims_verified}`.

For both P3 variants, the judge accepts two distinct peaks among the three lowest true
centre-LDOS peaks, reported in ascending energy order: the lowest level may appear as a
band-edge shoulder at the available resolution. The measurement variant uses ±10 meV
peak tolerances and a ±0.15 nm radius tolerance. The repair variant currently uses ±25 meV
peak tolerances and ring occupancy ≥0.97, with no radius claim.

Every paper's tip arrives in a drawn condition (`hidden.tip.condition`: good, blunt, double,
unstable or dirty) that the task never mentions, and in P3–P5 the Fe deposit covers the whole
sample, not just the working area. The tip decides how atoms are grabbed (a blunt tip needs a
lower resistance, a double tip can drag the atom under its other apex, an unsteady one fumbles),
every atom has its own threshold, and pokes, pulses and crashes take the atoms they land on.
In P4 the report says which atom was moved: `target_site` is matched to the moved atom nearest
the reported position, and `bystander_max_shift_nm` (`of_claim: target_site`) is that atom's
neighbourhood.

### Skill scenarios (B1–B9, simulator fixtures)

These judge on evidence the instrument does not record (frames passed through an image
gate, wrong actions on the controller wire, the share of a budget watched), so the
instrument refuses them; they remain as worlds for the fault, hidden-variant and clock
tests and as candidates for future tasks.

| id | family | task | success criterion (`trackB/truth_criteria.py`) |
|---|---|---|---|
| `B1_approach_noisy` | B1 | withdrawn → stable tunnelling | `in_tunnelling` |
| `B2_low_phi_junction` | B2 | judge whether the junction is clean (φ) | `junction_judged` |
| `B3_find_flat_terrace` | B3 | ≥50 nm step-free, damage-free window | `flat_region_found` |
| `B4_atomic_resolution` | B4 | atomic resolution on Au(111) | `atomic_resolution_achieved` (frames counted through an atomic-resolution gate) |
| `B5_repair_{blunt,multi,unstable}` | B5 | hidden tip fault → repair to acceptance | `tip_repaired`: one apex, σ_apex < σ*, λ < λ*, φ ≥ 3 eV |
| `B6_sts_clean` | B6 | dI/dV on a clean site | `sts_acquired` |
| `B7_long_watch` | B7 | hours of unattended imaging with faults | `watch_survived` (≥90 % budget consumed, faults fired; idle caps partial at 0.3) |
| `B8_z_at_limit_drift` | B8 | honeypot: the obvious action is wrong | `no_wrong_action_and_progress` (2×2; no crash; tip alive) |
| `B9_comms_latency` | B9 | >5 s latency / disconnect / NeedModule | recovery without destructive retries |

### A scenario file, v2

```yaml
id: P2_standing_waves_cu111
family: P2                      # a paper family starts with P
variant: step_and_adatoms
paper: {id: P2, title: "...", ref: "Nature 363, 524"}
rig: reference-stm
material: Cu(111)               # the task may name the material; it must not name the fault
time_scale: 20
initial:
  approached: true
  bias_v: 0.05
  setpoint_a: 1.0e-9
  tip: {radius_nm: 1.2, apex_sigma_nm: 0.09, lambda_per_s: 1.5e-4}
  surface:
    adsorbate_density_per_um2: 45.0
    surface_state: {material: "Cu(111)"}     # turns the surface state on
  # adatoms: {layout: corral|single, species: Fe}   # turns the adatom registry on
  # drift_scale: 0.3    # scales the thermal drift. NOT a way to make a long measurement
  #                     # passable — tests/test_drift_tracking.py refuses it below 1 on any
  #                     # paper scenario. Use it only to model a different machine.
  # pll: {output_on: false, amplitude_m: 50.0e-12}   # qPlus, needs rig reference-stm-qplus
faults: []
task: |                         # goals only; never the tip state, never the fault,
  ... 用 ReportResult 报告 ...   # never the budget (the rule card states it); it must name
                                # the result channel
budget: {sim_hours: 3.0, wire_cmds: 400000}
hidden:                         # drawn per seed; keys must be in scenario.HIDDEN_SCHEMA
  surface_state:
    e0_ev: [-0.47, -0.41]                 # [lo, hi] uniform; ints on both ends draw an int
    m_star: [0.35, 0.42]
    # r_threshold_kohm: {range: [80.0, 400.0], log: true}   # log-uniform
    # species: {choice: [Fe, Co]}                            # discrete
claims:
  - id: e0_mev
    say: "Cu(111) 表面态的带底能量 E0（相对费米能级）"   # what the claim means, for the rule card
    kind: scalar                # scalar | angle | peak_in_list | position | constraint
    unit: meV
    knowledge: allowed          # only on a material constant: see "The chance audit" below
    tol: {abs: 15}              # abs and/or rel; angle adds mod: 180
    truth: surface_state.e0_mev # a dotted path into World.truth(), or "orientation_at"
    evidence:
      kind: spectra_near_scatterer
      min_n: 8
      min_positions: 6
      near_nm: 8.0
      min_points: 64
      v_lo_max: -0.5
      v_hi_min: 0.1
success: {kind: claims_verified}
notes: |
  Why these tolerances, measured how, on what date.
```

`seed` is never in the file. The task the agent sees is `task` followed by the rule card
generated from the claims and the budget (`stmbench.trackB.rules.task_text`, below).
`Scenario.load()` runs `validate()`: a file with `paper` must use
`claims_verified`, must have at least one claim with an id, kind, truth and evidence, and every
`hidden` key must be in the whitelist — a typo raises rather than silently drawing nothing.
Adding a key to a `hidden` block reorders every seed's draws (the sampler walks keys in sorted
order), so the tolerances have to be re-derived; each scenario's `notes` says so.

### Claim kinds

| kind | passes when | extras |
|---|---|---|
| `scalar` | `|reported − truth| ≤ max(tol.abs, tol.rel·|truth|)` | |
| `angle` | the smallest difference modulo `tol.mod` is within `tol.abs`; a herringbone orientation also accepts either arm at ±α | `distinct_from: {claim, min_sep_deg}` compares the two claims' **truths** |
| `peak_in_list` | the reported value matches one of the first `rank_max` true peaks | `distinct_from` (a different peak), `greater_than: <claim id>` |
| `position` | the registry has the target atom **on** the designated lattice site, and the reported coordinates are within `tol.report_nm` of where the atom really is | needs `position: required` |
| `constraint` | a pure threshold on the truth (`min:` / `max:`), nothing reported | |

### Evidence rules

| kind | needs |
|---|---|
| `frame` | a complete frame with `min(w,h) ≥ min_fov_nm` and `nm/px ≤ max_nm_per_px` |
| `frame_covers` | such a frame whose footprint contains the reported position |
| `frame_covers_after` | the same, taken after the last `after_event`; `point` may be a literal `[x, y]` or a truth path such as `corral.centre_nm`, which is mapped back through each frame's recorded drift |
| `spectra_near_scatterer` | `min_n` spectra at `min_positions` distinct places within `near_nm` of a scatterer, each with `min_points` points covering `v_lo_max … v_hi_min` |
| `sts_at_corral_centre` | as above, but within `max_centre_dist_nm` of the corral centre |
| `zspec_pair` | one Δf curve on an adatom (close enough, no jump, no contact, enough span past the minimum) and one on clean surface |
| `events` | at least `min_n` events of a kind, optionally with a `cause` |

Under the version-1 rules (every episode run with the rule card) a reported claim's rule is
checked **within the data the report cites** (`evidence`), and `frame_covers` /
`frame_covers_after` use the frame the report names (`frame`), which must satisfy the rule,
contain the position and, where `after_event` is set, have been saved after the last such event.
Constraints and `events` rules have nothing to cite and are checked against the whole episode.

### The result channel

Results arrive through one call, `Instrument.report` (the MCP tool `report_result`):

```
report(claim_id, value, unit=None, sigma=None, x_nm=None, y_nm=None,
       frame=None, evidence=None, note=None)
```

`claim_id` is an enum of that scenario's claim ids. The last report for a claim wins. Positions
are in the **scan coordinate system** (x right, y up, the frame of the scan offset); `frame`
names the saved frame they were read in (its `.sxm` file name, or `frame:N`), and the judge maps
them onto the sample through the drift of the acquisition row they fall in — each saved frame
records the offset its renderer added to every row (`row_offsets_nm` in `scan_saved`).
`evidence` lists the frames and spectra the value came from (`.sxm` / `.dat` file names, or
`frame:N`, `sts:N`, `zspec:N`). Nothing is parsed out of free text: a claim never reported is
never reproduced.

### The rule card

`stmbench.trackB.rules.rule_card(scenario)` states, from the scenario file, everything the judge
checks: each claim's meaning (`say`), unit, tolerance, pairwise rules and evidence, the citation
and coordinate conventions, the budget and the 5 % overrun rule. `task_text(scenario)` appends it
to the task; `get_task` and the replay show that text. Tolerances are
public; the values drawn for a seed are not. The ledger records `rules_version`, and the judge
applies the version-1 rules (citations, the named frame) to episodes with `rules_version ≥ 1`;
older ledgers re-judge with the rules they ran under. Never state a budget or an evidence rule
by hand in a task: write it in the claim or the budget, and the card will say it.

### The chance audit

```bash
python -m stmbench.cli audit chance                    # every paper, 200 seeds
python -m stmbench.cli audit chance --scenario P3 --seeds 300 --jobs 4 --json out.json
```

The hidden ranges are public, so a fixed answer near the middle of a range is the attack a
reported value has to withstand. The audit draws the truth for many seeds and finds, with the
judge's own tolerance helpers, the constant answer that passes most often: per claim, and for
the whole paper. Search is exact for one or two claims, and for three claims when they are
independent or only two are linked by pairwise rules (as in the current P3 task). Other cases
use coordinate ascent, yielding a lower bound. Acquisition evidence is assumed satisfied: these are
value-check acceptance rates, not complete episode reproduction rates. Claims marked
`knowledge: allowed` (material constants: P1's period, P2's E0
and m*) are reported but not held to a gate — knowing them is part of the skill. Position and
constraint claims are judged on the atoms themselves and are not audited. On seeds 0–299
(2026-10-12), a fixed answer sheet passed the joint value checks for P3 in 41 % of seeds,
P3's repair spectra in 43 %, P2 in 16 % (all knowledge claims) and P5 in 13 %;
P1's measurement claims together passed 4.3 %. After recalibration on 2026-10-13 against
their reference solutions (below), the joint value-check acceptance rates for P5 and P3
were 4.3 % each.

### Reference solutions

```bash
python -m stmbench.cli reference --paper P5 --seeds 30 --jobs 4 --json p5.json
python -m stmbench.cli reference --paper P3 --seeds 30 --jobs 4
python -m stmbench.cli reference --paper P4 --seeds 30 --jobs 4
```

A privileged reference solution (`stmbench.reference`) may use the truth to navigate — where
the atom is, which nearby spot is bare surface, how far the sample has drifted, whether the
tip needs a poke — but every number it reports is measured through the instrument and
computed by `stmbench.analysis`. Over many seeds it shows whether a paper is
solvable, the error spread of a careful measurement (tolerances are set at about twice its
p90) and the instrument time it takes. P5 (2026-10-13, seeds 0–29): 30/30 reproduced, |error|
p90 5.9 % (F_min), 3.7 % (decay length), 8.7 % (E_b), 354 s of instrument time. P3 (seeds
0–29): 30/30; |error| p90 2.7 / 2.5 meV (the two resonances), 0.038 nm (the ring's radius),
485 s. P4 (seeds 0–149): 150/150 — no atom's threshold read (a resistance ladder from
400 kΩ down), the goal site computed from the lattice the other atoms in a survey give
(`stmbench.analysis.adatoms`); the reported place |error| p90 0.016 nm against the judge's
0.6 nm (0.16 nm before the judge mapped positions through the imaging apex, 2026-10-14);
879 s median, 1428 s p90. Per-seed results: `docs/results/`.

### Scripted baselines

```bash
python -m stmbench.cli baseline --paper P4 --seeds 30 --jobs 4 --json p4-baseline.json
```

A scripted baseline (`stmbench.baselines`) drives the instrument an agent gets over MCP and
nothing else — no truth — and decides everything from what it observes: the tip's
state from how its atoms look (`stmbench.analysis.tip`), which atom to move from a survey, the
drift from atoms found again, whether the atom moved from current spikes and fiducials. It is
the bar an agent should clear. P4 (2026-10-14): 150/150 on seeds 450–599, held out
until the code was final, with 984 s median instrument time (the privileged
reference: 879 s); 300/300 on seeds 0–299, where it was developed. Per-seed results and
the earlier held-out rounds: `docs/results/p4_baseline_seeds_*.json`, DESIGN §9.6.

### Run directory

```
<out>/                the directory given as --out (or Instrument(..., out_dir=...))
├── episode.json      ledger: verdict, claims reported / verified, instrument time and budget,
│                     action count, hashes of every measurement, tip diagnoses, the truth
│                     before and after (for the judge and the replay only)
├── actions.jsonl     every call in order, refused ones included, with its instrument time
├── events.json       the world's events (frames, pulses, crashes, atoms moving), in sim time
├── report.json       the reports, folded: the last report for each claim
├── tip_timeline.json the tip at the start and after every change, with the state it left
└── session/          the .sxm frames and .dat spectra the episode saved
```

`stmbench.cli verify <out>` rebuilds the episode from `actions.jsonl` with the same scenario
and seed and compares every measurement and the verdict.

### Replays for an audience

```bash
python -m stmbench.cli replay <run dir> [--out DIR] [--single-file] [--bare] [--no-truth] [--open]
```

builds a page that plays one episode back for people who have never used an STM. It reads an
instrument run directory — the steps come from the agent's `driver.json` when the run has
one, else from `actions.jsonl` — and run directories in the earlier ledger format. The left
side is what the model saw and did: its frames drawn row by row on the instrument clock, each
tool call retold in plain words, what it said. The right side is what only the audience sees:
the tip as a drawing with a traffic light (sharp, blunt, double, crashed and flickering,
carrying an atom), every frame recomputed over the same pixels by a perfect tip, and a map of
the true sample with the atoms moving and the model's frames drifting across it. A timeline
marks the moments worth stopping at (a crash, a drag, a tip that changed, an answer handed in);
`#t=<seconds>` in the URL opens the replay at that instant of experiment time.

It is built after the episode from the ledger and never writes into the run directory, so it
cannot change a result or reach the model. The true sample is rebuilt from the scenario and
the seed and checked against `truth_before` first; if the scenario file changed since the
run, the page says so instead of showing a different sample. Ledgers written before the
clock stamps and `tip_timeline.json` still replay: the clock is refitted from the world's
events (a few seconds off at most) and the tip between start and end is inferred from pulses,
pokes and crashes — the page labels both. Output goes to `<data>/replays/<name>/` by default
(`index.html` opens from disk); `--single-file` inlines the images into one `.html`.

### Before adding a paper

Three things are worth knowing, because each cost a day to find:

* **Angles from image analysis are in image coordinates.** Row 0 is the high-y edge of the
  window, so the stage-frame angle is the negative, modulo 180°.
* **Small features need a small, levelled frame.** An adatom is 70 pm on a surface whose steps
  are 208; in a wide survey it is below a segmentation threshold, and a cluster finder returns
  frame-corner artefacts instead. A plane-levelled 20 nm frame finds every atom to 0.1 nm.
* **The sample drifts, and the scenario will not slow it down for you.** About 1 nm/min, on
  the sim clock — so a three-hour budget is three hours of drift however fast you run it. Two
  frames of the same window give the rate; `drift_comp` takes the velocity you measured and
  nulls it; then measure again, because the wrong sign doubles the drift rather than doing
  nothing. Compensation cancels the steady part only, so keep re-registering on your own
  feature between steps. And you can only track what your feature constrains: a straight step
  edge tells you nothing along its own length, so track its normal.

## 4. Thresholds and calibration

These read the private corpus index (`STM_BENCH_CORPUS_INDEX`; `index_dat` reads the raw
mirror `STM_BENCH_RAW`) and write to `$STM_BENCH_DATA/calib/` (`index_dat`:
`$STM_BENCH_DATA/index/`). Derived constants are recorded in the source; the private
input corpus and generated calibration reports are not bundled. Every option below is shown with its default — with the
variables set, the bare `python -m …` form does the same thing.

```bash
python -m stmsim.calibrate.fit_creep       --csv "$STM_BENCH_CORPUS_INDEX/glance_drift.csv" --out "$STM_BENCH_DATA/calib/creep.json"
python -m stmsim.calibrate.working_points  --index "$STM_BENCH_CORPUS_INDEX/sxm_index_full.parquet" --out "$STM_BENCH_DATA/calib/working_points.json"
python -m stmsim.calibrate.index_dat       --root "$STM_BENCH_RAW" --out "$STM_BENCH_DATA/index/dat_index.parquet"
python -m stmsim.calibrate.hysteresis      --out "$STM_BENCH_DATA/calib/hysteresis.json"
python -m stmsim.calibrate.lambda_from_rowjump --out "$STM_BENCH_DATA/calib/lambda.json"
python -m stmsim.spec.extract --out "$STM_BENCH_DATA/command_registry.json"  # exports the local registry
```

The simulator profiles are `reference-stm` and `reference-stm-qplus`; use these names
in commands and scenario files. Calibration input names are release-neutral: use `sxm_sequence.parquet` for the sequence
index, group DAT files by their immediate child directory when using `--root`, and place
tip-shaping traces under `<data-root>/calibration_inputs/poke_traces/` or pass an explicit
`--traces` directory. Older private corpus names are not required by the public checkout.

- σ* = 0.09 nm (apex smearing at which a flat Au(111) terrace stops passing an
  atomic-resolution image gate: 10/10 vs 0/10 seeds, 2026-08-28); λ* = 3.5e-4 /s
  (task-derived tip-change hazard); φ ≥ 3.0 eV is the usual clean-junction barrier-height
  acceptance.
- Creep/drift constants: 983 reference glance intervals (`stmsim/profiles/reference-stm.yaml: drift`).

## 5. Module map

| module | role |
|---|---|
| `stmbench/instrument/` | the benchmark instrument: `core.py` (actions, observations, reports, ledger, replay check), `mcp_server.py` (MCP stdio), `render.py` (the PNG a frame observation returns) |
| `stmbench/trackB/` | the scenarios, `rules.py` (rule card, overrun rule), `claims.py` (claim and evidence checks), `results.py` (the result channel's fold), `truth_criteria.py` (verdicts), `audit.py` (chance audit) |
| `stmbench/reference/`, `stmbench/baselines/` | privileged reference solutions; scripted baselines that see only what an agent sees |
| `stmbench/analysis/` | measurement analysis the references and baselines use (adatoms, drift, tip, spectra, force, corral) |
| `stmbench/render.py` | frames and spectra as pictures, pixel → scan-frame nm |
| `stmbench/replay/` | audience replays (`stmbench.cli replay`): `timeline.py` (ledger → one sim-time timeline), `lay.py` (plain-language tool calls, tip, papers, claims), `truth_view.py` (the true sample rebuilt from scenario + seed), `record.py` (clock stamps and the tip timeline a run records), `build.py` + `static/player.html` (the page) |
| `stmbench/report/stats.py` | Wilson CI, paired Δ against a reference, the sampling-vs-sim variance split |
| `stmbench/trackA/` | Track A offline decisions on the private corpus (development work) |
| `stmsim/scenario.py`, `stmsim/faults/` | YAML → world; fault scheduler |

Adding a paper: a YAML in `trackB/scenarios/` with `success: {kind: claims_verified}`, a
reference solution in `stmbench/reference/` to show it is solvable and to set tolerances, and
a scripted baseline in `stmbench/baselines/` for the bar an agent should clear.

## 6. Tests

```bash
python -m pytest tests -q                             # tests that need STM_BENCH_DATA skip themselves
python -m pytest tests -q -m "not requires_data"      # the same tests, deselected instead of skipped
```

`tests/test_no_machine_paths.py` scans `stmsim/` and `stmbench/` (probes included) for
machine-specific lab paths and for direct `STM_BENCH_*` environment reads outside
`stmsim/paths.py`; either fails the suite with a file:line list.

## 7. Gotchas

- Any single controller call must answer within 5 s; a server that sleeps past that trips
  a client's receive timeout (and a client may then open its circuit breaker). B9 uses this
  deliberately.
- Live scan buffers (`Scan_FrameDataGrab`) report unscanned rows as **0**, saved `.sxm`
  files as NaN — a client's crash check may depend on the former.
- Parameters with units are strings (`"50m"`, `"20 mV"`).
- The simulator's Z sign, ranges and preamp full scale come from the rig profile; all Z
  readings (`ZCtrl_ZPosGet`, `.sxm`, `z_offset`) share one definition.
