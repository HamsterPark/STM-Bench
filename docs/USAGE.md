# STM-Bench usage

Practical reference for the two packages. The design record is `DESIGN.md`; this file
documents what the code does today.

## 1. Environment

| variable | meaning | default |
|---|---|---|
| `STM_BENCH_DATA` | default root for `runs/`, `calib/`, `sessions/`, `index/`, `trackA/`, `fidelity/`, logs — read through `stmsim.paths.data_root()`; explicit CLI paths can override defaults | `~/stm_bench` on every platform |
| `STM_BENCH_CORPUS_INDEX` | read-only index of the real-instrument corpus (private; calibration scripts and Track A manifests) — `stmsim.paths.corpus_index_dir()` | `<data-root>/corpus_index` |
| `STM_BENCH_RAW` | raw instrument-file mirror (`.dat` tree for `stmsim.calibrate.index_dat`; private) — `stmsim.paths.raw_mirror_root()` | `<data-root>/raw` |
| `MAST_ROOT` | MAST checkout; `tests/conftest.py` adds `$MAST_ROOT/MASTv2` to `sys.path`; `stmsim.paths.mast_root()` | the checkout that owns the importable `mast` package |
| `PYTHONPATH` | add `<MAST>/MASTv2` when MAST is not already importable in the active environment; `MAST_ROOT` alone changes the import path only in tests | — |
| `STM_BENCH_INDEX` | **deprecated** old name of `STM_BENCH_CORPUS_INDEX`; still honoured when the new one is unset, with a `DeprecationWarning` | — |
| `MAST_NANONIS_PORT_{MAIN,MONITOR,DATA,EMERGENCY}` | where MAST connects; the harness sets these to the simulator's ports per episode | 6501–6504 |
| `MAST2_PROJECT_ROOT`, `MAST2_USER_ROOT` | MAST's settings/db/artifacts root; the harness isolates each episode under `<run dir>/mast_root` | — |
| `MAST2_API_KEY_DIR` | directory of `*.env` provider keys for LLM modes (else `<MAST repo>/api key`, else the provider's own env var) | — |
| `MAST_VISION_BACKEND` | vision backend the host selects for MAST | harness default |

All four `STM_BENCH_*` / `MAST_ROOT` lookups live in one module, `stmsim/paths.py`
(`stmbench.paths` re-exports it); blank values count as unset and `~` is expanded. Script
`--out` / `--index` / `--root` defaults are computed from it at parse time, so setting the
variables is enough — no script needs an explicit path on the lab machine.

Python 3.13. MAST and the private calibration corpus are not included in this repository.

```bash
python -m pip install -e ".[dev,bench]"
export STM_BENCH_DATA=/data/stm_bench
export MAST_ROOT=/path/to/MAST            # optional; only MAST-backed features need it
```

In PowerShell, use `$env:STM_BENCH_DATA = "$env:USERPROFILE/stm_bench"` and,
if needed, `$env:MAST_ROOT = "C:/path/to/MAST"`. Existing installations that used
the former lab defaults must set the data, corpus and raw-data variables explicitly.

For MAST-backed commands, use an environment with MAST's dependencies and ensure `mast`
is importable. If needed, add the checkout's `MASTv2` directory to `PYTHONPATH`;
setting `MAST_ROOT` alone does not configure the CLI's import path.

## 2. The simulator on its own

```bash
python -m stmsim serve --profile reference-stm --seed 0 --material "Au(111)" \
    --ports 6501,6502,6503,6504 --session-dir "$STM_BENCH_DATA/sessions/serve" \
    --time-scale 1.0 --contamination 0.0 [--approached] [-v]
```

- `--profile` — rig profile YAML in `stmsim/profiles/` (`reference-stm`: 4 K, ±169.5 nm Z,
  10 nA preamp, `z_extend_sign = -1`). Rig numbers live in YAML, never in code.
- `--material` — `Au(111)`, `Cu(111)`, `HOPG` (lattice, herringbone, LDOS template).
- `--ports` — four loopback ports, one per MAST connection role (main / monitor / data /
  emergency). Each connection is served serially; all share one `World`.
- `--session-dir` — where `Scan_Save` writes `.sxm` (MAST looks for the newest file in
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

## 3. Running episodes

```bash
python -m stmbench.cli run --scenario <path | id prefix> [--mode C|A|B0|B1] [--seeds 0,1,2]
       [--out DIR] [--time-scale X] [--params JSON] [--model ID] [--policy default|honeypot]
       [--max-model-calls N] [--max-tool-calls N] [--run RUN_ID]
```

- `--scenario` — a YAML path or a prefix matched against `stmbench/trackB/scenarios/`
  (`B5` runs all three B5 variants; `B5_repair_blunt` one).
- `--mode` — `C` scripted baseline (no LLM, no key); `A` / `B0` / `B1` LLM modes (need
  `--model`). See README for the tool surfaces.
- `--seeds` — comma list; the same seeds are used for every model (paired design).
- `--out` — run root, default `$STM_BENCH_DATA/runs`.
- `--time-scale` — overrides the scenario's (default 20 for most families).
- `--params` — JSON of extra parameters for the family's composite skill (mode C only).
- `--policy` — HITL auto-resolver policy; default `honeypot` for B8, else `default`.
- `--max-model-calls` / `--max-tool-calls` — hard caps on the loop (default 400 each; the
  scenario's sim-hour and command budgets are what should bind).
- `--run` — run id; default UTC timestamp. An existing run directory is refused.

### Mode H: drive the instrument yourself

```bash
python -m stmbench.cli gui [--port 8765] [--out DIR] [--time-scale 20] [--no-browser]
```

opens a local page (`http://127.0.0.1:8765/`) on which a person takes the model's seat.
It is not a separate harness: `stmbench.human.HumanPort` implements MAST's `ChatModelPort`
and is handed to `run_llm_episode` exactly where a provider model goes, so the person gets
the same system prompt, the same tool surface as mode A (stubs for unloaded packs
included), the same `ReportResult` / `ReportTipState` channels, the same budgets polled
after every tool, the same `[DONE]` / `[ABORT]` end markers and the same judge. The clock
protocol holds too: the sim clock stands still while the person thinks and runs only
while a tool executes. The ledger lands under `<out>/<scenario>/seed<N>/H_human_<policy>/`
and `stmbench.report` reads it like any other episode; mode H is never pooled with the
leaderboard (it is a way to *feel* a task, not a human anchor — the design record says
why there is none).

What the page adds that a model does not get: the frames and spectra the instrument
saved, as pictures (`stmbench.human.render`), with a hover read-out of scan-frame
coordinates — the same coordinates `ReportResult` wants and the judge's footprint test
uses (`render.pixel_to_scan_nm` is pinned to `claims.frame_covering` by a test). Everything
else on the page is what the model sees: tolerances and the scenario's notes stay hidden.

Pick a scenario and a seed on the start page, then: choose a tool on the left, fill the
form the tool's own JSON schema generates (dimensioned parameters are SI strings such as
`60n`, as MAST demands from a model), call it, read the result in the middle column, look
at the frames on the right, report each claim through `ReportResult`, and press
**完成 [DONE]**. The verdict shows every claim with your value, the hidden truth and the
evidence check. One episode at a time per process; a second episode in the same process
inherits MAST's process-level state (see the `isolated` note in §7), so restart the command
for a clean run.

### The paper scenarios

These are the benchmark. One file per paper, plus a second variant for P3.

| id | family | what has to be reproduced | budget |
|---|---|---|---|
| `P1_barth1990_au111` | P1 | herringbone stripe period, one orientation with its position, two rotational domains | 2.0 h / 300 k cmds |
| `P2_standing_waves_cu111` | P2 | Cu(111) surface-state band bottom E₀ and effective mass m* | 3.0 h / 400 k |
| `P3_corral_cu111` | P3 | the two lowest resolvable resonances at a quantum corral's centre | 1.5 h / 150 k |
| `P3_corral_repair_cu111` | P3 | the same, after finishing a ring that arrives with 3–6 atoms missing | 2.5 h / 300 k |
| `P4_atom_positioning_cu111` | P4 | any isolated Fe atom moved 4 nm along +x onto its lattice site, the atoms within 10 nm unmoved | 1.5 h / 200 k |
| `P5_sader_jarvis_qplus` | P5 | force minimum, decay length and binding energy from Δf(z) | 1.5 h / 100 k |

All six use `success: {kind: claims_verified}`.

Every paper's tip arrives in a drawn condition (`hidden.tip.condition`: good, blunt, double,
unstable or dirty) that the task never mentions, and in P3–P5 the Fe deposit covers the whole
sample, not just the working area. The tip decides how atoms are grabbed (a blunt tip needs a
lower resistance, a double tip can drag the atom under its other apex, an unsteady one fumbles),
every atom has its own threshold, and pokes, pulses and crashes take the atoms they land on.
In P4 the report says which atom was moved: `target_site` is matched to the moved atom nearest
the reported position, and `bystander_max_shift_nm` (`of_claim: target_site`) is that atom's
neighbourhood.

### Regression families (B1–B9, not on the leaderboard)

| id | family | task | success criterion (`trackB/truth_criteria.py`) |
|---|---|---|---|
| `B1_approach_noisy` | B1 | withdrawn → stable tunnelling | `in_tunnelling` |
| `B2_low_phi_junction` | B2 | judge whether the junction is clean (φ) | `junction_judged` |
| `B3_find_flat_terrace` | B3 | ≥50 nm step-free, damage-free window | `flat_region_found` |
| `B4_atomic_resolution` | B4 | atomic resolution on Au(111) | `atomic_resolution_achieved` (frames counted through MAST's atomic-phase gate) |
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
task: |                         # goals only; never the tip state, never the fault
  ... 用 ReportResult 报告 ...   # the task must name the result channel
budget: {sim_hours: 3.0, wire_cmds: 400000}
hidden:                         # drawn per seed; keys must be in scenario.HIDDEN_SCHEMA
  surface_state:
    e0_ev: [-0.47, -0.41]                 # [lo, hi] uniform; ints on both ends draw an int
    m_star: [0.35, 0.42]
    # r_threshold_kohm: {range: [80.0, 400.0], log: true}   # log-uniform
    # species: {choice: [Fe, Co]}                            # discrete
claims:
  - id: e0_mev
    kind: scalar                # scalar | angle | peak_in_list | position | constraint
    unit: meV
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

`seed` is never in the file. `Scenario.load()` runs `validate()`: a file with `paper` must use
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

### `ReportResult`

The harness injects one tool in every mode:

```
ReportResult(claim_id, value, unit=None, sigma=None, x_nm=None, y_nm=None, note=None)
```

`claim_id` is an enum of that scenario's claim ids. The last report for a claim wins. Positions
are in the **current scan coordinate system**; the judge maps them onto the sample through the
drift recorded with each frame. Nothing is parsed out of free text: a claim never reported is
never reproduced, and every turn's continue message lists the claims still missing.

### Run directory

```
<out>/<scenario_id>/seed<N>/<mode>_<model>_<policy>/<run_id>/     LLM modes
<out>/<scenario_id>/seed<N>/C/<run_id>/                            mode C
├── episode.json      ledger: verdict, truth before/after, budget as enforced, faults,
│                     command counts, wall/sim time, settings snapshot, policy, caps
├── driver.json       LLM modes: turns, tool calls, stub calls, HITL events, diagnosis reports;
│                     every event carries the instrument's sim_s
├── events.json       the world's events (frames, pulses, crashes, atoms moving), in sim time
├── tip_timeline.json the tip at the start and after every change, with the state it left
│                     (the model never sees it; stmbench.replay shows it to an audience)
├── mast_root/        MAST's isolated project root (settings, db, artifacts)
└── session/          .sxm frames the episode saved
```

### Replays for an audience

```bash
python -m stmbench.cli replay <run dir> [--out DIR] [--single-file] [--bare] [--no-truth] [--open]
```

builds a page that plays one episode back for people who have never used an STM. The left
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

### Reports

```bash
python -m stmbench.report.summarize --runs "$STM_BENCH_DATA/gate1" [--csv out.csv] [--md out.md]
```

One row per `episode.json`: scenario, family, seed, mode, model, policy, run id, success,
partial, failed checks, driver outcome, sim seconds vs budget, controller commands, model /
tool calls, HITL asks, stub calls, `diag` (`ReportTipState` vs truth), wall seconds. Paper
episodes add `paper`, `reproduced` and `claims` (verified / total).

The Markdown report leads with the headline table, **paper × mode × model**: n, reproduced
k/n, a Wilson 95 % interval, the mean number of claims verified, and the median sim hours and
cost. The regression families follow it.

### The solvability gate

Before a paper can be put to a model, the scripted baseline has to show the measurement is
reachable with the tools on offer inside the stated budget:

```bash
python -m stmbench.cli run --scenario P1_barth1990_au111 --mode C        --seeds 0,1,2,3,4,5,6,7,8,9 --time-scale 20 --out "$STM_BENCH_DATA/gate_papers"
python -m stmbench.report.summarize --runs "$STM_BENCH_DATA/gate_papers" --md gate.md
```

The target is ≥90 % reproduced. A family below it means the tolerances, the budget or the
baseline's route needs work — **not** that the tolerances should be widened until it passes.
Whatever the baseline's error distribution turns out to be is what the scenario's `notes`
records, and the tolerance is set from that (about twice the p90).

As of 2026-09-08 only P1 comes close (6/10 under real drift; the baseline was edited after that gate and has to be re-measured on an idle machine). What the other four are blocked on, family by
family, is in `docs/DESIGN.md` §9.1; in every case it is the acquisition rather than the
analysis, and each scenario's `notes` says which measurement is missing.

Three things are worth knowing before adding a paper of your own, because each cost a day to
find (`docs/DESIGN.md` §9.2 has the full list):

* **Angles from image analysis are in image coordinates.** Row 0 is the high-y edge of the
  window, so the stage-frame angle is the negative, modulo 180°.
* **Small features need a small, levelled frame.** An adatom is 70 pm on a surface whose steps
  are 208; in a wide survey it is below the segmentation threshold, and `ExtractClusters`
  returns frame-corner artefacts instead. `level="plane"` with `threshold_mad=2.0` on a 20 nm
  frame finds every atom to 0.1 nm.
* **The sample drifts, and the scenario will not slow it down for you.** About 1 nm/min, on
  the sim clock — so a three-hour budget is three hours of drift however fast you run it. Two
  frames of the same window and `MeasureFrameDrift` give the rate; `SetDriftCompensation` takes
  the velocity you measured and nulls it; then measure again, because the wrong sign doubles
  the drift rather than doing nothing. Compensation cancels the steady part only, so keep
  re-registering on your own feature between steps. And you can only track what your feature
  constrains: a straight step edge tells you nothing along its own length, so track its normal.
* **A parsed controller reply is `(error, raw_bytes, [values])`.** The numbers are one level down
  and name lists are two. Scanning only the top level silently returns nothing.

The whole matrix, once there are models to run:

```bash
python -m stmbench.cli gate --families P1,P2,P3,P4,P5 --modes C,A --models <m1,m2,...>        --seeds 0,1,2 --gate papers1 --parallel 2
```

## 4. Thresholds and calibration

These read the private corpus index (`STM_BENCH_CORPUS_INDEX`; `index_dat` reads the raw
mirror `STM_BENCH_RAW`) and write to `$STM_BENCH_DATA/calib/` (`index_dat`:
`$STM_BENCH_DATA/index/`). Derived constants are recorded in the source; the private
input corpus and generated calibration reports are not bundled. Every option below is shown with its default — with the
variables set, the bare `python -m …` form does the same thing.

```bash
python -m stmbench.trackB.derive_thresholds --seeds 8 --out "$STM_BENCH_DATA/calib/thresholds.json" \
       --index "$STM_BENCH_CORPUS_INDEX/sxm_index_full.parquet"
python -m stmsim.calibrate.fit_creep       --csv "$STM_BENCH_CORPUS_INDEX/glance_drift.csv" --out "$STM_BENCH_DATA/calib/creep.json"
python -m stmsim.calibrate.working_points  --index "$STM_BENCH_CORPUS_INDEX/sxm_index_full.parquet" --out "$STM_BENCH_DATA/calib/working_points.json"
python -m stmsim.calibrate.index_dat       --root "$STM_BENCH_RAW" --out "$STM_BENCH_DATA/index/dat_index.parquet"
python -m stmsim.calibrate.hysteresis      --out "$STM_BENCH_DATA/calib/hysteresis.json"
python -m stmsim.calibrate.lambda_from_rowjump --out "$STM_BENCH_DATA/calib/lambda.json"
python -m stmsim.spec.extract --out "$STM_BENCH_DATA/command_registry.json"  # exports the local registry; no MAST needed
```

The simulator profiles are `reference-stm` and `reference-stm-qplus`; use these names
in commands and scenario files. Calibration input names are release-neutral: use `sxm_sequence.parquet` for the sequence
index, group DAT files by their immediate child directory when using `--root`, and place
tip-shaping traces under `<data-root>/calibration_inputs/poke_traces/` or pass an explicit
`--traces` directory. Older private corpus names are not required by the public checkout.

- σ* = 0.09 nm (apex smearing at which a flat Au(111) terrace stops passing MAST's
  atomic-phase gate: 10/10 vs 0/10 seeds); λ* = 3.5e-4 /s (task-derived tip-change
  hazard); φ ≥ 3.0 eV is MAST's own barrier-height acceptance.
- Creep/drift constants: 983 reference glance intervals (`stmsim/profiles/reference-stm.yaml: drift`).

## 5. Harness pieces (for extending)

| module | role |
|---|---|
| `stmbench/harness/runtime_host.py` | starts the simulator, exports the port variables, isolates `MAST2_PROJECT_ROOT`, seeds MAST settings (current monitor on Osci2T, tool refine off…), registers tip / experiment / vacuum attestation / coarse-drive ceiling, hosts `CoreRuntime.setup()` |
| `stmbench/harness/episode.py` | one episode end to end; run-dir claim; budget caps; verdict incl. budget overrun rule (5 % tolerance, partial ≤ 0.5) and B4 frame counting |
| `stmbench/harness/ic_driver.py` | multi-turn unattended driver of MAST's instrument-control loop; `ClockPausedPort`; `ReportTipState`; `[DONE]`/`[ABORT]` end-of-reply rule |
| `stmbench/harness/bind_all_names_mw.py` | every tool name bound on every call; stub calls load the pack and ask for a retry |
| `stmbench/harness/modes.py` | tool surfaces A / B0 / B1 / C (level ≤ 2 and not instrument-driving = primitive) |
| `stmbench/harness/hitl_autoresolver.py` | answers `ask_user` / dangerous-action / workflow questions under `default` or `honeypot` |
| `stmbench/human/` | mode H: `port.py` (a person as the `ChatModelPort`, shared session state), `server.py` (the local web GUI, `stmbench.cli gui`), `render.py` (frames / spectra as PNG, pixel → scan-frame nm) |
| `stmbench/replay/` | audience replays (`stmbench.cli replay`): `timeline.py` (ledger → one sim-time timeline; clock refit for old ledgers), `lay.py` (plain-language tool calls, tip, papers, claims), `truth_view.py` (true sample rebuilt from scenario + seed: perfect-tip frames, overview), `build.py` + `static/player.html` (the page) |
| `stmbench/trackB/truth_criteria.py` | verdict functions per family, thresholds table |
| `stmsim/scenario.py`, `stmsim/faults/` | YAML → world; fault scheduler |
| `stmbench/harness/host_trial.py`, `forge_trial.py` | manual trials of one skill under the host (`--root`, `--skill`) |
| `stmbench/harness/probe_*.py` | provider probes (tool-count cliff, forced-call name constraint); throw-away |

Adding a family: a YAML in `trackB/scenarios/`, a criterion in `truth_criteria.CRITERIA`,
and (mode C) the family → composite mapping in `episode.py`.

## 6. Tests

```bash
python -m pytest tests -q -m "not requires_mast and not requires_data"  # public checkout
python -m pytest tests -q -m "not requires_mast"      # CI (scratch data root absent)
python -m pytest tests/test_codec.py -q               # wire codec vs MAST's client (needs MAST)
python -m pytest tests/test_mast_skills_e2e.py -q     # MAST's real skills on the simulator (needs MAST)
```

`tests/test_no_machine_paths.py` scans `stmsim/` and `stmbench/` (probes included) for
machine-specific lab paths and for direct `STM_BENCH_*` / `MAST_ROOT` environment
reads outside `stmsim/paths.py`; either fails the suite with a file:line list.

## 7. Gotchas

- Any single controller call must answer within 5 s; a server that sleeps past that trips
  the client's receive timeout and MAST's shared circuit breaker (all four roles, 20 s).
  B9 uses this deliberately.
- Live scan buffers (`Scan_FrameDataGrab`) report unscanned rows as **0**, saved `.sxm`
  files as NaN — MAST's crash check depends on the former.
- MAST's runtime takes over stdout once it is up; write trial conclusions to files
  (`host_trial` persists `truth_after.json`).
- Parameters with units are strings (`"50m"`, `"20 mV"`).
- The simulator's Z sign, ranges and preamp full scale come from the rig profile; all Z
  readings (`ZCtrl_ZPosGet`, `.sxm`, `z_offset`) share one definition.


### Tests with an isolated runtime

`tests/test_tilt.py::test_mast_tilt_calibrate_and_auto_tilt_level_the_sim` carries the
`isolated` marker: its `RuntimeHost` must be the only one in its process. MAST has
process-level singletons; a second host inherits state from the first. In the trial that
motivated this marker, AutoTilt measured 0.006° on a 0.29° slope. With MAST available,
run the groups in separate processes:

```bash
python -m pytest tests -q -m "not isolated"
python -m pytest tests -q -m isolated
```

The first command includes `slow` tests; add `and not slow` to its marker expression for
a shorter run. CI does not install MAST and deselects `requires_mast`, so it does not
need the second invocation.
