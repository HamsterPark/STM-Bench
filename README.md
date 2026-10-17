# STM-Bench

**English** · [中文](README.zh-CN.md)

**A simulated scanning tunnelling microscope and a benchmark for experimental intelligence.**

STM-Bench turns landmark scanning-probe experiments into reproducible tests of scientific agency. An agent operates a microscope, establishes useful measurement conditions, locates nanoscale structures, acquires images and spectra, manipulates atoms, and supports its conclusions with recorded experimental evidence.

> Can an AI carry an experiment from an unfamiliar instrument state to a verified scientific result?

The project brings together a stateful physical simulator, six paper-inspired scenarios in five families, an independent instrument interface, and an evidence-aware judge. Full experiments are available to MCP-capable agents through `stmbench.instrument`. Measurement references, seeded variation, and action replay make both the task and its evaluation open to inspection.

**Current research release:** standalone MCP episodes; reproducible virtual instrument time; versioned scoring rules with explicit evidence and coordinate frames; dI/dV imaging; and documented multi-seed reference results for quantum-corral spectroscopy, single-atom positioning, and qPlus force spectroscopy. See [reference results](#measurement-reference-results) and [getting started](#getting-started).

## Why STM is a powerful test of experimental intelligence

Scanning tunnelling microscopy connects perception, physical intervention, and scientific inference at the atomic scale. Tunnelling current is approximately exponential in tip–sample separation; in constant-current imaging, feedback converts the junction's response into a height signal. The electronic structure of the sample and the state of the tip both contribute to the observation. The [Tersoff–Hamann theory](https://doi.org/10.1103/PhysRevB.31.805) provides a foundational account of this relationship.

An STM image is an observation produced by an experiment. Its interpretation depends on bias, current, scan geometry, feedback, and acquisition history. A sharp feature can move because an atom moved, because the sample drifted, or because the tip changed. A spectrum becomes scientifically useful when its location, stabilization conditions, and reference measurement are understood. Successful operation requires the experimenter to choose the next observation that resolves the uncertainty.

| Experimental challenge | What an agent must achieve |
|---|---|
| **Partially observed tip–sample junction** | Infer tip condition from measurements, condition the tip when needed, and confirm recovery. |
| **A moving coordinate system** | Estimate drift and creep, revisit a feature, and connect measurements acquired at different times and scales. |
| **Coupled measurement settings** | Balance field of view, spatial resolution, scan speed, signal quality, and instrument time. |
| **Actions that change the sample** | Find a manipulation regime, follow the atom's response, preserve neighbouring structures, and verify the final state. |
| **Quantitative scientific claims** | Connect geometry, spectra, and physical models to values with units, tolerances, and acquisition evidence. |

These demands make STM a compact, scientifically rich setting for studying active perception, planning under uncertainty, and experimental control. Classical demonstrations of surface reconstruction, electron standing waves, quantum confinement, and atom manipulation provide concrete objectives with physically meaningful success criteria. STM-Bench builds on this experimental tradition to evaluate the complete measurement process.

## Six experiments, five scientific families

Each scenario supplies a task, an instrument-time budget, and a rule card generated from the scoring configuration. Seeded worlds vary relevant geometry, material parameters, tip conditions, and manipulation behaviour. The agent observes the instrument; the evaluator retains the hidden state needed to check the outcome.

| Scenario | Scientific objective | Required result |
|---|---|---|
| **P1 · Au(111) reconstruction** | Image the herringbone reconstruction and locate rotational domains, following [Barth et al.](https://doi.org/10.1103/PhysRevB.42.9307). | Stripe period, a local orientation, and positions/orientations in two distinct domains. |
| **P2 · Cu(111) standing waves** | Recover surface-state dispersion from spatially and energetically resolved measurements, following [Crommie et al.](https://doi.org/10.1038/363524a0). | Band-bottom energy and effective mass, supported by spectra near a scatterer or dI/dV maps at multiple energies. |
| **P3 · Quantum-corral spectroscopy** | Relate the geometry of an Fe corral to its confined electronic states, following [Crommie et al.](https://doi.org/10.1126/science.262.5131.218). | Corral radius and two distinct, ascending low-energy resonances from a spectrum at its centre. |
| **P3-repair · Build, then measure** | Restore a damaged corral before measuring its electronic states. | At least 97% ring occupancy, post-manipulation imaging, and two centre resonances measured after manipulation. |
| **P4 · Single-atom positioning** | Move an isolated Fe atom on Cu(111) by 4 nm along +x to the nearest lattice site while preserving neighbours; an adaptation of [Eigler and Schweizer](https://doi.org/10.1038/344524a0). | Verified target-site occupation, a final position tied to a follow-up image, and neighbour preservation. |
| **P5 · qPlus force spectroscopy** | Acquire atom and background Δf(z) curves and apply [Sader–Jarvis force inversion](https://doi.org/10.1063/1.1667267), inspired by [Huber et al.](https://doi.org/10.1126/science.aay3444). | Short-range force minimum, decay length, and binding energy. |

The scenarios adapt the scientific objectives of these papers to explicit simulator models. P4 uses Fe/Cu(111); the original Eigler–Schweizer experiment used Xe/Ni(110). P3's peak rule accepts two different peaks among the three lowest true resonances, accommodating a lowest level that may appear as a shoulder in the measured spectrum. Its current tolerances are ±10 meV for peaks and ±0.15 nm for radius. P5 uses relative tolerances of 15%, 8%, and 20% for force minimum, decay length, and binding energy. The [scenario files](stmbench/trackB/scenarios/) define the complete, versioned contract; P3-repair retains its own rules and is undergoing reference-protocol calibration.

## From measurement to a verified result

The benchmark's central contract is simple: **a scientific result must be accompanied by the required acquisition evidence.** A completed episode combines instrument operation, structured claims, and a check against the simulated sample.

1. **Read the experiment.** `get_task` provides the objective, scoring rules, and instrument manual. Tolerances and evidence requirements are visible to the agent.
2. **Acquire and act.** The agent controls bias, current, feedback, scan geometry, spectroscopy, drift compensation, tip conditioning, and tip motion. It receives images, numerical observations, and saved data identifiers.
3. **Report with provenance.** `report_result` names the claim, its value and units, and its supporting frames or spectra. Positional reports identify the frame in which the coordinates were measured.
4. **Verify the experiment.** The judge checks values, acquisition conditions, spatial coverage, event ordering, and required sample-state constraints. Complete success requires all claims in the scenario to pass within the episode budget.

Frame provenance matters at STM length scales. The evaluator maps reported coordinates using the drift recorded for the relevant acquisition row. A post-manipulation image must follow the atom's motion; a spectrum must cover the required energy range at the required location. The [version-1 rules](stmbench/trackB/rules.py) expose these conditions directly to the agent.

Acquisition checks establish that the required measurements occurred and that the reported evidence satisfies the task. They do not inspect the agent's internal derivation. Material knowledge is explicitly allowed for selected claims, including P1's period and P2's material parameters; their acquisition requirements still apply. This makes the role of prior knowledge and the role of experimental evidence explicit.

### Reproducible instrument time and action replay

The standalone instrument uses a **virtual clock**. Scans, spectra, motion, and waiting advance time by their physical duration. Reading a saved image or checking status consumes neither instrument time nor random draws. Host speed and model response latency therefore do not change the instrument's evolution between actions.

Each episode records actions, events, reports, measurement hashes, and acquired `.sxm` / `.dat` files. `stmbench verify` reconstructs the episode from its seed and actions, then compares measurements and the verdict. Reproducibility can be checked at the level of the acquired data. Record the checkout revision and software environment alongside each run so that replay uses the same physical model and numerical environment.

## A microscope with consequential physics

`stmsim` models a coupled experimental system: a surface, a changing tip, a scanner, a tunnel junction, feedback, and measurement channels. Instrument settings affect what can be seen and what a subsequent action will do.

| Subsystem | Modelled experimental behaviour |
|---|---|
| **Imaging and tip state** | Constant-current topography; two-dimensional tip-radius blur; sharp, blunt, double, unstable, and dirty tips; electronic noise; stochastic tip changes and conditioning. |
| **Scanner and feedback** | Drift and creep during acquisition, active drift compensation, feedback lag, and scan-speed-dependent displacement of apparent features. |
| **Electronic spectroscopy** | Cu(111) surface-state dispersion, scattering at steps and adatoms, T-matrix corral resonances, and integration-time-dependent spectral noise. |
| **dI/dV imaging** | Bias-dependent lock-in contrast acquired with a scan, enabling spatial and energy-resolved standing-wave measurements. The map stores lock-in X current; spectroscopy reports dI/dV in A/V. |
| **Atom manipulation** | Lattice-constrained motion controlled by junction resistance, speed, tip condition, and atom-specific thresholds; slips, pickup, recovery, and local debris obstacles. |
| **qPlus spectroscopy** | Finite-amplitude frequency shifts from short-range Morse and long-range van der Waals forces; noise dependent on amplitude, integration time, and averaging; force reconstruction from measured curves. |

The [fidelity card](docs/FIDELITY.md) connects each behaviour to its literature basis, laboratory fit, or modelling assumption, together with its checks and remaining scope. For example, current dI/dV maps model energy-dependent LDOS contrast without setpoint effects, modulation broadening, or a structured tip DOS. The qPlus extension supports force spectroscopy with constant-current positioning; constant-Δf imaging is outside the implemented scope. This traceable physical specification is part of the benchmark's scientific foundation.

## Measurement reference results

Reference protocols establish **task solvability, achievable measurement precision, and instrument-time cost**. They are executable protocols in [`stmbench.reference`](stmbench/reference/) with analysis routines in [`stmbench.analysis`](stmbench/analysis/).

These protocols have privileged access for navigation and operational decisions—for example, locating an atom, selecting bare surface, tracking drift, and deciding when to condition the tip. Their reported physical values come from instrument acquisitions and numerical analysis. The results below characterize this measurement reference. Autonomous-agent performance is evaluated separately.

**Documented calibration results, 13 October 2026 (P4 rerun 14 October):**

| Reference protocol | Seeds | Complete task success | Median instrument time | 90th-percentile measurement error |
|---|---:|---:|---:|---|
| **P3 · Corral geometry and resonances** | 0–29 | **30/30** | **485 s** | Peak energies: **2.7 / 2.5 meV** absolute error; radius: **0.038 nm** absolute error |
| **P4 · Single-atom positioning** | 0–149 | **150/150** | **879 s** | Reported position: **0.016 nm** error |
| **P5 · Force spectroscopy** | 0–29 | **30/30** | **354 s** | Absolute relative errors: **5.9%** in force minimum, **3.7%** in decay length, **8.7%** in binding energy |

The calibration records are retained with the [P3](stmbench/trackB/scenarios/P3_corral_cu111.yaml), [P4](stmbench/trackB/scenarios/P4_atom_positioning_cu111.yaml), and [P5](stmbench/trackB/scenarios/P5_sader_jarvis_qplus.yaml) scenarios and in the [design record](docs/DESIGN.md). Per-seed results for every row are in [`docs/results/`](docs/results/); the commands below regenerate them for a specified checkout. Since 14 October the judge maps a reported position through the tip apex that imaged it as well as through drift. Before that, P4's 90th-percentile position error was 0.16 nm, almost all of it the apex offset.

P4 illustrates why the control problem is substantial. Its reference searches a resistance ladder without reading each atom's manipulation threshold, estimates lattice geometry from measured adatom positions, and re-images to follow the result. The 150 documented runs include 100 initial conditioning episodes and 32 runs requiring further conditioning during manipulation. Of the final sites, 147 were the nearest site and three satisfied the stated 0.02 nm near-tie rule. The position error in the table measures the reported location; actual site occupation and neighbour preservation are checked separately. P4's 90th-percentile instrument time was 1,428 s, within its 5,400 s budget.

**Without privileged access.** A scripted P4 baseline in [`stmbench.baselines`](stmbench/baselines/) drives only the interface an MCP agent receives and reads nothing from the simulator. It judges the tip from how its atoms look, chooses the atom from a survey, measures drift on atoms it finds again, and follows the move against fiducial atoms. On seeds 450–599, held out until the code was final, it succeeded on **150/150**, with a median of **984 s** of instrument time against 879 s for the reference. On seeds 0–299, where it was developed, it succeeded on **300/300**. Two earlier held-out rounds, run before the fixes their failures led to, are recorded in the [design record](docs/DESIGN.md). In nearly every run the moved atom had neighbours within the 10 nm that the bystander check covers. This shows that P4 can be solved from observations alone within its budget. It is a script, not an agent result.

### Calibrating what a good measurement must accomplish

The reference protocols connect tolerance design to measured error distributions. A complementary **fixed-answer audit** asks how often one constant set of numerical answers can satisfy the value checks across seeded worlds.

For the recalibrated P3 and P5 tasks, the best fixed answer passed the joint numerical checks on **4.3% of seeds 0–299** in each case. This is a finite-seed audit with acquisition evidence assumed satisfied, not a full-episode success rate or a universal guessing probability. It helps check that seeded task variation makes measurement informative. See the [audit implementation](stmbench/trackB/audit.py) and [calibration history](docs/DESIGN.md).

The current reference command covers P3, P4, and P5, and the scripted baseline covers P4. P1/P2 reference coverage and the P3-repair protocol remain development priorities. Autonomous-agent evaluation can now build on these instrument and scoring foundations; a current-version model leaderboard has not yet been published.

## A recorded experiment in motion

![Annotated replay of an Astra agent repeatedly imaging and manipulating an Fe atom, with a target and neighbouring atoms visible.](docs/assets/p4-homepage/astra-replay.gif)

This historical P4 example shows the experimental loop: inspect, move, image again, adjust, and verify. In the recorded September 2026 mode-H trial, an external Astra agent operated through MAST and satisfied both P4 checks. The trial used the **easy profile, seed 0**, before the current simulator calibration. The animation combines saved scans with post-episode trajectory annotations and compressed timing. It is a process demonstration, separate from the multi-seed reference results above. The [replay record](docs/assets/p4-homepage/README.md) preserves conditions, provenance, comparison cases, and the original development scorecards. It also records a 15 October 2026 development batch in which eight free-tier models, driven through an agent harness on the September code, each ran this episode once: one verified both checks, four verified only neighbour preservation, and three verified neither.

## Place in the SPM research ecosystem

STM-Bench provides a common experimental environment in which different control runtimes and visual reasoning strategies can be studied. The companion projects address complementary parts of scientific instrument autonomy:

| Project | Primary responsibility | Relationship to STM-Bench |
|---|---|---|
| **[STM-Bench](https://github.com/HamsterPark/STM-Bench)** | Simulated microscope, paper-inspired tasks, seeded variation, evidence-aware evaluation, and replay. | Defines the experimental environment and checks the scientific outcome. |
| **[VISTA-SPM](https://github.com/HamsterPark/VISTA-SPM)** | Reusable visual reasoning, calibrated inspection, metrology, and visual evidence archives. | Supplies a complementary approach to interpreting, revisiting, and measuring SPM observations; a natural component for comparative evaluation. |
| **[dsh-spm](https://github.com/HamsterPark/dsh-spm)** | TypeScript SPM domain tools and visual-framework integration for DeepSeek Harness. | Provides another agent/runtime integration layer in which SPM reasoning and control methods can develop. |

VISTA-SPM's emphasis on persistent visual evidence is especially relevant here: an STM experiment repeatedly returns to the same structure under changing scale, drift, and tip conditions. STM-Bench supplies the changing physical scene, recorded measurements, and outcome criteria with which such methods can be investigated. Its standalone instrument interface also makes the benchmark useful independently of these companion projects.

Related scientific tools include [AFS3D](https://github.com/HamsterPark/AFS3D) and [dft-spm-toolkit](https://github.com/HamsterPark/dft-spm-toolkit), which extend the wider research workflow into force-spectroscopy analysis and first-principles/SPM calculations.

## Getting started

Use **Python 3.13 or newer**; CI runs on Python 3.13. From a repository checkout in a virtual environment:

```bash
python -m pip install -e ".[dev,bench]"
python -m stmbench.cli --help
```

Core dependencies are NumPy, SciPy, and PyYAML. The `dev` extra adds pytest; `bench` adds reporting and plotting dependencies. Instrument episodes and reference protocols run without a model API key. An agent client supplies its own model connection when evaluating a model.

### Run a measurement reference

This executes one P5 reference experiment and writes its results:

```bash
python -m stmbench.cli reference --paper P5 --seeds 1 --jobs 1 --json p5-reference.json
```

For multi-seed reference measurements and a fixed-answer audit:

```bash
python -m stmbench.cli reference --paper P3 --seeds 30 --jobs 4 --json p3-reference.json
python -m stmbench.cli reference --paper P4 --seeds 150 --jobs 4 --json p4-reference.json
python -m stmbench.cli reference --paper P5 --seeds 30 --jobs 4 --json p5-reference-30.json
python -m stmbench.cli audit chance --scenario P3 --seeds 300 --jobs 4 --json p3-audit.json
```

For `reference` and `audit`, `--seeds` is a **count**, starting at seed 0 by default; use `--first-seed` to change the starting seed. Record the Git revision and environment with new results, since physics and scoring evolve with the research release.

### Give an MCP agent the instrument

Configure an MCP-capable client to launch this **stdio server**, using the installed environment's Python executable and the repository as its working directory:

```bash
python -m stmbench.cli episode --scenario P4_atom_positioning_cu111 --seed 0 --out runs/p4_s0
```

The command opens one experiment for the connected agent; the agent performs the experiment through the exposed tools. Begin with `get_task`, acquire images and spectra, submit `report_result` calls with evidence, and end with `finish`. Use a fresh output directory for each episode. Task/rule text currently follows the Chinese scenario definitions, while the tool interface includes English descriptions.

After completion, replay the recorded actions and verify the measurements and verdict:

```bash
python -m stmbench.cli verify runs/p4_s0
```

To seat a model at the instrument with these tools and nothing else, `stmbench agent` runs the loop itself — Claude through the anthropic SDK (`pip install -e ".[agent]"`), or any OpenAI-compatible endpoint — and records every turn next to the instrument's ledger:

```bash
python -m stmbench.cli agent --scenario P4_atom_positioning_cu111 --seed 0 --out runs/p4_s0 --model claude-opus-5-5
```

For a person, or an agent that works through a screen, `stmbench console` serves the same instrument as a web page and writes the same ledger; the page never shows the truth or the score:

```bash
python -m stmbench.cli console --scenario P4_atom_positioning_cu111 --seed 0 --out runs/console_p4
```

The [instrument guide](docs/USAGE.md#the-benchmark-instrument) also documents direct Python use through `Instrument`, the available actions, observation formats, and episode files.

### Optional controller interface

The simulator also exposes a local controller TCP interface:

```bash
python -m stmsim serve --profile reference-stm --seed 0 --material "Au(111)"
```

This path lets instrument-control software that speaks the controller's wire protocol drive the simulator. It runs on the host clock (sim time = wall time × `--time-scale`), so host speed changes what a session measures: it is not a judged benchmark condition, and results obtained through it should be labelled with the interface, scenario version, seed and configuration.

## Development and documentation

STM-Bench is an **alpha research release**. Its public source contains the simulator, standalone instrument, scenarios, judge, reference protocols, scripted baselines, analysis modules, and historical replay assets. The immediate research programme is to extend calibrated reference coverage, complete the corral-repair protocol, and evaluate autonomous agents under clearly recorded interface conditions.

Run the checks (tests that need the private calibration data skip themselves):

```bash
python -m pytest tests -q
```

| Resource | What it explains |
|---|---|
| [Usage guide](docs/USAGE.md) | Instrument and MCP operation, references, baselines, audits, and run records. |
| [Physical fidelity card](docs/FIDELITY.md) | Model sources, checks, assumptions, and physical scope. |
| [Design and experimental record](docs/DESIGN.md) | Scientific rationale, calibration decisions, and dated development results; primarily Chinese. |
| [Scenario definitions](stmbench/trackB/scenarios/) | Tasks, hidden distributions, claims, tolerances, and acquisition requirements. |
| [Repository guide](AGENTS.md) | Source map, invariants, and contribution/validation guidance. |
| [Historical replay evidence](docs/assets/p4-homepage/README.md) | Recorded P4 agent demonstrations and their provenance. |

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
