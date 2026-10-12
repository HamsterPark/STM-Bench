# Fidelity card

What the simulated instrument models, where each part comes from, how it is checked, and where
it is known to differ from a real microscope. A result on STM-Bench is a result on this
instrument; this card is what lets a reader judge how far it carries.

Source labels: **fit** — fitted to the maintainer's lab data (the corpus is private; counts and
dates are given); **lit** — a published relation or value; **model** — a modelling assumption
chosen for plausibility, not measured. Checks name the tests in `tests/` unless stated.
Design notes in Chinese: [DESIGN.md](DESIGN.md) §5 and §9.

## Time

| Behaviour | Source | Check | Known differences |
|---|---|---|---|
| Two clocks: hardware-side processes (pulses, pokes, PLL ring-up, approach cycles, spectrum sweeps) on `wall`, scan lines and slow physics on `sim` | model | `test_clock_freeze_and_faults.py` | On the host clock (MAST mode) sim time is wall × `time_scale`, so host speed changes outcomes (`test_determinism_probe.py`, strict xfail) |
| Virtual instrument clock for benchmark episodes: time moves only with actions, by their physical duration | model | `test_determinism_probe.py`, `test_instrument.py` | No time is charged for software round-trips or for looking at data |

## Tip

| Behaviour | Source | Check | Known differences |
|---|---|---|---|
| Multiple apexes (offset, height, weight); height steps quantised to ±n × 235 pm | fit (real multi-apex frames) | `test_physics.py` (ghost offset, step split) | Apex sets are drawn, not grown from a crystal model |
| Starting condition per seed: good / blunt / double / unstable / dirty, never told | model | `test_manipulation_realism.py::test_each_condition_is_what_it_says` | Five classes; real tips are a continuum |
| Radius blur of the surface, in both directions; loop/ADC bandwidth blur along the line (σ ≥ 1 px) | model + fit (10–90 % step width ~1.2 px, 72 rig frames) | `test_physics.py::test_a_blunt_tip_widens_a_feature_in_both_directions` | Gaussian kernel, not a tip-shape convolution. Until 2026-10-13 the radius blur acted along the fast axis only |
| Spontaneous tip changes (Poisson rate λ); a change mid-frame shows as a step between rows | model + fit (row-jump statistics per session give an upper bound on λ) | `test_scenario_faults.py` (mid-scan tip change) | λ is an upper-bound estimate |
| Pulse and poke outcomes (apex re-sampled, blunting, multi-apex, clusters, pits) | model; poke segments 1–3 timing **fit** (168 traces: hardware lag 0.22–0.33 s, indentation delay median 0.48 s) | `test_physics.py::test_tip_pulse_and_poke_change_state_reproducibly` | Poke segment 4 (feedback recovery) is a single observation, uncalibrated |
| Tip DOS (featureless, or an extra peak for a dirty tip) | model | `test_manipulation_realism.py::test_each_condition_is_what_it_says` | Not in dI/dV maps |

## Surface

| Behaviour | Source | Check | Known differences |
|---|---|---|---|
| Terraces and steps (Au median terrace 17 nm), reference-frame tiles | fit (frame index) | `test_physics.py::test_surface_step_height_and_lattice_period`, `test_world_smoke.py` | Steps are straight lines with a wander term |
| Atomic lattices (Au 0.2884, Cu 0.255, Ag 0.289, HOPG 0.246 nm nearest neighbour) | lit | `test_physics.py::test_surface_step_height_and_lattice_period` | — |
| Au(111) herringbone with rotational domains (Voronoi), period and arm angle drawn per seed | lit + model | `test_herringbone.py` | The zig-zag phase drifts linearly along the stripes, so the chevron (elbow) period is not a measurable quantity; P1 does not score it |
| Shockley surface state: 2-D band, energy-dependent lifetime, step scattering (J0 + Struve), point scatterers, T-matrix multiple scattering in corrals | lit (Crommie et al. 1993; standard 2-D scattering) | `test_surface_state.py`, `test_didv_map.py` | Thermal broadening is folded into the lifetime width (Γ_eff = hypot(Γ, 3.5 k_BT)); modulation broadening is not modelled |
| Adatoms on fcc hollows, drawn as a Gaussian bump of 80–140 pm (point tip), which a good tip shows at about 0.6, i.e. 50–90 pm | lit (single Co ≈ 70 pm, Cu ≈ 50 pm on Cu(111), metal tip, 10 mV: arXiv:1912.05292) | `test_physics.py::test_a_good_tip_shows_adatoms_as_tall_as_real_ones` | Height does not depend on bias. Until 2026-10-13 the drawn range was 50–90 pm |
| Lateral manipulation: the atom follows with probability 1/(1+(R/R_th)^8)·1/(1+(v/v_max)^2), one site per hop; pick-up below a second threshold; grip set by the tip's state; each atom's thresholds vary (σ 0.18 log-normal) | model (form) + lit (Eigler & Schweizer 1990 procedure) | `test_manipulation_realism.py`, `test_p4_verification_frame.py` | fcc sites only, so no fcc/hcp alternation during a drag; the instrument reports the number of hops as current spikes, not the current trace of a drag |

## Junction and electronics

| Behaviour | Source | Check | Known differences |
|---|---|---|---|
| Tunnelling current I = ∫ρ_s ρ_t T dE with κ from the barrier φ | lit (Tersoff–Hamann form) | `test_physics.py` (I–z returns its φ) | — |
| Preamplifier ranges, saturation clamp (10.004 nA on the 10 nA range) and desaturation (τ = 0.45 s) | model | `test_world_smoke.py` | τ is not calibrated |
| Current noise: white floor + shot, 1/f, mains, spectral lines, telegraph noise on dirty tips, coarse-step spikes (82–375 pA) | fit (spike amplitudes) + model | none dedicated | Amplitudes are representative, not per-instrument |
| Spectrum dI/dV noise: 3 % of the peak at 5 ms per point, ∝ 1/√(integration) | model | `test_physics.py::test_a_longer_integration_lowers_the_spectrum_noise` | Until 2026-10-13 fixed at 3 % whatever the integration; modulation amplitude does not change it |
| dI/dV map (lock-in during constant-current scans): sample LDOS at the bias energy per pixel × nominal conductance × modulation | model | `test_didv_map.py` | Leaves out the constant-current setpoint effect, modulation broadening and the tip DOS |

## Feedback and scanner

| Behaviour | Source | Check | Known differences |
|---|---|---|---|
| ≥ 2nd-order loop (PI × piezo low-pass 1.5 kHz × resonance 15 kHz, ζ 0.3); excessive gain rings | model (ported from the VIGIL scanner) | `test_physics.py::test_loop_ringing_and_instability_with_gain`, `::test_loop_tracks_step_with_second_order_response` | — |
| Fast-axis hysteresis: loop profile, mean 1.17 % of the axis, edge fraction 0.62 | fit (39 Au frames 80–120 nm) | `test_fidelity.py::test_retrace_offset_reads_back_at_the_real_median_with_the_measured_shape`, `::test_hysteresis_profile_keeps_the_mean_and_rejects_the_closed_loop` | Slow-axis hysteresis is not modelled |
| Creep after a move (logarithmic), thermal drift (3-D, ~1 nm/min lateral on the cold rig), drift compensation that saturates | fit (983 glance intervals: Δz 94 → 26 pm, Δx 0.39 → 0.20 nm) | `test_drift_tracking.py`, `test_drift_mapping.py` | Drift is linear in time between moves |
| Good-tip trace/retrace correlation on 100 nm verify frames ≈ 0.95–0.98 (real frames 0.93–0.98), which sets the bandwidth blur | fit | `test_fidelity.py` (MAST-gated parts) | Unchanged by the 2026-10-13 blur fix (to 1e-3) |
| Fast scanning with a fragile tip scratches the surface | model | none dedicated | Threshold speeds are representative |

## Coarse motion and approach

| Behaviour | Source | Check | Known differences |
|---|---|---|---|
| Coarse steps with spread; lateral steps of ≥ 3 move to a new area | model | `test_world_smoke.py` | No lateral position feedback |
| Pulsed auto-approach; each motor pulse injects a crosstalk spike, so a low setpoint (50 pA) lands falsely when the noise is high (120 pA does not) | fit (spike amplitudes) + model | `test_world_smoke.py`, `test_verdict_rules.py` | Per-cycle travel is an estimate (107 µm over ~128 cycles), capped at 80 % of the fine range |
| Crashes (Z driven into the surface; a lateral coarse move while tunnelling) leave a metastable multi-apex tip, a pit in the surface and the atoms around it gone | model | `test_physics.py::test_crash_leaves_an_unstable_quantised_multi_apex_that_a_shallow_poke_pins` | Pit depth and radius scale with a single severity number |

## qPlus force channel

| Behaviour | Source | Check | Known differences |
|---|---|---|---|
| Δf from a Morse + van der Waals force law, finite amplitude (Giessibl, Gauss–Chebyshev N = 64) | lit | `test_forces.py` (closed forms against the Sader–Jarvis inversion) | No electrostatic force; no CO flexibility or tip relaxation |
| PLL ring-up τ = Q/(π f0), contact damping | lit + model | `test_qplus_zspectr.py` | Constant-Δf imaging is not implemented (feedback is always constant current) |
| Sensor constants k, f0, Q drawn per seed, readable as registered constants (`Instrument.info`) | model | `test_instrument.py` | The registered k is the true one; a real calibration has its own error |

## What this means for scores

- Claims that turn on a modelled quantity with a literature basis (periods, band parameters,
  force-law parameters) are as credible as that relation; claims that turn on modelled
  behaviour (manipulation thresholds, tip-change rates) test operation against this model.
- Tip-diagnosis statements can be scored only since the 2026-10-13 blur fix.
- The simulator's images are cleaner than real ones in places (no thermal noise beyond the Z
  floor, no tip-shape convolution beyond the Gaussian): perception results on it are an upper
  bound for the same perception on real data.
