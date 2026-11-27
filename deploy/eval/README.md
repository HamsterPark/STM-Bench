# deploy/eval: the task tree of the internal evaluation

Builds the `stm` domain of the internal evaluation version: 7 families x 3 worlds = 21 tasks, in the layout of the
OSWorld-Science data tree (`stm/tasks`, `stm/private/reference_private/<id>`, `stm/private/grader_tests`), plus the
private `tasks.json` for the guest image and a MANIFEST fragment. The OSWorld delivery's own builder
(`output/osworld-p3/trackC`) is untouched.

| key | scenario | task ids (`_01` to `_03`) | max_steps | grader tests |
|---|---|---|---|---|
| p1 | `P1_barth1990_au111` | `stm_eval_p1_herringbone` | 250 | `run_tests_p1.py` (hook: no reference yet) |
| p2 | `P2_standing_waves_cu111` | `stm_eval_p2_waves` | 250 | `run_tests_p2.py` (hook: no reference yet) |
| p3 | `P3_corral_cu111` | `stm_eval_p3_corral` | 250 | `run_tests_p3.py` |
| p3_repair | `P3_corral_repair_cu111` | `stm_eval_p3_corral_repair` | 250 | `run_tests_p3_repair.py` |
| p4 | `P4_atom_positioning_cu111` | `stm_eval_p4_atom` | 250 | `run_tests_p4.py` |
| p5 | `P5_sader_jarvis_qplus` | `stm_eval_p5_force` | 250 | `run_tests_p5.py` |
| p7 | `P7_pull_threshold_cu111` | `stm_eval_p7_pull` | 400 | `run_tests_p7.py` |

Every task id starts with `stm_eval_` (for example `stm_eval_p5_force_01`), so that none shares a name with a task of the
OSWorld delivery; the ids still match the launcher's pattern `[a-z][a-z0-9_]{0,47}`.

The table is `families.py` (`FAMILIES`), with the shared text constants (`FRAME`, `DATA_SENTENCE`, `SNAPSHOT`,
`TASK_VERSION`) and each family's two audit texts. The step budgets are provisional until a blind trial fixes them;
`--max-steps p7=500` overrides one.

## Inputs

Public (this directory): `families.py`, `build_tree.py`, `make_tasks_json.py`, `make_fragment.py`, `tree_src/`.

Private (never committed): the **worlds file**, `{"tasks": {<task id>: {"scenario", "seed", "note"}}}`. It must hold
exactly the 21 ids (or the families chosen with `--families`), each with its family's scenario, all seeds distinct.
`note` is free text for the maintainers (starting tip class, what the world was checked for) and goes only into that
task's private README. The repository holds no seed, no world value and no mapping from task id to tip class.

Also needed: a checkout of STM-Bench at the commit under test (`--code`, Python 3.13 with its pins), the adapter
`deploy/osworld/stm_metric.py` (`--adapter`), the grader image tag (`--grader-image`).

## Rebuild

```
python build_tree.py --worlds <worlds.json> --out <root> --code <checkout> \
    --adapter <checkout>/deploy/osworld/stm_metric.py --grader-image stmbench-grader:<12 hex> [--fixtures <dir>]
python make_tasks_json.py <root> <tasks.json>     # mode 0600, holds the seeds
python make_fragment.py <root> <fragment.json>    # after anything is added to the tree
```

The reference exports (the fixtures of the grader tests) are made on khwu, never on the desktop:
`tree_src/grader_tests/reference_export.py --private <root>/stm/private --out fixtures --code <checkout> --tasks <id>`
(add `--variant unrepaired|incomplete` for the repair tasks' two negative episodes), then build again with
`--fixtures`. `STMBENCH_GRADER_CMD="<python 3.13> -m stmbench.cli"` makes the adapter grade without Docker.

## What the builder guarantees

- Every task's instruction is FRAME + `task_text(scenario, "en")` from the code at `--code` + the one data sentence;
  nothing is hand-copied. It stops if the text holds a URL or a seed of 1000 or more.
- `config` (four steps), `evaluator` and `bench.task_version` ("1.0.0") are the same for every task; the snapshot is
  `ubuntu_stm_eval`.
- `gt.json` is written only by each task's `make_gt.py` from the worlds file; `make_tasks_json.py` reads the same
  `gt.json` files, so the guest and the grader cannot disagree.
- It refuses a worlds file with a missing or extra id, a wrong scenario, or two tasks sharing a seed.

It does not guarantee that a world is solvable, that the max_steps suit a model, or that a family's reference
solution exists: P1 and P2 have none yet, and their grader tests are SKIPPED (not passed) until it does.

## Grader tests

`run_tests_<family>.py` holds a family's table of cases; `stm_cases.py` the shared export model, mutations and loop.
Each case is graded the way the runner grades (`score.py <task> --submission <dir> --json`, replaying the whole
episode). Positive: the reference export, a value moved by half its tolerance, an episode without `finish`, a digest
hash differing in its last bits. Negative: another task's export, another seed, a log cut short or with a record
removed or altered, tampered or missing digests, an empty episode, a report without measurements, a value outside
its tolerance, missing or wrong-kind evidence, and the family's own cases (see each file's docstring).
