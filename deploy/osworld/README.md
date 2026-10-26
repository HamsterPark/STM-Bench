# STM-Bench in OSWorld-Science: grading an episode outside the machine it ran on

An agent that works through a screen plays the STM-Bench console inside a virtual machine; the
runner copies named files out of the machine when the task ends and scores them on the host. This
directory is the host side. Status: written against the runner's public documentation
(`docs/user_guide/adding-a-task.md`, `docs/developer_guide/adding-a-domain.md`,
`src/osworld_science/evaluators/external.py` in `DiscoAILab/OSWorld-Science`). It has not run inside the runner, and
the Docker image has not been built.

```
virtual machine                                   host
console (its own account) writes the export       runner: result = 4 x vm_file  ->  <submission>/export/
  session.json   actions.jsonl                    evaluator module_metric
  digests.jsonl  report.json                        -> stm_metric.grade_episode(<submission>/export/actions.jsonl, rules)
                                                    -> docker run --rm --network none -v <export>:/export:ro \
                                                         stmbench-grader:<tag> grade /export --scenario <id> --seed <n>
                                                    <- JSON: score, note, ...        returns (score, note)
```

## The task entry

The export directory is written by the console while the episode happens
(`stmbench/instrument/export.py`; the format is fixed). It holds the instrument's log records, the
hashes of every saved frame and spectrum, and no seed, truth or verdict. The task's `evaluator`
collects it into one directory and names `actions.jsonl` as the file `module_metric` hands over;
the adapter reads the other three beside it:

```jsonc
"evaluator": {
  "conj": "weighted", "weights": [1.0], "pass_score": 1.0,
  "func": ["module_metric"],
  "labels": ["[stm] replayed episode"],
  "params": [{"module": "reference_private/<task_id>/stm_metric.py", "func": "grade_episode",
              "file": "export/actions.jsonl", "oracle_dir": "reference_private/<task_id>"}],
  "result": [
    {"type": "vm_file", "path": "<export dir in the guest>/session.json",  "dest": "export/session.json"},
    {"type": "vm_file", "path": "<export dir in the guest>/actions.jsonl", "dest": "export/actions.jsonl"},
    {"type": "vm_file", "path": "<export dir in the guest>/digests.jsonl", "dest": "export/digests.jsonl"},
    {"type": "vm_file", "path": "<export dir in the guest>/report.json",   "dest": "export/report.json"}
  ]
}
```

The keys follow the runner's task format; `osci tasks validate` has not been run on this. The
score is the fraction of the paper's claims the replay verified (1.0 reproduces the paper), so
`pass_score` 1.0 passes exactly the successes. `report.json` is collected for whoever opens the
directory; the grader does not read it, because the records hold every report.

Private tree, `data/<domain>/private/reference_private/<task_id>/` (never enters the machine):

- `stm_metric.py`, a copy of `deploy/osworld/stm_metric.py`: standard library only, Python 3.11
  syntax, loaded by path. It runs the real grader as a separate process, because the runner
  requires only Python 3.11 and brings its own dependencies, while STM-Bench needs Python 3.13
  and numpy 2. It never raises: any failure is
  `(0.0, "grader failed: <why>")`, with a 30 minute limit (`docker kill` on a timeout).
- `gt.json`: `{"scenario": "P4_atom_positioning_cu111", "seed": 17, "grader_image": "stmbench-grader:<tag>"}`.

Without Docker (development), set `STMBENCH_GRADER_CMD` to a command prefix such as
`/path/to/python3.13 -m stmbench.cli` (split like a shell, so forward slashes): the adapter runs
`<prefix> grade <export dir> --scenario <id> --seed <n>`. The interpreter needs the pinned
requirements below and the package importable from the harness's working directory.
`tests/test_grade.py` drives the adapter this way.

## The image

```bash
docker build -f deploy/osworld/grader/Dockerfile -t stmbench-grader:<tag> .   # context = repository root
python -m stmbench.cli grade <export dir> --scenario <id> --seed <n>          # what the image runs
```

`python:3.13-slim`, the pins in `grader/requirements.txt`, `pip install --no-deps .`, a non-root
user, entrypoint `python -m stmbench.cli`; `--build-arg PYTHON_IMAGE=...@sha256:<digest>` fixes the
base image's libraries too. Build it from the commit the console in the machine runs. The image
runs as uid 10001, so the collected files must be readable by others (files 0644, directories
0755). Not verified: the build itself (Docker was not available when this was written).
`tests/test_grade.py` checks that its inputs exist and are not excluded by `.dockerignore`, and
grading from a copy of just the installed files (Python files and declared package data, without
pandas, matplotlib or PySide6) worked once by hand.

## What "consistent" means

The grader replays every `act`, `report` and `diagnosis` record with the scenario and seed in
`gt.json`, skips `refused` and `observe` records (they changed nothing), stops at `finish` or at
the end of the log, and judges the replay as it stands: an agent that ran out of steps is graded
like one that finished. The replay is *consistent* when

1. each action the replay re-ran replied as recorded: every reply field and both instrument-time
   stamps (`t0_s`, `t1_s`) are equal, and the replay refused none of the recorded calls;
2. every line of `digests.jsonl` equals the hash of the replay's frame or spectrum of that name, and
3. every frame and spectrum the replay saved has such a line.

The hashes are of the raw float64 arrays, so consistent means bit-identical measurements. If it
is not, the episode scores 0 and is no success, and the numbers that describe the replay are
withheld (`partial`, `claims_verified`, `instrument_time_s`, `actions` are `null`): the replay was
not the episode the agent saw. The result lists what differed in `replay.result_mismatches` and
`replay.digest_mismatches` (record number and field or file name, or the instrument's refusal; no
replayed values); `grade --no-strict` scores the replay as it stands and still lists them. A
wrong seed shows as inconsistent as soon as the episode saved a frame or a spectrum (tested for
P5). A wrong scenario, or another rules, API or export version, is an error (exit 2), not a
score. The result carries no seed, truth or per-claim detail.

Bit-identity needs the same code and numerics on both sides: the same STM-Bench commit, Python
3.13, and numpy, scipy and PyYAML at the versions in `grader/requirements.txt` (the machine's
environment must use them too). A world reads nothing from the host besides its scenario and rig
profile, which are package data.

Not established: that a replay on another CPU or C library matches. numpy chooses SIMD code
paths per CPU and calls the C library's math functions, and only runs on one machine have been
compared (separate processes, same environment). Before relying on strict grading, grade an
export made in the machine inside the image; hash mismatches there mean the two environments
differ.

What replay cannot show is that the records are every call the agent made. A record taken out of
or added to the middle of the log shifts the time stamps of the actions after it, and a frame or
spectrum added to the end has no hash line, but a call left off the end (or added at the end that
saves nothing) is caught only by the record count in `session.json`, which the same account
writes. Keeping the export out of the agent's reach is the machine's side of this.

## Open, not solved

- How the seed reaches the console's launcher in the machine without appearing in public task
  files. The task JSON's `config` steps and `public/assets` are public, and with the seed anyone
  can replay offline and read the hidden targets.
- The deployment inside the machine: a separate account for the console, permissions on the
  export directory (readable by the runner's file copy, not writable by the agent's account), and
  the private ledger directory.
- Copy `session.json` first when collecting: it counts the log's records, and a count above what
  `actions.jsonl` holds is refused as a log cut short.
